"""
Learn per-position feature-group weights for the engineered similarity engine.

The app's KNN weights every feature group by a scout-chosen slider (default
1.0). Here we OPTIMISE those weights per position against real substitutions:
random search + coordinate ascent over a weight grid, maximising hit@3
(tie-break hit@1) on the VALIDATION fold only. The untouched test fold is
scored exactly once at the end, next to the all-ones baseline.

Output: data/processed/learned_weights.json — the app uses these as slider
defaults (scouts can still override; the sliders stay).

Run after train_metric.py --sweep/--final (needs substitution_split.json):
    python learn_weights.py
"""

import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
          "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[v] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import random
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from engine import FEATURE_GROUPS, POSITION_GROUP_OPTIONS
from evaluation import build_sub_trials, dedupe_by_id, full_features

PROCESSED = Path(__file__).parent / "data" / "processed"

SEED = 42
WEIGHT_GRID = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0]
RANDOM_ITERS = 400
ASCENT_PASSES = 2

# engineered groups only: the learned Triplet metric is a competing method,
# and Node2Vec is kept solely as a negative baseline
EXCLUDED = {"Learned Similarity (Triplet)", "Style Embeddings (Node2Vec)"}


def position_groups(pos):
    return [g for g in POSITION_GROUP_OPTIONS[pos] if g not in EXCLUDED]


def score(Xs, groups_cols, w_by_group, trials, row_of):
    """hit@1/hit@3 with per-group weights (w / sqrt(group size) per column)."""
    col_w = np.concatenate([
        np.full(len(cols), w_by_group[g] / np.sqrt(len(cols)))
        for g, cols in groups_cols.items()])
    if not col_w.any():
        return 0.0, 0.0
    Xw = Xs * col_w
    hits1 = hits3 = 0
    for p_out, p_in, _, cand in trials:
        a = Xw[row_of[p_out]][None, :]
        C = Xw[[row_of[c] for c in cand]]
        sims = cosine_similarity(a, C)[0]
        order = np.argsort(-sims)
        rank = [cand[i] for i in order].index(p_in) + 1
        hits1 += rank == 1
        hits3 += rank <= 3
    n = max(1, len(trials))
    return hits1 / n, hits3 / n


def optimise_position(pos, feats, val_trials, rng):
    groups = position_groups(pos)
    groups_cols = {g: FEATURE_GROUPS[g] for g in groups}
    all_cols = [c for cols in groups_cols.values() for c in cols]

    pool = feats[feats.position == pos]
    Xs = StandardScaler().fit_transform(pool[all_cols].fillna(0))
    row_of = {pid: i for i, pid in enumerate(pool.index)}
    trials = [t for t in val_trials if t[2] == pos]
    if len(trials) < 30:
        print(f"  {pos}: only {len(trials)} val trials — keeping all-ones")
        return {g: 1.0 for g in groups}, None

    def fitness(w):
        h1, h3 = score(Xs, groups_cols, w, trials, row_of)
        return (h3, h1)

    baseline = {g: 1.0 for g in groups}
    best_w, best_fit = baseline, fitness(baseline)

    # random search
    for _ in range(RANDOM_ITERS):
        w = {g: rng.choice(WEIGHT_GRID) for g in groups}
        if not any(w.values()):
            continue
        f = fitness(w)
        if f > best_fit:
            best_w, best_fit = w, f

    # coordinate ascent refinement
    for _ in range(ASCENT_PASSES):
        for g in groups:
            for v in WEIGHT_GRID:
                w = {**best_w, g: v}
                if not any(w.values()):
                    continue
                f = fitness(w)
                if f > best_fit:
                    best_w, best_fit = w, f

    base_fit = fitness(baseline)
    print(f"  {pos:11s} ({len(trials)} val trials)  "
          f"baseline hit@3 {base_fit[0]:.1%} -> learned {best_fit[0]:.1%}   "
          f"weights {dict(sorted(best_w.items()))}")
    return best_w, {"val_baseline_hit3": round(base_fit[0], 4),
                    "val_learned_hit3": round(best_fit[0], 4),
                    "n_val_trials": len(trials)}


def main():
    rng = random.Random(SEED)
    feats = dedupe_by_id(full_features())
    split = json.loads((PROCESSED / "substitution_split.json").read_text())
    if "val" not in split:
        raise SystemExit("no validation fold — run train_metric.py --sweep first")

    to_pairs = lambda fold: [(s["out"], s["in"], s["team"]) for s in split[fold]]
    val_trials, _, _ = build_sub_trials(feats, to_pairs("val"))
    test_trials, _, _ = build_sub_trials(feats, to_pairs("test"))

    learned, diagnostics = {}, {}
    for pos in ["Goalkeeper", "Defender", "Midfielder", "Forward"]:
        w, diag = optimise_position(pos, feats, val_trials, rng)
        learned[pos] = w
        if diag:
            diagnostics[pos] = diag

    # ---- one-shot test comparison: all-ones vs learned, same trials ----
    print("\ntest fold (touched once):")
    test_res = {}
    for pos in learned:
        groups_cols = {g: FEATURE_GROUPS[g] for g in position_groups(pos)}
        all_cols = [c for cols in groups_cols.values() for c in cols]
        pool = feats[feats.position == pos]
        Xs = StandardScaler().fit_transform(pool[all_cols].fillna(0))
        row_of = {pid: i for i, pid in enumerate(pool.index)}
        trials = [t for t in test_trials if t[2] == pos]
        if not trials:
            continue
        b1, b3 = score(Xs, groups_cols, {g: 1.0 for g in groups_cols}, trials, row_of)
        l1, l3 = score(Xs, groups_cols, learned[pos], trials, row_of)
        test_res[pos] = {"baseline": {"hit@1": round(b1, 4), "hit@3": round(b3, 4)},
                         "learned": {"hit@1": round(l1, 4), "hit@3": round(l3, 4)},
                         "n_trials": len(trials)}
        print(f"  {pos:11s} baseline {b1:.1%}/{b3:.1%} -> learned {l1:.1%}/{l3:.1%} "
              f"({len(trials)} trials)")

    out = {"weights": learned, "validation": diagnostics, "test": test_res,
           "objective": "val hit@3 (tie-break hit@1), random search + "
                        "coordinate ascent, engineered groups only"}
    (PROCESSED / "learned_weights.json").write_text(json.dumps(out, indent=1))
    print("\n✓ learned_weights.json written — app will use these as slider defaults")


if __name__ == "__main__":
    main()
