"""
Deep metric learning: triplet network on real substitution pairs, with a
leakage-safe protocol and hyperparameter tuning.

Data protocol (three-way split BY MATCH DATE — the leakage gate):
    train 55%  |  validation 15%  |  test 30%
  - hyperparameters and model selection use ONLY train -> validation
  - the test fold (identical to the original 70/30 test boundary) is scored
    exactly once, by evaluation.py, on the shipped artifact
  - substitution_split.json stores all three folds

Negative mining (the knob that matters most):
  - "random": negatives sampled uniformly (same position, prefer same squad)
  - "semi-hard": negatives that the current model places close to the anchor —
    farther than the positive but within the margin (fallback: hardest) —
    recomputed every epoch from the evolving embedding space

Usage:
    python train_metric.py --sweep            # grid over mining/margin/dim, 3 seeds,
                                              # selected on validation -> tuning_results.json
    python train_metric.py --variance         # 5-seed val variance of the best config
    python train_metric.py --final            # train best config on train+val, export
                                              # embeddings (full/_h1/_h2) + model
    python train_metric.py                    # --final with default config if no sweep ran
"""

import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
          "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[v] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import itertools
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from build_dataset import LEAGUES, fix_text
from engine import FEATURE_GROUPS

PROCESSED = Path(__file__).parent / "data" / "processed"
RAW = Path(__file__).parent / "data" / "raw"

TRAIN_FRAC, VAL_FRAC = 0.55, 0.15          # test = remaining 0.30 (same boundary as before)
EPOCHS = 60
LR = 1e-3
HIDDEN = 128
NEG_PER_PAIR = {"random": 4, "semi-hard": 2}   # quality over quantity when mining
MINE_CANDIDATES = 25

DEFAULT_CONFIG = {"mining": "semi-hard", "margin": 0.2, "dim": 32}
SWEEP_GRID = {
    "mining": ["random", "semi-hard"],
    "margin": [0.1, 0.2, 0.3],
    "dim": [16, 32, 64],
}

INPUT_COLS = (FEATURE_GROUPS["Attacking"] + FEATURE_GROUPS["Creation & Passing"]
              + FEATURE_GROUPS["Defending"] + FEATURE_GROUPS["Physique"]
              + FEATURE_GROUPS["Goalkeeping"] + FEATURE_GROUPS["Passing Network"]
              + FEATURE_GROUPS["Defensive Synergy"]
              + FEATURE_GROUPS["Structural Embeddings (GraphWave)"])


class Encoder(nn.Module):
    def __init__(self, n_in, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, HIDDEN), nn.ReLU(),
                                 nn.Linear(HIDDEN, dim))

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


# ------------------------------------------------------------------ data ---
def load_features(suffix=""):
    players = pd.read_parquet(PROCESSED / f"players{suffix}.parquet")
    gf = pd.read_parquet(PROCESSED / f"graph_features{suffix}.parquet")
    df = players.merge(gf, on="player_name", how="inner")
    return df.sort_values("minutes", ascending=False) \
             .drop_duplicates(subset="player_id")


def load_substitutions_with_dates():
    teams_raw = json.load(open(RAW / "teams.json"))
    team_names = {t["wyId"]: fix_text(t["name"]) for t in teams_raw}
    subs = []
    for country in LEAGUES:
        for m in json.load(open(RAW / "matches" / f"matches_{country}.json")):
            for tid, td in m["teamsData"].items():
                f = td.get("formation") or {}
                for s in (f.get("substitutions") or []):
                    if isinstance(s, dict):
                        subs.append({"out": s["playerOut"], "in": s["playerIn"],
                                     "team": team_names.get(int(tid)),
                                     "date": m["dateutc"]})
    return subs


def make_split(feats):
    subs = load_substitutions_with_dates()
    row_ok = set(feats.index)
    valid = [s for s in subs
             if s["out"] in row_ok and s["in"] in row_ok
             and feats.loc[s["out"], "position"] == feats.loc[s["in"], "position"]]
    valid.sort(key=lambda s: s["date"])
    n = len(valid)
    i1, i2 = int(n * TRAIN_FRAC), int(n * (TRAIN_FRAC + VAL_FRAC))
    split = {
        "train": valid[:i1], "val": valid[i1:i2], "test": valid[i2:],
        "cutoff_date": valid[i2]["date"],           # test boundary (unchanged)
        "val_cutoff_date": valid[i1]["date"],
    }
    (PROCESSED / "substitution_split.json").write_text(json.dumps(split, indent=1))
    print(f"split: {len(split['train'])} train / {len(split['val'])} val "
          f"/ {len(split['test'])} test  (test starts {split['cutoff_date'][:10]})")
    return split


# ------------------------------------------------------------------ train ---
def train_one(config, seed, pairs, feats, Xs, row_of, quiet=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    by_pos = defaultdict(list)
    by_team_pos = defaultdict(list)
    for pid in feats.index:
        pos = feats.loc[pid, "position"]
        by_pos[pos].append(pid)
        by_team_pos[(feats.loc[pid, "team_name"], pos)].append(pid)

    # fixed candidate pool per pair: squad mates + random same-position players
    cand_pool = []
    for s in pairs:
        pos = feats.loc[s["out"], "position"]
        squad = [p for p in by_team_pos[(feats.loc[s["out"], "team_name"], pos)]
                 if p not in (s["out"], s["in"])]
        extra = [p for p in random.sample(by_pos[pos],
                                          min(MINE_CANDIDATES, len(by_pos[pos])))
                 if p not in (s["out"], s["in"])]
        cand_pool.append(list(dict.fromkeys(squad + extra)))

    model = Encoder(len(INPUT_COLS), config["dim"])
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.TripletMarginLoss(margin=config["margin"])
    Xt = torch.from_numpy(Xs)
    k = NEG_PER_PAIR[config["mining"]]

    for epoch in range(EPOCHS):
        # ---- build triplets for this epoch ----
        if config["mining"] == "semi-hard":
            model.eval()
            with torch.no_grad():
                E = model(Xt).numpy()
            a_idx, p_idx, n_idx = [], [], []
            for s, cands in zip(pairs, cand_pool):
                ia, ip = row_of[s["out"]], row_of[s["in"]]
                d_ap = np.linalg.norm(E[ia] - E[ip])
                ic = np.array([row_of[c] for c in cands])
                d_an = np.linalg.norm(E[ic] - E[ia], axis=1)
                semi = np.where((d_an > d_ap) & (d_an < d_ap + config["margin"]))[0]
                chosen = (semi[np.argsort(d_an[semi])][:k] if len(semi)
                          else np.argsort(d_an)[:k])       # fallback: hardest
                for j in chosen:
                    a_idx.append(ia); p_idx.append(ip); n_idx.append(ic[j])
        else:
            a_idx, p_idx, n_idx = [], [], []
            for s, cands in zip(pairs, cand_pool):
                for c in random.sample(cands, min(k, len(cands))):
                    a_idx.append(row_of[s["out"]])
                    p_idx.append(row_of[s["in"]])
                    n_idx.append(row_of[c])

        perm = np.random.permutation(len(a_idx))
        a_idx = np.array(a_idx)[perm]
        p_idx = np.array(p_idx)[perm]
        n_idx = np.array(n_idx)[perm]

        model.train()
        total = 0.0
        for i in range(0, len(a_idx), 512):
            sl = slice(i, i + 512)
            opt.zero_grad()
            loss = loss_fn(model(Xt[a_idx[sl]]), model(Xt[p_idx[sl]]),
                           model(Xt[n_idx[sl]]))
            loss.backward()
            opt.step()
            total += loss.item() * len(a_idx[sl])
        if not quiet and (epoch % 20 == 0 or epoch == EPOCHS - 1):
            print(f"    epoch {epoch:3d}  loss {total / len(a_idx):.4f}")
    return model


def val_score(model, feats, Xs, val_pairs):
    """hit@1/hit@3 on validation pairs using the model's embedding space."""
    from evaluation import build_sub_trials, sub_hit_rates
    model.eval()
    with torch.no_grad():
        E = model(torch.from_numpy(Xs)).numpy()
    dim = E.shape[1]
    cols = [f"tr_{i}" for i in range(dim)]
    emb = feats[["team_name", "position", "minutes"]].copy()
    emb[cols] = E
    trials, _, _ = build_sub_trials(
        emb, [(s["out"], s["in"], s["team"]) for s in val_pairs])
    return sub_hit_rates(emb, cols, trials)


# ------------------------------------------------------------------ modes ---
def prepare():
    feats = load_features().set_index("player_id")
    X = feats[INPUT_COLS].fillna(0).values.astype(np.float32)
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd < 1e-8] = 1.0
    Xs = (X - mu) / sd
    row_of = {pid: i for i, pid in enumerate(feats.index)}
    split_path = PROCESSED / "substitution_split.json"
    if split_path.exists():
        split = json.loads(split_path.read_text())
        if "val" not in split:                 # old 2-way split -> rebuild
            split = make_split(feats)
    else:
        split = make_split(feats)
    return feats, X, mu, sd, Xs, row_of, split


def sweep(seeds):
    feats, X, mu, sd, Xs, row_of, split = prepare()
    results = []
    combos = [dict(zip(SWEEP_GRID, v))
              for v in itertools.product(*SWEEP_GRID.values())]
    print(f"sweep: {len(combos)} configs × {seeds} seeds")
    for cfg in combos:
        h1s, h3s = [], []
        for seed in range(seeds):
            model = train_one(cfg, seed, split["train"], feats, Xs, row_of,
                              quiet=True)
            h1, h3 = val_score(model, feats, Xs, split["val"])
            h1s.append(h1)
            h3s.append(h3)
        rec = {**cfg,
               "val_hit1_mean": round(float(np.mean(h1s)), 4),
               "val_hit1_std": round(float(np.std(h1s)), 4),
               "val_hit3_mean": round(float(np.mean(h3s)), 4),
               "val_hit3_std": round(float(np.std(h3s)), 4)}
        results.append(rec)
        print(f"  {cfg['mining']:9s} m={cfg['margin']} d={cfg['dim']:3d}  "
              f"val hit@1 {rec['val_hit1_mean']:.1%}±{rec['val_hit1_std']:.1%}  "
              f"hit@3 {rec['val_hit3_mean']:.1%}±{rec['val_hit3_std']:.1%}")

    results.sort(key=lambda r: (r["val_hit3_mean"], r["val_hit1_mean"]),
                 reverse=True)
    best = {k: results[0][k] for k in ("mining", "margin", "dim")}
    out = {"grid": results, "best": best, "seeds": seeds,
           "selection": "val_hit3_mean (tie-break val_hit1_mean)"}
    (PROCESSED / "tuning_results.json").write_text(json.dumps(out, indent=1))
    print(f"\n✓ best config: {best} -> tuning_results.json")


def variance(seeds):
    feats, X, mu, sd, Xs, row_of, split = prepare()
    cfg = best_config()
    print(f"variance of {cfg} over {seeds} seeds (train -> val)")
    h1s, h3s = [], []
    for seed in range(seeds):
        model = train_one(cfg, seed, split["train"], feats, Xs, row_of, quiet=True)
        h1, h3 = val_score(model, feats, Xs, split["val"])
        h1s.append(h1)
        h3s.append(h3)
        print(f"  seed {seed}: hit@1 {h1:.1%}  hit@3 {h3:.1%}")
    print(f"  mean±std  hit@1 {np.mean(h1s):.1%}±{np.std(h1s):.1%}  "
          f"hit@3 {np.mean(h3s):.1%}±{np.std(h3s):.1%}")
    path = PROCESSED / "tuning_results.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    data["variance"] = {"seeds": seeds,
                        "val_hit1_mean": round(float(np.mean(h1s)), 4),
                        "val_hit1_std": round(float(np.std(h1s)), 4),
                        "val_hit3_mean": round(float(np.mean(h3s)), 4),
                        "val_hit3_std": round(float(np.std(h3s)), 4)}
    path.write_text(json.dumps(data, indent=1))


def best_config():
    path = PROCESSED / "tuning_results.json"
    if path.exists():
        return json.loads(path.read_text())["best"]
    return DEFAULT_CONFIG


def final():
    feats, X, mu, sd, Xs, row_of, split = prepare()
    cfg = best_config()
    print(f"final training: {cfg} on train+val "
          f"({len(split['train']) + len(split['val'])} pairs), seed 42")
    model = train_one(cfg, 42, split["train"] + split["val"], feats, Xs, row_of)

    model.eval()

    def export(suffix):
        df = load_features(suffix)
        Z = (df[INPUT_COLS].fillna(0).values.astype(np.float32) - mu) / sd
        with torch.no_grad():
            emb = model(torch.from_numpy(Z)).numpy()
        out = pd.DataFrame(emb, columns=[f"tr_{i}" for i in range(cfg["dim"])])
        out.insert(0, "player_name", df["player_name"].values)
        out.to_parquet(PROCESSED / f"triplet_embeddings{suffix}.parquet", index=False)
        return len(out)

    for sfx in ("", "_h1", "_h2"):
        n = export(sfx)
        print(f"  wrote triplet_embeddings{sfx}.parquet ({n} players)")

    torch.save({"state_dict": model.state_dict(), "mu": mu, "sd": sd,
                "input_cols": INPUT_COLS, "embed_dim": cfg["dim"],
                "config": cfg},
               PROCESSED / "triplet_model.pt")
    print("✓ done — run evaluation.py for the one-shot test-fold numbers")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--variance", action="store_true")
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    if args.sweep:
        sweep(args.seeds)
    elif args.variance:
        variance(max(args.seeds, 5))
    else:
        final()
