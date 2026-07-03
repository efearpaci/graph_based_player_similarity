"""
Evaluation of the similarity engine — the thesis results module.

Three experiments, following the evaluation methodology of Davis et al. (2024)
and the substitution-label idea of Yılmaz & Öğüdücü (SAC '22):

1. IDENTITY RETRIEVAL (split-half reliability)
   A player's representation from the first half of the season should retrieve
   the *same player* from the second half. Compared across feature configs:
   tabular / graph topology / Node2Vec / hybrid. Node2Vec is included
   deliberately: embeddings trained on different graphs live in different,
   unaligned spaces, so it is expected to fail — an argument for structural
   (cross-graph comparable) features.

2. SUBSTITUTION PAIRS (weak real-world labels)
   Real in-match substitutions (playerOut -> playerIn) are used as labels of
   interchangeability. For each same-position pair, rank playerIn among the
   squad's same-position candidates by similarity to playerOut.

3. ROLE COHERENCE
   Fraction of a player's top-5 neighbours sharing their refined role label,
   vs. the pool base rate.

Requires:  python build_dataset.py --leagues all --halves
Run:       python evaluation.py            (writes data/processed/evaluation_results.json)
"""

import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
          "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[v] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from build_dataset import LEAGUES, fix_text
from engine import N2V_DIM, build_graph_features, build_graphs
from real_data import _refine_role_label

PROCESSED = Path(__file__).parent / "data" / "processed"
RAW = Path(__file__).parent / "data" / "raw"

TABULAR = ["goals", "shots", "shots_on_target", "dribbles_won",
           "assists", "key_passes", "smart_passes", "crosses",
           "progressive_passes", "passes", "pass_accuracy",
           "tackles_won", "interceptions", "clearances",
           "aerials_won", "aerial_win_rate",
           "saves", "save_rate", "gk_exits"]
TOPOLOGY = ["in_degree", "out_degree", "betweenness", "pagerank",
            "closeness", "clustering", "total_xt_generated",
            "def_pagerank", "def_in_degree", "def_out_degree"]
NODE2VEC = [f"p_n2v_{i}" for i in range(N2V_DIM)] + [f"d_n2v_{i}" for i in range(N2V_DIM)]

CONFIGS = {
    "Tabular (per-90 stats)": TABULAR,
    "Graph topology": TOPOLOGY,
    "Node2Vec (unaligned)": NODE2VEC,
    "Hybrid (tabular + topology)": TABULAR + TOPOLOGY,
}

POSITIONS = ["Goalkeeper", "Defender", "Midfielder", "Forward"]


# ------------------------------------------------------------------ helpers ---
def half_features(suffix):
    """players + graph features for one half-season (cached to parquet)."""
    cache = PROCESSED / f"graph_features{suffix}.parquet"
    players = pd.read_parquet(PROCESSED / f"players{suffix}.parquet")
    if cache.exists():
        gf = pd.read_parquet(cache)
    else:
        pass_df = pd.read_parquet(PROCESSED / f"pass_edges{suffix}.parquet")
        def_df = pd.read_parquet(PROCESSED / f"def_edges{suffix}.parquet")
        pg, dg = build_graphs(pass_df, def_df)
        print(f"  building graph features for {len(pg)} teams [{suffix}] …")
        gf = build_graph_features(pg, dg)
        gf.to_parquet(cache, index=False)
    return players.merge(gf, on="player_name", how="inner")


def full_features():
    players = pd.read_parquet(PROCESSED / "players.parquet")
    gf = pd.read_parquet(PROCESSED / "graph_features.parquet")
    df = players.merge(gf, on="player_name", how="inner")
    df["role_label"] = df.apply(_refine_role_label, axis=1)
    return df


# -------------------------------------------------- 1. identity retrieval ---
def identity_retrieval():
    print("\n[1/3] identity retrieval (split-half) …")
    h1, h2 = half_features("_h1"), half_features("_h2")

    # players who moved between leagues mid-season appear once per league —
    # drop them entirely (their halves describe different teams/leagues and
    # duplicate ids would corrupt the row alignment of the score matrix)
    for df in (h1, h2):
        df.drop_duplicates(subset="player_id", keep=False, inplace=True)

    # players eligible in BOTH halves, matched by player_id
    common = set(h1[h1.eligible].player_id) & set(h2[h2.eligible].player_id)
    h1 = h1[h1.player_id.isin(common)].set_index("player_id").sort_index()
    h2 = h2[h2.player_id.isin(common)].set_index("player_id").sort_index()
    assert h1.index.equals(h2.index)

    results = {}
    for config, cols in CONFIGS.items():
        per_pos = {}
        ranks_all = []
        for pos in POSITIONS:
            ids = h1.index[h1.position == pos]
            if len(ids) < 10:
                continue
            A = StandardScaler().fit_transform(h1.loc[ids, cols].fillna(0))
            B = StandardScaler().fit_transform(h2.loc[ids, cols].fillna(0))
            sims = cosine_similarity(A, B)         # query h1 -> gallery h2
            order = np.argsort(-sims, axis=1)
            ranks = np.array([np.where(order[i] == i)[0][0] + 1
                              for i in range(len(ids))])
            per_pos[pos] = {
                "n": int(len(ids)),
                "top1": round(float((ranks == 1).mean()), 3),
                "top5": round(float((ranks <= 5).mean()), 3),
                "mrr": round(float((1 / ranks).mean()), 3),
                "median_rank": int(np.median(ranks)),
            }
            ranks_all.append(ranks)

        ranks_all = np.concatenate(ranks_all)
        results[config] = {
            "overall": {
                "n": int(len(ranks_all)),
                "top1": round(float((ranks_all == 1).mean()), 3),
                "top5": round(float((ranks_all <= 5).mean()), 3),
                "mrr": round(float((1 / ranks_all).mean()), 3),
                "median_rank": int(np.median(ranks_all)),
            },
            "per_position": per_pos,
        }
        o = results[config]["overall"]
        print(f"  {config:32s} top1={o['top1']:.1%} top5={o['top5']:.1%} "
              f"MRR={o['mrr']:.3f} median={o['median_rank']}")

    # random baseline: expected top-k for pool sizes used
    pool_sizes = [len(h1.index[h1.position == p]) for p in POSITIONS
                  if len(h1.index[h1.position == p]) >= 10]
    results["_random_baseline"] = {
        "top1": round(float(np.mean([1 / n for n in pool_sizes])), 4),
        "top5": round(float(np.mean([min(1, 5 / n) for n in pool_sizes])), 4),
        "pool_sizes": pool_sizes,
    }
    return results


# ------------------------------------------------- 2. substitution pairs ---
def load_substitutions():
    teams_raw = json.load(open(RAW / "teams.json"))
    team_names = {t["wyId"]: fix_text(t["name"]) for t in teams_raw}
    pairs = Counter()
    for country in LEAGUES:
        matches = json.load(open(RAW / "matches" / f"matches_{country}.json"))
        for m in matches:
            for tid, td in m["teamsData"].items():
                f = td.get("formation") or {}
                for s in (f.get("substitutions") or []):
                    if isinstance(s, dict):
                        pairs[(s["playerOut"], s["playerIn"], team_names.get(int(tid)))] += 1
    return pairs


def substitution_eval(features):
    print("\n[2/3] substitution-pair retrieval …")
    pairs = load_substitutions()
    # keep one row per player (mid-season league movers appear twice)
    feats = features.sort_values("minutes", ascending=False) \
                    .drop_duplicates(subset="player_id").set_index("player_id")
    cols = TABULAR + TOPOLOGY

    # scale within position pools once
    scaled = {}
    for pos in POSITIONS:
        pool = feats[feats.position == pos]
        X = StandardScaler().fit_transform(pool[cols].fillna(0))
        scaled[pos] = pd.DataFrame(X, index=pool.index)

    hits1 = hits3 = total = 0
    rand_expect1 = rand_expect3 = 0.0
    skipped_cross_pos = 0
    for (p_out, p_in, team), _cnt in pairs.items():
        if p_out not in feats.index or p_in not in feats.index:
            continue
        out_row, in_row = feats.loc[p_out], feats.loc[p_in]
        if out_row["position"] != in_row["position"]:
            skipped_cross_pos += 1
            continue
        pos = out_row["position"]
        # candidates: same squad, same position, not the leaving player
        cand = feats[(feats.team_name == team) & (feats.position == pos)].index
        cand = [c for c in cand if c != p_out]
        if p_in not in cand or len(cand) < 2:
            continue
        S = scaled[pos]
        sims = cosine_similarity(S.loc[[p_out]], S.loc[cand])[0]
        order = [cand[i] for i in np.argsort(-sims)]
        rank = order.index(p_in) + 1
        hits1 += rank == 1
        hits3 += rank <= 3
        total += 1
        rand_expect1 += 1 / len(cand)
        rand_expect3 += min(1, 3 / len(cand))

    res = {
        "n_pairs": total,
        "skipped_cross_position": skipped_cross_pos,
        "hit@1": round(hits1 / total, 3),
        "hit@3": round(hits3 / total, 3),
        "random@1": round(rand_expect1 / total, 3),
        "random@3": round(rand_expect3 / total, 3),
    }
    print(f"  {total} same-position substitution pairs "
          f"(skipped {skipped_cross_pos} cross-position)")
    print(f"  hit@1={res['hit@1']:.1%} (random {res['random@1']:.1%})   "
          f"hit@3={res['hit@3']:.1%} (random {res['random@3']:.1%})")
    return res


# ---------------------------------------------------- 3. role coherence ---
def role_coherence(features):
    print("\n[3/3] role coherence of top-5 neighbours …")
    out = {}
    for config, cols in CONFIGS.items():
        agree, base, n = [], [], 0
        for pos in POSITIONS:
            pool = features[(features.position == pos) & features.eligible]
            if len(pool) < 20:
                continue
            X = StandardScaler().fit_transform(pool[cols].fillna(0))
            sims = cosine_similarity(X)
            np.fill_diagonal(sims, -np.inf)
            top5 = np.argsort(-sims, axis=1)[:, :5]
            roles = pool["role_label"].values
            share = Counter(roles)
            for i in range(len(pool)):
                agree.append(np.mean(roles[top5[i]] == roles[i]))
                base.append((share[roles[i]] - 1) / (len(pool) - 1))
                n += 1
        out[config] = {
            "coherence": round(float(np.mean(agree)), 3),
            "base_rate": round(float(np.mean(base)), 3),
            "lift": round(float(np.mean(agree) / np.mean(base)), 2),
            "n": n,
        }
        print(f"  {config:32s} coherence={out[config]['coherence']:.1%} "
              f"(base {out[config]['base_rate']:.1%}, lift ×{out[config]['lift']})")
    return out


if __name__ == "__main__":
    features = full_features()
    results = {
        "identity_retrieval": identity_retrieval(),
        "substitution_pairs": substitution_eval(features),
        "role_coherence": role_coherence(features),
        "meta": {
            "dataset": "Wyscout/Pappalardo Big-5 leagues 2017/18",
            "half_split": "first vs second half of season by match date, "
                          "eligibility ≥450 min per half",
        },
    }
    out_path = PROCESSED / "evaluation_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n✓ results written to {out_path}")
