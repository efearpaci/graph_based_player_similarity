"""
GraphWave embedding-size sweep (deterministic — no seeds needed).

Sweeps the number of characteristic-function sample points (n_points ∈ 2/4/8
-> 8/16/32 dims per graph, x2 graphs) and scores each size on:
  - identity retrieval (split-half, GraphWave-only config)
  - role coherence (GraphWave-only)
  - substitution hit@3 on the VALIDATION fold (GraphWave-only)

Precedent: Yılmaz & Öğüdücü (SAC '22) found larger GraphWave embeddings
helped. If a size clearly wins here, update GW_POINTS in engine.py and
rebuild the feature caches.

Run:  python tune_graphwave.py
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

from engine import build_graphs, graphwave_embeddings
from evaluation import POSITIONS, build_sub_trials, dedupe_by_id, sub_hit_rates
from real_data import _refine_role_label

PROCESSED = Path(__file__).parent / "data" / "processed"
N_POINTS = [2, 4, 8]


def gw_frame(suffix, n_points):
    """players + GraphWave embeddings (pass & def graphs) for one dataset."""
    players = pd.read_parquet(PROCESSED / f"players{suffix}.parquet")
    pass_df = pd.read_parquet(PROCESSED / f"pass_edges{suffix}.parquet")
    def_df = pd.read_parquet(PROCESSED / f"def_edges{suffix}.parquet")
    pg, dg = build_graphs(pass_df, def_df)

    dim = 2 * n_points * 2
    rows = {}
    for team, G in pg.items():
        emb_p = graphwave_embeddings(G, n_points=n_points)
        emb_d = graphwave_embeddings(dg[team], n_points=n_points) \
            if len(dg[team]) else {}
        for node, v in emb_p.items():
            d = emb_d.get(node, np.zeros(dim))
            rows[node] = np.concatenate([v, d])

    cols = [f"g_{i}" for i in range(2 * dim)]
    gw = pd.DataFrame.from_dict(rows, orient="index", columns=cols)
    gw.index.name = "player_name"
    return players.merge(gw.reset_index(), on="player_name", how="inner"), cols


def identity_top5(h1, h2, cols):
    for df in (h1, h2):
        df.drop_duplicates(subset="player_id", keep=False, inplace=True)
    common = set(h1[h1.eligible].player_id) & set(h2[h2.eligible].player_id)
    a = h1[h1.player_id.isin(common)].set_index("player_id").sort_index()
    b = h2[h2.player_id.isin(common)].set_index("player_id").sort_index()
    ranks_all = []
    for pos in POSITIONS:
        ids = a.index[a.position == pos]
        if len(ids) < 10:
            continue
        A = StandardScaler().fit_transform(a.loc[ids, cols].fillna(0))
        B = StandardScaler().fit_transform(b.loc[ids, cols].fillna(0))
        order = np.argsort(-cosine_similarity(A, B), axis=1)
        ranks_all.append(np.array(
            [np.where(order[i] == i)[0][0] + 1 for i in range(len(ids))]))
    ranks = np.concatenate(ranks_all)
    return float((ranks <= 5).mean())


def coherence(full, cols):
    agree, base = [], []
    for pos in POSITIONS:
        pool = full[(full.position == pos) & full.eligible]
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
    return float(np.mean(agree)), float(np.mean(agree) / np.mean(base))


def main():
    split = json.loads((PROCESSED / "substitution_split.json").read_text())
    val_pairs = [(s["out"], s["in"], s["team"]) for s in split["val"]]

    results = []
    for np_ in N_POINTS:
        full, cols = gw_frame("", np_)
        full["role_label"] = full.apply(_refine_role_label, axis=1)
        h1, _ = gw_frame("_h1", np_)
        h2, _ = gw_frame("_h2", np_)

        top5 = identity_top5(h1, h2, cols)
        coh, lift = coherence(full, cols)

        feats = dedupe_by_id(full)
        trials, _, _ = build_sub_trials(feats, val_pairs)
        v1, v3 = sub_hit_rates(feats, cols, trials)

        rec = {"n_points": np_, "dims_total": len(cols),
               "identity_top5": round(top5, 4),
               "role_coherence": round(coh, 4), "role_lift": round(lift, 2),
               "val_sub_hit1": round(v1, 4), "val_sub_hit3": round(v3, 4)}
        results.append(rec)
        print(f"n_points={np_} ({len(cols)} dims)  identity top5 {top5:.1%}  "
              f"coherence {coh:.1%} (×{lift:.2f})  val subs {v1:.1%}/{v3:.1%}")

    (PROCESSED / "graphwave_sweep.json").write_text(json.dumps(results, indent=1))
    best = max(results, key=lambda r: (r["role_lift"], r["identity_top5"]))
    print(f"\n✓ graphwave_sweep.json written — best by role lift: "
          f"n_points={best['n_points']}")


if __name__ == "__main__":
    main()
