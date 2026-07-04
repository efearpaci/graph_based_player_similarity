"""
Similarity engine: graph feature extraction + scout-configurable KNN search.

The scout chooses which FEATURE GROUPS to compare on (and how much each group
matters). Similarity is always computed within the target's positional pool.

Works on the real Wyscout 2017/18 dataset (see build_dataset.py); all tabular
stats are per-90 values from a full season of event data.
"""

import networkx as nx
import numpy as np
import pandas as pd
from node2vec import Node2Vec
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import StandardScaler

N2V_DIM = 8
GW_SCALES = 2       # GraphWave heat-kernel scales (auto-selected per graph)
GW_POINTS = 4       # characteristic-function sample points per scale
GW_DIM = GW_SCALES * GW_POINTS * 2  # (Re, Im) per sample point

# ---------------------------------------------------------------- groups ---
FEATURE_GROUPS = {
    "Attacking": ["goals", "shots", "shots_on_target", "dribbles_won"],
    "Creation & Passing": ["assists", "key_passes", "smart_passes", "crosses",
                           "progressive_passes", "passes", "pass_accuracy"],
    "Defending": ["tackles_won", "interceptions", "clearances",
                  "aerials_won", "aerial_win_rate"],
    "Physique": ["height", "weight"],
    "Goalkeeping": ["saves", "save_rate", "gk_exits"],
    "Passing Network": ["in_degree", "out_degree", "betweenness", "pagerank",
                        "closeness", "clustering", "total_xt_generated"],
    "Defensive Synergy": ["def_pagerank", "def_in_degree", "def_out_degree"],
    "Structural Embeddings (GraphWave)":
        [f"gw_p_{i}" for i in range(GW_DIM)] + [f"gw_d_{i}" for i in range(GW_DIM)],
    "Style Embeddings (Node2Vec)":
        [f"p_n2v_{i}" for i in range(N2V_DIM)] + [f"d_n2v_{i}" for i in range(N2V_DIM)],
    "Learned Similarity (Triplet)": [f"tr_{i}" for i in range(32)],
}

GROUP_DESCRIPTIONS = {
    "Attacking": "Goals, shots, on-target %, dribbles won (per 90)",
    "Creation & Passing": "Assists, key/smart passes, crosses, progression (per 90)",
    "Defending": "Tackles, interceptions, clearances, aerials (per 90)",
    "Physique": "Height and weight",
    "Goalkeeping": "Saves, save rate, sweeping exits (per 90)",
    "Passing Network": "Graph centralities from the team's season passing network",
    "Defensive Synergy": "Influence in the defensive co-action network",
    "Structural Embeddings (GraphWave)":
        "Heat-wavelet structural role signatures — comparable across teams",
    "Style Embeddings (Node2Vec)":
        "Random-walk embeddings — NOT aligned across teams (kept for comparison)",
    "Learned Similarity (Triplet)":
        "Deep metric learning: distance trained on real substitution pairs",
}

POSITION_GROUP_DEFAULTS = {
    "Goalkeeper": ["Goalkeeping", "Passing Network", "Defensive Synergy",
                   "Structural Embeddings (GraphWave)"],
    "Defender": ["Defending", "Physique", "Passing Network", "Defensive Synergy",
                 "Structural Embeddings (GraphWave)"],
    "Midfielder": ["Creation & Passing", "Attacking", "Defending",
                   "Passing Network", "Structural Embeddings (GraphWave)"],
    "Forward": ["Attacking", "Creation & Passing", "Physique",
                "Passing Network", "Structural Embeddings (GraphWave)"],
}

POSITION_GROUP_OPTIONS = {
    "Goalkeeper": ["Goalkeeping", "Defending", "Physique", "Creation & Passing",
                   "Passing Network", "Defensive Synergy",
                   "Structural Embeddings (GraphWave)", "Style Embeddings (Node2Vec)",
                   "Learned Similarity (Triplet)"],
    "Defender": [g for g in FEATURE_GROUPS if g != "Goalkeeping"],
    "Midfielder": [g for g in FEATURE_GROUPS if g != "Goalkeeping"],
    "Forward": [g for g in FEATURE_GROUPS if g != "Goalkeeping"],
}


# ---------------------------------------------------------- graph features ---
def build_graphs(pass_df, def_df):
    """One passing graph (xT-weighted) and one defensive graph per team."""
    pass_graphs, def_graphs = {}, {}
    for team in pass_df["team_name"].unique():
        G = nx.DiGraph()
        for row in pass_df[pass_df["team_name"] == team].itertuples():
            G.add_edge(row.passer_name, row.receiver_name,
                       weight=max(row.total_xt, 1e-4), passes=row.pass_count)
        pass_graphs[team] = G

        D = nx.DiGraph()
        for row in def_df[def_df["team_name"] == team].itertuples():
            D.add_edge(row.passer_name, row.receiver_name,
                       weight=row.synergy_score)
        def_graphs[team] = D
    return pass_graphs, def_graphs


def graphwave_embeddings(G, n_points=GW_POINTS, etas=(0.95, 0.80)):
    """
    GraphWave (Donnat et al., KDD 2018): structural node embeddings from the
    empirical characteristic function of spectral heat-wavelet coefficients.

    Unlike Node2Vec, these are deterministic functions of a node's structural
    role, so embeddings from DIFFERENT graphs live in the same space — which is
    what cross-team player comparison requires.

    Returns {node: vector of len(etas) * n_points * 2}.
    """
    nodes = list(G.nodes())
    n = len(nodes)
    dim = len(etas) * n_points * 2
    if n < 3:
        return {v: np.zeros(dim) for v in nodes}

    # symmetric, scale-normalised adjacency (teams have different weight scales)
    A = nx.to_numpy_array(G, nodelist=nodes, weight="weight")
    A = (A + A.T) / 2.0
    nz = A[A > 0]
    if nz.size:
        A = A / nz.mean()
    L = np.diag(A.sum(axis=1)) - A

    lam, U = np.linalg.eigh(L)
    lam = np.clip(lam, 0.0, None)
    pos = lam[lam > 1e-8]
    if pos.size == 0:
        return {v: np.zeros(dim) for v in nodes}

    # automatic scale selection around the spectrum's geometric mean
    s_ref = np.sqrt(pos[0] * pos[-1])
    taus = [-np.log(eta) / s_ref for eta in etas]
    t_pts = np.linspace(0, 100, n_points + 1)[1:]   # skip t=0 (constant)

    feats = np.empty((n, dim))
    k = 0
    for s in taus:
        Psi = U @ np.diag(np.exp(-s * lam)) @ U.T   # column a = wavelet at node a
        for t in t_pts:
            phi = np.exp(1j * t * Psi).mean(axis=0)  # characteristic fn per node
            feats[:, k] = phi.real
            feats[:, k + 1] = phi.imag
            k += 2
    return {v: feats[i] for i, v in enumerate(nodes)}


def build_graph_features(pass_graphs, def_graphs, progress=None):
    """Centralities + dual Node2Vec embeddings for every player."""
    rows = []
    teams = list(pass_graphs)
    for t_i, team in enumerate(teams):
        if progress:
            progress((t_i + 1) / len(teams), team)
        G = pass_graphs[team]
        D = def_graphs[team]

        in_degree = nx.in_degree_centrality(G)
        out_degree = nx.out_degree_centrality(G)
        betweenness = nx.betweenness_centrality(G, weight="weight")
        pagerank = nx.pagerank(G, weight="weight")
        closeness = nx.closeness_centrality(G, distance="weight")
        clustering = nx.clustering(G.to_undirected(), weight="weight")
        total_xt = dict(G.out_degree(weight="weight"))

        def_pr = nx.pagerank(D, weight="weight") if len(D) else {}
        def_in = nx.in_degree_centrality(D) if len(D) else {}
        def_out = nx.out_degree_centrality(D) if len(D) else {}

        n2v_pass = Node2Vec(G, dimensions=N2V_DIM, walk_length=10, num_walks=40,
                            workers=1, quiet=True)
        model_pass = n2v_pass.fit(window=5, min_count=1, batch_words=4)
        model_def = None
        if len(D):
            n2v_def = Node2Vec(D, dimensions=N2V_DIM, walk_length=10, num_walks=40,
                               workers=1, quiet=True)
            model_def = n2v_def.fit(window=5, min_count=1, batch_words=4)

        gw_pass = graphwave_embeddings(G)
        gw_def = graphwave_embeddings(D) if len(D) else {}

        for node in G.nodes():
            row = {
                "player_name": node,
                "in_degree": in_degree[node],
                "out_degree": out_degree[node],
                "betweenness": betweenness[node],
                "pagerank": pagerank[node],
                "closeness": closeness[node],
                "clustering": clustering[node],
                "total_xt_generated": total_xt.get(node, 0),
                "def_pagerank": def_pr.get(node, 0),
                "def_in_degree": def_in.get(node, 0),
                "def_out_degree": def_out.get(node, 0),
            }
            for i, v in enumerate(model_pass.wv[node]):
                row[f"p_n2v_{i}"] = float(v)
            for i in range(N2V_DIM):
                has = model_def is not None and node in model_def.wv
                row[f"d_n2v_{i}"] = float(model_def.wv[node][i]) if has else 0.0
            for i, v in enumerate(gw_pass[node]):
                row[f"gw_p_{i}"] = float(v)
            gw_d = gw_def.get(node)
            for i in range(GW_DIM):
                row[f"gw_d_{i}"] = float(gw_d[i]) if gw_d is not None else 0.0
            rows.append(row)

    return pd.DataFrame(rows)


# ------------------------------------------------------------- archetypes ---
def assign_archetypes(df):
    """Interpretable archetypes from percentile ranks within each position."""
    df = df.copy()

    def pctl(pool, col):
        return pool[col].rank(pct=True)

    df["archetype"] = "—"
    for pos, pool_idx in df.groupby("position").groups.items():
        pool = df.loc[pool_idx]
        p = {c: pctl(pool, c) for c in
             ["passes", "progressive_passes", "key_passes", "crosses", "assists",
              "goals", "aerials_won", "tackles_won", "dribbles_won", "gk_exits"]}

        for idx in pool_idx:
            role = df.at[idx, "role_label"]
            if pos == "Goalkeeper":
                a = "Sweeper Keeper" if (p["passes"][idx] > 0.6 or p["gk_exits"][idx] > 0.7) \
                    else "Shot Stopper"
            elif pos == "Defender":
                if role == "Full-Back":
                    a = "Attacking Full-Back" if (p["crosses"][idx] + p["assists"][idx]) / 2 > 0.55 \
                        else "Defensive Full-Back"
                else:
                    a = "Ball-Playing Defender" if p["progressive_passes"][idx] > 0.6 else "Stopper"
            elif pos == "Midfielder":
                if role == "Defensive Mid":
                    a = "Deep-Lying Playmaker" if p["progressive_passes"][idx] > 0.6 else "Anchor"
                elif role == "Attacking Mid":
                    a = "Classic No.10" if p["key_passes"][idx] > 0.6 else "Shadow Striker"
                else:
                    if p["tackles_won"][idx] > 0.65:
                        a = "Ball-Winner"
                    elif p["key_passes"][idx] > 0.6:
                        a = "Tempo Controller"
                    else:
                        a = "Box-to-Box Engine"
            else:  # Forward
                if role == "Winger":
                    a = "Inverted Winger" if p["goals"][idx] > 0.6 else "Wide Playmaker"
                else:
                    if p["aerials_won"][idx] > 0.7:
                        a = "Target Man"
                    elif p["goals"][idx] > 0.7:
                        a = "Poacher"
                    else:
                        a = "Complete Forward"
            df.at[idx, "archetype"] = a
    return df


# -------------------------------------------------------------- similarity ---
def find_similar_players(target_name, features_df, selected_groups,
                         group_weights=None, metric="cosine", top_n=5,
                         leagues=None):
    """
    KNN search inside the target's positional pool (eligible players only),
    using the columns of the selected feature groups weighted by the scout.

    Returns (results_df, per_group_similarity: dict[group -> dict]).
    """
    group_weights = group_weights or {}
    target = features_df[features_df["player_name"] == target_name].iloc[0]

    pool = features_df[features_df["position"] == target["position"]]
    if "eligible" in pool.columns:
        pool = pool[pool["eligible"] | (pool["player_name"] == target_name)]
    if leagues:
        pool = pool[pool["league"].isin(leagues) | (pool["player_name"] == target_name)]
    pool = pool.copy()

    cols, col_weights = [], []
    for group in selected_groups:
        gcols = FEATURE_GROUPS[group]
        w = group_weights.get(group, 1.0)
        per_col = w / np.sqrt(len(gcols))
        cols.extend(gcols)
        col_weights.extend([per_col] * len(gcols))

    X = StandardScaler().fit_transform(pool[cols].fillna(0))
    Xw = X * np.array(col_weights)

    t_loc = pool.index.get_loc(target.name)
    if metric == "cosine":
        scores = cosine_similarity(Xw)[t_loc]
    else:
        dists = euclidean_distances(Xw)[t_loc]
        scores = 1.0 / (1.0 + dists / np.sqrt(len(cols)))

    pool["similarity"] = scores
    results = pool[pool["player_name"] != target_name] \
        .sort_values("similarity", ascending=False).head(top_n)

    breakdown = {}
    scaled = pd.DataFrame(X, columns=cols, index=pool.index)
    for group in selected_groups:
        gcols = FEATURE_GROUPS[group]
        g_target = scaled.loc[target.name, gcols].values.reshape(1, -1)
        sims = {}
        for idx in results.index:
            g_match = scaled.loc[idx, gcols].values.reshape(1, -1)
            sims[results.loc[idx, "player_name"]] = float(
                cosine_similarity(g_target, g_match)[0][0])
        breakdown[group] = sims

    return results, breakdown


def percentile_profile(features_df, player_name, groups):
    """Mean percentile (0-100) within the positional pool per feature group."""
    target = features_df[features_df["player_name"] == player_name].iloc[0]
    pool = features_df[features_df["position"] == target["position"]]
    if "eligible" in pool.columns:
        pool = pool[pool["eligible"]]

    profile = {}
    for group in groups:
        gcols = FEATURE_GROUPS[group]
        pcts = [(pool[c] <= target[c]).mean() * 100 for c in gcols]
        profile[group] = float(np.mean(pcts))
    return profile
