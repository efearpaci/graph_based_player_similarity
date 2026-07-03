import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path

import networkx as nx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network

import styles
from engine import (FEATURE_GROUPS, GROUP_DESCRIPTIONS, POSITION_GROUP_DEFAULTS,
                    POSITION_GROUP_OPTIONS, assign_archetypes, build_graph_features,
                    build_graphs, find_similar_players, percentile_profile)
from real_data import data_available, load_real_data

# ------------------------------------------------------------------ theme ---
PURPLE = "#a855f7"
FUCHSIA = "#d946ef"
BG = "#0a0812"
TEXT = "#e8e3f5"
MUTED = "#9d93b8"

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=TEXT, family="Inter, sans-serif"),
)

FEATURES_CACHE = Path(__file__).parent / "data" / "processed" / "graph_features.parquet"


# ------------------------------------------------------------------- data ---
@st.cache_data
def load_data():
    return load_real_data()


@st.cache_resource(show_spinner=False)
def load_engine(players_df, pass_df, def_df):
    pass_graphs, def_graphs = build_graphs(pass_df, def_df)

    if FEATURES_CACHE.exists():
        graph_feats = pd.read_parquet(FEATURES_CACHE)
    else:
        bar = st.progress(0.0, "Building passing networks & style embeddings "
                               "(one-time, ~5 min — cached afterwards)")
        graph_feats = build_graph_features(
            pass_graphs, def_graphs,
            progress=lambda f, team: bar.progress(f, f"Embedding {team} …"))
        graph_feats.to_parquet(FEATURES_CACHE, index=False)
        bar.empty()

    features_df = players_df.merge(graph_feats, on="player_name", how="inner")
    features_df = assign_archetypes(features_df)
    return pass_graphs, def_graphs, features_df


# -------------------------------------------------------------------- viz ---
def network_html(G, coords, target=None, height="430px", max_edges=110):
    """Pyvis network, purple/black theme, laid out by real average positions."""
    if len(G.edges) > max_edges:
        keep = sorted(G.edges(data=True), key=lambda e: -e[2]["weight"])[:max_edges]
        H = nx.DiGraph()
        H.add_edges_from(keep)
        G = H

    net = Network(height=height, width="100%", directed=True,
                  bgcolor=BG, font_color=TEXT, cdn_resources="in_line")
    net.toggle_physics(False)

    for node in G.nodes():
        ax, ay = coords.get(node, (50, 50))
        x, y = (ax - 50) * 9, (ay - 50) * 5.5
        if node == target:
            color, size = FUCHSIA, 26
        else:
            color, size = "#6d4fa8", 13
        label = node.split()[-1] if node != target else node
        net.add_node(node, label=label, title=node, color=color,
                     size=size, x=x, y=y,
                     font={"color": TEXT, "size": 12, "face": "Inter"})

    for u, v, d in G.edges(data=True):
        net.add_edge(u, v, value=d["weight"], color="#2e2249",
                     title=f"{u} → {v}: {d['weight']:.2f}")

    html = net.generate_html()
    override = (
        "<style>"
        f"body{{margin:0;background:{BG};}}"
        f".card{{border:none!important;background:{BG}!important;}}"
        f"#mynetwork{{border:none!important;background:{BG}!important;border-radius:16px;}}"
        "</style>"
    )
    html = html.replace("</head>", override + "</head>")
    fit_script = ("<script>network.once('afterDrawing', "
                  "function(){network.fit({animation:false});});</script>")
    return html.replace("</body>", fit_script + "</body>")


def radar_figure(profiles, names):
    fig = go.Figure()
    colors = [PURPLE, FUCHSIA]
    for (profile, name, color) in zip(profiles, names, colors):
        cats = list(profile.keys())
        vals = list(profile.values())
        fig.add_trace(go.Scatterpolar(
            r=vals + vals[:1], theta=cats + cats[:1],
            fill="toself", name=name,
            line=dict(color=color, width=2),
            opacity=0.55,
        ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(255,255,255,0.02)",
            radialaxis=dict(visible=True, range=[0, 100], gridcolor="#241b38",
                            tickfont=dict(color=MUTED, size=10)),
            angularaxis=dict(gridcolor="#241b38", tickfont=dict(size=12)),
        ),
        legend=dict(orientation="h", y=-0.12),
        height=430, margin=dict(l=60, r=60, t=30, b=30),
        **PLOTLY_LAYOUT,
    )
    return fig


def breakdown_figure(breakdown, match_name):
    groups = list(breakdown.keys())
    vals = [max(0.0, breakdown[g].get(match_name, 0.0)) * 100 for g in groups]
    fig = go.Figure(go.Bar(
        x=vals, y=groups, orientation="h",
        marker=dict(color=vals, colorscale=[[0, "#3b2a5e"], [1, FUCHSIA]],
                    cmin=0, cmax=100, line=dict(width=0)),
        text=[f"{v:.0f}%" for v in vals], textposition="outside",
    ))
    fig.update_layout(
        xaxis=dict(range=[0, 112], gridcolor="#241b38",
                   title="similarity within group (%)"),
        yaxis=dict(autorange="reversed"),
        height=90 + 42 * len(groups), margin=dict(l=10, r=10, t=10, b=10),
        **PLOTLY_LAYOUT,
    )
    return fig


# --------------------------------------------------------------------- app ---
st.set_page_config(page_title="ScoutGraph — Player Similarity",
                   page_icon="🔮", layout="wide")
st.markdown(styles.CSS, unsafe_allow_html=True)

if not data_available():
    st.error("Real dataset not found. Run `python download_data.py` and then "
             "`python build_dataset.py --leagues all` first.")
    st.stop()

players_df, pass_df, def_df = load_data()
pass_graphs, def_graphs, features_df = load_engine(players_df, pass_df, def_df)
COORDS = dict(zip(features_df["player_name"],
                  zip(features_df["avg_x"], features_df["avg_y"])))

# ------------------------------------------------------------- sidebar ---
with st.sidebar:
    st.markdown('<p class="section-label">🔮 ScoutGraph</p>', unsafe_allow_html=True)
    st.caption("Wyscout event data · Big-5 European leagues · 2017/18 season")
    st.markdown('<p class="section-label">Scouting target</p>', unsafe_allow_html=True)

    eligible = features_df[features_df["eligible"]]
    display = eligible.sort_values(["league", "team_name", "player_name"])
    labels = [f"{r.player_name}  ·  {r.team_name}" for r in display.itertuples()]
    label_to_name = {l: r.player_name for l, r in zip(labels, display.itertuples())}
    selected_label = st.selectbox(
        "Search a player to replace", labels,
        help=f"{len(eligible):,} players with ≥900 minutes played. Type to search.")
    target_name = label_to_name[selected_label]
    target = features_df[features_df["player_name"] == target_name].iloc[0]

    all_leagues = sorted(features_df["league"].unique())
    leagues = st.multiselect("Search in leagues", all_leagues, default=all_leagues,
                             help="Restrict where replacements may come from.")

    st.divider()
    st.markdown('<p class="section-label">Comparison parameters</p>', unsafe_allow_html=True)
    st.caption(f"Choose what *similar* means for this **{target['position']}**.")

    options = POSITION_GROUP_OPTIONS[target["position"]]
    defaults = POSITION_GROUP_DEFAULTS[target["position"]]

    selected_groups, group_weights = [], {}
    for group in options:
        on = st.checkbox(group, value=group in defaults,
                         help=GROUP_DESCRIPTIONS[group], key=f"chk_{group}")
        if on:
            selected_groups.append(group)

    with st.expander("⚖️ Group weights"):
        st.caption("How much each selected group matters (1.0 = neutral).")
        for group in selected_groups:
            group_weights[group] = st.slider(group, 0.0, 3.0, 1.0, 0.25,
                                             key=f"w_{group}")

    st.divider()
    st.markdown('<p class="section-label">Search settings</p>', unsafe_allow_html=True)
    metric = st.radio("Similarity metric", ["cosine", "euclidean"], horizontal=True,
                      help="Cosine = same *style* regardless of output level. "
                           "Euclidean = same style **and** same output level.")
    top_n = st.slider("Number of matches", 3, 10, 5)

# ---------------------------------------------------------------- header ---
st.markdown(styles.hero(
    "ScoutGraph",
    "Graph-based player similarity on real event data — every pass of the 2017/18 "
    "Premier League, La Liga, Serie A, Bundesliga and Ligue 1 seasons. Pick a player, "
    "choose which dimensions of their game matter, and find their closest structural "
    "and statistical replacements across Europe."
), unsafe_allow_html=True)

st.markdown('<p class="section-label">Scouting target</p>', unsafe_allow_html=True)
st.markdown(styles.profile_card(target), unsafe_allow_html=True)

if not selected_groups:
    st.warning("Select at least one comparison parameter in the sidebar.")
    st.stop()
if not leagues:
    st.warning("Select at least one league to search in.")
    st.stop()

results, breakdown = find_similar_players(
    target_name, features_df, selected_groups, group_weights, metric, top_n,
    leagues=leagues)

# ------------------------------------------------------------ match cards ---
st.markdown(f'<p class="section-label">Top {len(results)} replacements · '
            f'{target["position"]}s only · {len(leagues)} league(s) · '
            f'{len(selected_groups)} parameter groups</p>',
            unsafe_allow_html=True)

cols = st.columns(min(5, len(results)))
for i, (_, row) in enumerate(results.iterrows()):
    with cols[i % len(cols)]:
        st.markdown(styles.match_card(row, i + 1), unsafe_allow_html=True)

st.write("")

# ---------------------------------------------------------------- detail ---
inspect_name = st.selectbox("🔍 Inspect a match in detail",
                            results["player_name"].tolist())
match = results[results["player_name"] == inspect_name].iloc[0]

tab_radar, tab_break, tab_net, tab_stats = st.tabs(
    ["◈ Radar profile", "◈ Why similar?", "◈ Network view", "◈ Full stats"])

with tab_radar:
    c1, c2 = st.columns([1.4, 1])
    with c1:
        prof_t = percentile_profile(features_df, target_name, selected_groups)
        prof_m = percentile_profile(features_df, inspect_name, selected_groups)
        st.plotly_chart(radar_figure([prof_t, prof_m], [target_name, inspect_name]),
                        use_container_width=True)
    with c2:
        st.markdown('<p class="section-label">Reading the radar</p>', unsafe_allow_html=True)
        st.markdown(
            f"Each axis is a **parameter group** you selected. Values are the player's "
            f"average percentile among all eligible **{target['position']}s** in the "
            f"Big-5 leagues — 90 means *top 10% for their position in Europe*.")
        st.metric("Overall similarity", f"{max(0, match['similarity']) * 100:.1f}%")
        st.metric("Age", int(match["age"]), delta=int(match["age"] - target["age"]),
                  delta_color="inverse")
        st.metric("Minutes played", f"{int(match['minutes']):,}",
                  delta=f"{int(match['minutes'] - target['minutes']):+,} vs target")

with tab_break:
    st.markdown('<p class="section-label">Similarity per parameter group</p>',
                unsafe_allow_html=True)
    st.caption(f"Which dimensions make **{inspect_name}** a match for **{target_name}** — "
               "and where they differ.")
    st.plotly_chart(breakdown_figure(breakdown, inspect_name), use_container_width=True)

    strongest = max(breakdown, key=lambda g: breakdown[g].get(inspect_name, -1))
    weakest = min(breakdown, key=lambda g: breakdown[g].get(inspect_name, 2))
    st.info(f"**Scout note:** the match is driven mostly by **{strongest}** "
            f"(near-identical profiles there), while the biggest stylistic gap is in "
            f"**{weakest}**. Archetypes: **{match['archetype']}** vs "
            f"**{target['archetype']}**.")

with tab_net:
    layer = st.radio("Network layer",
                     ["Passing network (xT-weighted)", "Defensive synergy"],
                     horizontal=True)
    graphs = pass_graphs if "Passing" in layer else def_graphs

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f'<p class="section-label">Target · {target_name} '
                    f'({target["team_name"]})</p>', unsafe_allow_html=True)
        components.html(network_html(graphs[target["team_name"]], COORDS,
                                     target=target_name), height=445)
    with c2:
        st.markdown(f'<p class="section-label">Match · {inspect_name} '
                    f'({match["team_name"]})</p>', unsafe_allow_html=True)
        components.html(network_html(graphs[match["team_name"]], COORDS,
                                     target=inspect_name), height=445)
    st.caption("Season networks, strongest links only. Nodes sit at each player's real "
               "average on-ball position (attack →). The highlighted node is the player; "
               "edge thickness = xT / synergy volume. The *shape* of a player's "
               "connections is what the graph features encode.")

with tab_stats:
    stat_cols = []
    for g in selected_groups:
        if g != "Style Embeddings":
            stat_cols.extend(FEATURE_GROUPS[g])
    compare = pd.DataFrame({
        "Metric": stat_cols,
        target_name: [target[c] for c in stat_cols],
        inspect_name: [match[c] for c in stat_cols],
    })
    compare["Δ"] = (compare[inspect_name] - compare[target_name]).round(3)
    st.dataframe(compare, use_container_width=True, hide_index=True, height=520)
    st.caption("Per-90 season values for every metric in the selected groups "
               "(style embeddings are latent vectors and not human-readable).")
