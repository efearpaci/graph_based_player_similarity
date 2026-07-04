"""Validation page — shows the results of evaluation.py (thesis experiments)."""

import json
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

import styles

PURPLE = "#a855f7"
FUCHSIA = "#d946ef"
DIM = "#3b2a5e"
TEXT = "#e8e3f5"
MUTED = "#9d93b8"

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=TEXT, family="Inter, sans-serif"),
)

RESULTS = Path(__file__).parent.parent / "data" / "processed" / "evaluation_results.json"

st.set_page_config(page_title="ScoutGraph — Validation", page_icon="📊", layout="wide")
st.markdown(styles.CSS, unsafe_allow_html=True)

st.markdown(styles.hero(
    "Validation",
    "How do we know the similarity engine works? Three experiments on the 2017/18 "
    "Big-5 season: split-half identity retrieval (Davis et al., 2024), real "
    "substitution pairs as weak labels (Yılmaz & Öğüdücü, 2022), and role "
    "coherence of the neighbourhoods."
), unsafe_allow_html=True)

if not RESULTS.exists():
    st.error("No results yet — run `python evaluation.py` first.")
    st.stop()

res = json.loads(RESULTS.read_text())

# --------------------------------------------- 0. algorithm comparison ---
st.markdown('<p class="section-label">Algorithm comparison at a glance</p>',
            unsafe_allow_html=True)

import pandas as pd

ir_all = res["identity_retrieval"]
rc_all = res["role_coherence"]
sub_all = res["substitution_pairs"].get("configs", {})
summary_rows = []
for cfg in ir_all:
    if cfg.startswith("_"):
        continue
    o = ir_all[cfg]["overall"]
    rc = rc_all.get(cfg, {})
    sb = sub_all.get(cfg, {})
    summary_rows.append({
        "Algorithm / features": cfg,
        "Identity top-1": f"{o['top1']*100:.1f}%",
        "Identity top-5": f"{o['top5']*100:.1f}%",
        "MRR": f"{o['mrr']:.3f}",
        "Median rank": o["median_rank"],
        "Sub hit@1": f"{sb.get('hit@1', 0)*100:.1f}%",
        "Sub hit@3": f"{sb.get('hit@3', 0)*100:.1f}%",
        "Role coherence": f"{rc.get('coherence', 0)*100:.1f}%",
        "Role lift": f"×{rc.get('lift', 0)}",
    })
st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
st.caption(
    "Higher is better everywhere except median rank. Random top-1 ≈ "
    f"{res['identity_retrieval']['_random_baseline']['top1']*100:.2f}% "
    "(position pools of ~140–460 players). Key comparison: **GraphWave beats "
    "Node2Vec on every metric** because its structural signatures are comparable "
    "across team graphs, while independently trained Node2Vec spaces are not. "
    "Tabular stats remain the strongest *identity* signal — structure encodes "
    "*role*, and the two are complementary."
)

st.divider()

# ------------------------------------------------- 1. identity retrieval ---
st.markdown('<p class="section-label">1 · Identity retrieval (split-half reliability)</p>',
            unsafe_allow_html=True)
st.markdown(
    "Each player is represented twice — once from the **first half** of the season, "
    "once from the **second**. A good style fingerprint should retrieve the *same "
    "player* from the other half. **Node2Vec fails by design**: embeddings trained "
    "on different graphs live in unaligned spaces — the argument for combining "
    "tabular stats with *structural* graph features."
)

ir = res["identity_retrieval"]
configs = [c for c in ir if not c.startswith("_")]
baseline = ir.get("_random_baseline", {})

fig = go.Figure()
for metric, color in [("top1", PURPLE), ("top5", FUCHSIA)]:
    fig.add_trace(go.Bar(
        name=f"{metric} accuracy",
        x=[ir[c]["overall"][metric] * 100 for c in configs],
        y=configs, orientation="h", marker_color=color,
        text=[f"{ir[c]['overall'][metric]*100:.1f}%" for c in configs],
        textposition="outside",
    ))
if baseline:
    fig.add_vline(x=baseline["top5"] * 100, line_dash="dot", line_color=MUTED,
                  annotation_text="random top-5", annotation_font_color=MUTED)
fig.update_layout(
    barmode="group", height=340,
    xaxis=dict(range=[0, 108], gridcolor="#241b38", title="accuracy (%)"),
    yaxis=dict(autorange="reversed"),
    legend=dict(orientation="h", y=1.12),
    margin=dict(l=10, r=10, t=10, b=10), **PLOTLY_LAYOUT,
)
st.plotly_chart(fig, use_container_width=True)

with st.expander("Per-position breakdown"):
    import pandas as pd
    rows = []
    for c in configs:
        for pos, m in ir[c]["per_position"].items():
            rows.append({"Config": c, "Position": pos, **m})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ------------------------------------------------- 2. substitution pairs ---
st.markdown('<p class="section-label">2 · Real substitution pairs (held-out)</p>',
            unsafe_allow_html=True)
sub = res["substitution_pairs"]
st.markdown(
    f"Across **{sub['n_pairs']:,}** real same-position substitutions "
    f"({sub['note']}), each algorithm ranks the player who actually came on "
    "among the squad's candidates. The **Triplet metric was trained on earlier "
    "substitutions only** — this is its true test set, and every method is "
    "scored on the same pairs.")

fig_sub = go.Figure()
sub_configs = list(sub["configs"])
for metric, color in [("hit@1", PURPLE), ("hit@3", FUCHSIA)]:
    fig_sub.add_trace(go.Bar(
        name=metric,
        x=[sub["configs"][c][metric] * 100 for c in sub_configs],
        y=sub_configs, orientation="h", marker_color=color,
        text=[f"{sub['configs'][c][metric]*100:.1f}%" for c in sub_configs],
        textposition="outside",
    ))
for base, dash in [("random@1", "dot"), ("random@3", "dash")]:
    fig_sub.add_vline(x=sub[base] * 100, line_dash=dash, line_color=MUTED,
                      annotation_text=base, annotation_font_color=MUTED)
fig_sub.update_layout(
    barmode="group", height=90 + 52 * len(sub_configs),
    xaxis=dict(range=[0, 78], gridcolor="#241b38", title="hit rate (%)"),
    yaxis=dict(autorange="reversed"),
    legend=dict(orientation="h", y=1.1),
    margin=dict(l=10, r=10, t=10, b=10), **PLOTLY_LAYOUT,
)
st.plotly_chart(fig_sub, use_container_width=True)

# ---------------------------------------------------- 3. role coherence ---
st.markdown('<p class="section-label">3 · Role coherence of top-5 neighbours</p>',
            unsafe_allow_html=True)
st.markdown(
    "If similarity captures *role*, a Ball-Winner's neighbours should be "
    "Ball-Winners. Coherence = share of top-5 neighbours with the same refined "
    "role label, vs. the base rate of that label in the position pool.")

rc = res["role_coherence"]
fig2 = go.Figure()
fig2.add_trace(go.Bar(
    name="top-5 coherence",
    x=[rc[c]["coherence"] * 100 for c in rc], y=list(rc),
    orientation="h", marker_color=PURPLE,
    text=[f"{rc[c]['coherence']*100:.0f}%  (×{rc[c]['lift']})" for c in rc],
    textposition="outside",
))
fig2.add_trace(go.Bar(
    name="base rate", x=[rc[c]["base_rate"] * 100 for c in rc], y=list(rc),
    orientation="h", marker_color=DIM,
))
fig2.update_layout(
    barmode="group", height=320,
    xaxis=dict(range=[0, 100], gridcolor="#241b38", title="share of neighbours (%)"),
    yaxis=dict(autorange="reversed"),
    legend=dict(orientation="h", y=1.12),
    margin=dict(l=10, r=10, t=10, b=10), **PLOTLY_LAYOUT,
)
st.plotly_chart(fig2, use_container_width=True)

st.caption(res["meta"]["dataset"] + " · " + res["meta"]["half_split"])
