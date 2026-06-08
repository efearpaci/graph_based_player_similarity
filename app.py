import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import pandas as pd
import networkx as nx
import numpy as np
from node2vec import Node2Vec
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from pyvis.network import Network
import streamlit.components.v1 as components
import plotly.graph_objects as go
import random

# --- 1. MOCK DATA GENERATION ---
@st.cache_data
def generate_mock_data():
    teams = {
        "Man City": "Possession", "Arsenal": "Possession", "Liverpool": "High Press", 
        "Aston Villa": "Balanced", "Tottenham": "High Press", "Chelsea": "Possession", 
        "Newcastle": "Balanced", "Man United": "Direct", "West Ham": "Direct", 
        "Crystal Palace": "Direct", "Brighton": "Possession", "Bournemouth": "Balanced", 
        "Fulham": "Balanced", "Wolves": "Direct", "Everton": "Direct", 
        "Brentford": "Direct", "Nottm Forest": "Direct", "Luton": "Direct", 
        "Burnley": "Possession", "Sheff Utd": "Direct"
    }
    roles = ["GK", "CB1", "CB2", "LB", "RB", "DM", "CM", "AM", "LW", "RW", "ST"]
    
    pass_data = []
    def_data = []
    stats_data = []
    
    base_logic = {
        "GK": [("CB1", 10), ("CB2", 10), ("LB", 5), ("RB", 5), ("ST", 2)],
        "CB1": [("CB2", 15), ("LB", 10), ("DM", 15), ("GK", 5), ("CM", 5)],
        "CB2": [("CB1", 15), ("RB", 10), ("DM", 15), ("GK", 5), ("CM", 5)],
        "LB": [("CB1", 10), ("LW", 15), ("DM", 5), ("CM", 10)],
        "RB": [("CB2", 10), ("RW", 15), ("DM", 5), ("CM", 10)],
        "DM": [("CM", 20), ("AM", 15), ("LB", 10), ("RB", 10), ("LW", 5), ("RW", 5)],
        "CM": [("AM", 20), ("LW", 15), ("RW", 15), ("DM", 15), ("ST", 10)],
        "AM": [("ST", 25), ("LW", 15), ("RW", 15), ("CM", 10)],
        "LW": [("ST", 15), ("AM", 10), ("CM", 5), ("LB", 10)],
        "RW": [("ST", 15), ("AM", 10), ("CM", 5), ("RB", 10)],
        "ST": [("AM", 5), ("LW", 5), ("RW", 5)] 
    }
    
    def_logic = {
        "GK": [("CB1", 5), ("CB2", 5)],
        "CB1": [("CB2", 15), ("LB", 10), ("DM", 15), ("GK", 5)],
        "CB2": [("CB1", 15), ("RB", 10), ("DM", 15), ("GK", 5)],
        "LB": [("CB1", 10), ("LW", 15), ("DM", 5)],
        "RB": [("CB2", 10), ("RW", 15), ("DM", 5)],
        "DM": [("CM", 10), ("CB1", 15), ("CB2", 15), ("LB", 10), ("RB", 10)],
        "CM": [("AM", 10), ("DM", 15), ("LW", 10), ("RW", 10)],
        "AM": [("ST", 15), ("CM", 10)],
        "LW": [("ST", 5), ("LB", 15), ("CM", 5)],
        "RW": [("ST", 5), ("RB", 15), ("CM", 5)],
        "ST": [("AM", 15), ("LW", 5), ("RW", 5)] 
    }
    
    season_multiplier = 10 
    
    for team, style in teams.items():
        players = {role: f"{team}_{role}" for role in roles}
        
        # A. GENERATE PASSING NETWORK
        for passer_role, targets in base_logic.items():
            passer_name = players[passer_role]
            for target_role, base_weight in targets:
                target_name = players[target_role]
                weight = base_weight * season_multiplier
                xt_multiplier = 0.01
                
                if style == "Possession":
                    if passer_role in ["CB1", "CB2", "DM", "CM", "LB", "RB"] and target_role in ["CB1", "CB2", "DM", "CM", "LB", "RB"]:
                        weight *= 1.8 
                elif style == "Direct":
                    if passer_role in ["GK", "CB1", "CB2"] and target_role in ["ST", "LW", "RW"]:
                        weight *= 3.0 
                        xt_multiplier = 0.05
                    if passer_role in ["DM", "CM"] and target_role in ["DM", "CM"]:
                        weight *= 0.5 
                elif style == "High Press":
                    if passer_role in ["LW", "RW", "ST"] and target_role in ["AM", "LW", "RW", "ST"]:
                        weight *= 1.5
                        
                if target_role in ["ST", "LW", "RW"] and passer_role not in ["ST", "LW", "RW"]:
                    xt_multiplier += 0.03
                
                final_weight = max(1, int(weight * random.uniform(0.7, 1.3)))
                total_xt = final_weight * xt_multiplier * random.uniform(0.8, 1.2)
                
                pass_data.append({
                    "passer_name": passer_name,
                    "receiver_name": target_name,
                    "team_name": team,
                    "pass_count": final_weight,
                    "total_xt": total_xt
                })
                
        # B. GENERATE DEFENSIVE SYNERGY NETWORK
        for def_role, targets in def_logic.items():
            passer_name = players[def_role]
            for target_role, base_weight in targets:
                target_name = players[target_role]
                weight = base_weight * season_multiplier
                
                if style == "High Press" and def_role in ["ST", "LW", "RW", "AM"]:
                    weight *= 2.0
                elif style == "Possession" and def_role in ["CB1", "CB2", "GK"]:
                    weight *= 1.5
                    
                final_weight = max(1, int(weight * random.uniform(0.5, 1.5)))
                
                def_data.append({
                    "passer_name": passer_name,
                    "receiver_name": target_name,
                    "team_name": team,
                    "synergy_score": final_weight
                })
                
        # C. GENERATE TABULAR STATS
        for role in roles:
            player_name = players[role]
            
            speed, shooting, dribbling, defending, gk_saves, gk_reflexes = 40, 40, 40, 40, 10, 10
            
            if "GK" in role:
                broad_pos = "Goalkeeper"
                gk_saves = random.randint(70, 95)
                gk_reflexes = random.randint(70, 95)
                defending = random.randint(30, 50)
            elif "CB" in role or "LB" in role or "RB" in role:
                broad_pos = "Defender"
                defending = random.randint(75, 95)
                if "B" in role and "C" not in role: # Fullbacks
                    speed = random.randint(75, 95)
                    dribbling = random.randint(65, 80)
            elif "DM" in role or "CM" in role or "AM" in role:
                broad_pos = "Midfielder"
                dribbling = random.randint(75, 90)
                if "DM" in role: defending = random.randint(70, 85)
                if "AM" in role: shooting = random.randint(70, 85)
            else: # LW, RW, ST
                broad_pos = "Forward"
                shooting = random.randint(75, 95)
                speed = random.randint(75, 95)
                dribbling = random.randint(75, 95)
                
            speed += random.randint(-5, 5)
            shooting += random.randint(-5, 5)
            dribbling += random.randint(-5, 5)
            defending += random.randint(-5, 5)
            
            stats_data.append({
                "player_name": player_name,
                "team_name": team,
                "broad_position": broad_pos,
                "speed": max(1, min(99, speed)),
                "shooting": max(1, min(99, shooting)),
                "dribbling": max(1, min(99, dribbling)),
                "defending": max(1, min(99, defending)),
                "gk_saves": max(1, min(99, gk_saves)),
                "gk_reflexes": max(1, min(99, gk_reflexes))
            })
            
    return pd.DataFrame(pass_data), pd.DataFrame(def_data), pd.DataFrame(stats_data)

# --- 2. MULTIPLEX GRAPH CONSTRUCTION & FEATURE EXTRACTION ---
@st.cache_resource
def process_graphs_and_features(pass_df, def_df, stats_df):
    teams = pass_df['team_name'].unique()
    graph_features_list = []
    graphs = {}
    def_graphs = {}
    
    for team in teams:
        team_df = pass_df[pass_df['team_name'] == team]
        team_def_df = def_df[def_df['team_name'] == team]
        
        # 1. Passing Graph (xT Weighted)
        G = nx.DiGraph()
        for _, row in team_df.iterrows():
            G.add_edge(row['passer_name'], row['receiver_name'], weight=row['total_xt'])
        graphs[team] = G
        
        # 2. Defensive Synergy Graph
        D_G = nx.DiGraph()
        for _, row in team_def_df.iterrows():
            D_G.add_edge(row['passer_name'], row['receiver_name'], weight=row['synergy_score'])
        def_graphs[team] = D_G
        
        # Calculate Advanced Centralities
        in_degree = nx.in_degree_centrality(G)
        out_degree = nx.out_degree_centrality(G)
        betweenness = nx.betweenness_centrality(G, weight='weight')
        pagerank = nx.pagerank(G, weight='weight')
        
        # New Deep Dive Metrics
        total_xt_generated = dict(G.out_degree(weight='weight'))
        def_pagerank = nx.pagerank(D_G, weight='weight')
        
        # Convert to undirected for clustering coefficient
        undirected_G = G.to_undirected()
        clustering = nx.clustering(undirected_G, weight='weight')
        closeness = nx.closeness_centrality(G, distance='weight')
        
        # Dual Node2Vec Architecture
        n2v_pass = Node2Vec(G, dimensions=8, walk_length=10, num_walks=50, workers=1, quiet=True)
        model_pass = n2v_pass.fit(window=5, min_count=1, batch_words=4)
        
        n2v_def = Node2Vec(D_G, dimensions=8, walk_length=10, num_walks=50, workers=1, quiet=True)
        model_def = n2v_def.fit(window=5, min_count=1, batch_words=4)
        
        for node in G.nodes():
            n2v_p_emb = model_pass.wv[node]
            n2v_d_emb = model_def.wv[node]
            
            features = {
                "player_name": node,
                "in_degree": in_degree[node],
                "out_degree": out_degree[node],
                "betweenness": betweenness[node],
                "pagerank": pagerank[node],
                "clustering": clustering[node],
                "closeness": closeness[node],
                "total_xt_generated": total_xt_generated.get(node, 0),
                "def_pagerank": def_pagerank.get(node, 0)
            }
            # Append Passing Embeddings
            for i, val in enumerate(n2v_p_emb):
                features[f"p_n2v_{i}"] = val
            # Append Defensive Embeddings
            for i, val in enumerate(n2v_d_emb):
                features[f"d_n2v_{i}"] = val
                
            graph_features_list.append(features)
            
    graph_features_df = pd.DataFrame(graph_features_list)
    final_features_df = pd.merge(stats_df, graph_features_df, on="player_name")
    
    # Archetype Clustering
    exclude_cols = ["player_name", "team_name", "broad_position"]
    numeric_cols = [c for c in final_features_df.columns if c not in exclude_cols]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(final_features_df[numeric_cols])
    
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kmeans = KMeans(n_clusters=6, random_state=42, n_init="auto")
        final_features_df['cluster'] = kmeans.fit_predict(X_scaled)
    
    archetype_names = {
        0: "Deep-Lying Playmaker",
        1: "Advanced Target",
        2: "Ball-Playing Defender",
        3: "Sweeper Keeper",
        4: "Box-to-Box Engine",
        5: "Wide Threat"
    }
    final_features_df['archetype'] = final_features_df['cluster'].map(archetype_names)
    
    return graphs, def_graphs, final_features_df

# --- 3. DYNAMIC SIMILARITY ENGINE ---
def find_similar_players(target_player, features_df, top_n=3):
    target_data = features_df[features_df["player_name"] == target_player].iloc[0]
    target_broad_pos = target_data["broad_position"]
    
    filtered_df = features_df[features_df["broad_position"] == target_broad_pos].copy()
    
    exclude_cols = ["player_name", "team_name", "broad_position", "cluster", "archetype"]
    numeric_cols = [c for c in filtered_df.columns if c not in exclude_cols]
    
    X = filtered_df[numeric_cols].copy()
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=numeric_cols, index=filtered_df.index)
    
    tactical_cols = ["in_degree", "out_degree", "betweenness", "pagerank", "clustering", "closeness", "total_xt_generated", "def_pagerank"] + [c for c in numeric_cols if 'n2v' in c]
    
    # Give a standard unified boost to tactical columns so it prefers network shape
    for col in tactical_cols:
        X_scaled_df[col] = X_scaled_df[col] * 2.0
        
    if target_broad_pos == "Goalkeeper":
        X_scaled_df["gk_saves"] *= 2.0
        X_scaled_df["gk_reflexes"] *= 2.0
        
    similarity_matrix = cosine_similarity(X_scaled_df)
    
    target_idx_loc = filtered_df.index.get_loc(target_data.name)
    
    filtered_df["similarity_score"] = similarity_matrix[target_idx_loc]
    
    results = filtered_df[filtered_df["player_name"] != target_player]
    results = results.sort_values("similarity_score", ascending=False).head(top_n)
    
    return results

# --- 4. FENERBAHÇE UI VISUALIZATIONS ---
def plot_pyvis_graph(G, target_player=None, height="450px"):
    net = Network(height=height, width="100%", directed=True, bgcolor="#0E1117", font_color="white")
    net.toggle_physics(False) # Lock to pitch
    
    role_coords = {
        "GK": (-400, 0), "CB1": (-250, -100), "CB2": (-250, 100),
        "LB": (-150, -250), "RB": (-150, 250), "DM": (-50, 0),
        "CM": (50, -100), "AM": (150, 0), "LW": (250, -200),
        "RW": (250, 200), "ST": (350, 0)
    }
    
    for node in G.nodes():
        color = "#FF4B4B" if node == target_player else "#1F77B4"
        size = 25 if node == target_player else 15
        in_deg = dict(G.in_degree(weight='weight')).get(node, 0)
        out_deg = dict(G.out_degree(weight='weight')).get(node, 0)
        title = f"Player: {node}\\nIn-Volume: {in_deg:.2f}\\nOut-Volume: {out_deg:.2f}"
        label = node.split("_")[1] if "_" in node else node
        
        x, y = role_coords.get(label, (0,0))
        net.add_node(node, label=label, title=title, color=color, size=size, x=x, y=y)
        
    for u, v, data in G.edges(data=True):
        weight = data['weight']
        net.add_edge(u, v, value=weight, title=f"Weight: {weight:.2f}", color="#4b5563")
        
    path = "html_files"
    if not os.path.exists(path):
        os.makedirs(path)
    file_path = f"{path}/network.html"
    net.save_graph(file_path)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        html_data = f.read()
    return html_data

def plot_ego_graph(G, target_player, height="300px"):
    ego_G = nx.ego_graph(G, target_player, radius=1)
    
    net = Network(height=height, width="100%", directed=True, bgcolor="#0E1117", font_color="white")
    net.toggle_physics(False)
    
    role_coords = {
        "GK": (-400, 0), "CB1": (-250, -100), "CB2": (-250, 100),
        "LB": (-150, -250), "RB": (-150, 250), "DM": (-50, 0),
        "CM": (50, -100), "AM": (150, 0), "LW": (250, -200),
        "RW": (250, 200), "ST": (350, 0)
    }
    
    for node in ego_G.nodes():
        color = "#FF4B4B" if node == target_player else "#1F77B4"
        size = 25 if node == target_player else 15
        label = node.split("_")[1] if "_" in node else node
        
        x, y = role_coords.get(label, (0,0))
        net.add_node(node, label=label, color=color, size=size, x=x, y=y)
        
    for u, v, data in ego_G.edges(data=True):
        weight = data['weight']
        net.add_edge(u, v, value=weight, color="#4b5563")
        
    path = "html_files"
    if not os.path.exists(path):
        os.makedirs(path)
    file_path = f"{path}/ego_{target_player}.html"
    net.save_graph(file_path)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def plot_radar_chart(target_data, match_data):
    pos = target_data['broad_position'].values[0]
    
    if pos == "Goalkeeper":
        categories = ['In-Degree', 'Out-Degree', 'Clustering', 'GK Saves', 'GK Reflexes']
        target_vals = [
            target_data['in_degree'].values[0],
            target_data['out_degree'].values[0],
            target_data['clustering'].values[0],
            target_data['gk_saves'].values[0] / 100.0,
            target_data['gk_reflexes'].values[0] / 100.0
        ]
        match_vals = [
            match_data['in_degree'].values[0],
            match_data['out_degree'].values[0],
            match_data['clustering'].values[0],
            match_data['gk_saves'].values[0] / 100.0,
            match_data['gk_reflexes'].values[0] / 100.0
        ]
    else:
        categories = ['In-Degree', 'Out-Degree', 'Clustering', 'Speed', 'Shooting', 'Dribbling', 'Defending']
        target_vals = [
            target_data['in_degree'].values[0],
            target_data['out_degree'].values[0],
            target_data['clustering'].values[0],
            target_data['speed'].values[0] / 100.0,
            target_data['shooting'].values[0] / 100.0,
            target_data['dribbling'].values[0] / 100.0,
            target_data['defending'].values[0] / 100.0
        ]
        match_vals = [
            match_data['in_degree'].values[0],
            match_data['out_degree'].values[0],
            match_data['clustering'].values[0],
            match_data['speed'].values[0] / 100.0,
            match_data['shooting'].values[0] / 100.0,
            match_data['dribbling'].values[0] / 100.0,
            match_data['defending'].values[0] / 100.0
        ]
        
    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=target_vals,
        theta=categories,
        fill='toself',
        name=f"Target: {target_data['player_name'].values[0]}",
        line_color="#FF4B4B",
        opacity=0.6
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=match_vals,
        theta=categories,
        fill='toself',
        name=f"Match: {match_data['player_name'].values[0]}",
        line_color="#1F77B4",
        opacity=0.8
    ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        margin=dict(l=40, r=40, t=20, b=20),
        height=350,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color="white")
    )
    return fig

def plot_stat_comparison_bar(target_val, match_val, target_name, match_name, title):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=[target_name], x=[target_val],
        name=target_name, orientation='h',
        marker=dict(color="#FF4B4B")
    ))
    fig.add_trace(go.Bar(
        y=[match_name], x=[match_val],
        name=match_name, orientation='h',
        marker=dict(color="#1F77B4")
    ))
    fig.update_layout(
        title=title,
        barmode='group',
        height=200,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color="white"),
        xaxis=dict(range=[0, 1])
    )
    return fig

def generate_scout_summary(target_data, match_data):
    target_name = target_data['player_name'].values[0]
    match_name = match_data['player_name'].values[0]
    
    t_in = target_data['in_degree'].values[0]
    m_in = match_data['in_degree'].values[0]
    
    if abs(t_in - m_in) < 0.05:
        in_desc = "receives passes at a very similar volume"
    elif m_in > t_in:
        in_desc = "is more heavily relied upon to receive passes"
    else:
        in_desc = "receives passes less frequently"
        
    return f"**Scout Report:** {match_name} {in_desc} compared to {target_name}. Both players share the **{target_data['archetype'].values[0]}** archetype."

# --- 5. STREAMLIT APP ---
st.set_page_config(page_title="Graph Recruitment", layout="wide")

st.title("⚽ Advanced Player Similarity (Multiplex Graph Edition)")
st.markdown("""
This demo uses **Dual Node2Vec Embeddings** on both Passing and Defensive Synergy networks to find hyper-realistic replacements. 
It uses **Expected Threat (xT)**, **Clustering Coefficients**, and **K-Means Archetypes** to calculate deep tactical similarity without crashing!
""")

pass_df, def_df, stats_df = generate_mock_data()
graphs, def_graphs, features_df = process_graphs_and_features(pass_df, def_df, stats_df)

st.sidebar.header("Network Layer")
graph_layer = st.sidebar.radio("Select Network Layer", ["Passing Network (xT)", "Defensive Synergy"])
st.sidebar.markdown("---")

st.sidebar.header("Scouting Target")
selected_team = st.sidebar.selectbox("Select Team", pass_df['team_name'].unique())
team_players = features_df[features_df['team_name'] == selected_team]['player_name'].tolist()
selected_player = st.sidebar.selectbox("Select Player to Replace", team_players)

results = find_similar_players(selected_player, features_df)

col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader(f"1. {graph_layer} ({selected_team})")
    st.markdown("*Nodes are strictly mapped to pitch coordinates.*")
    
    display_graph = graphs[selected_team] if "Passing" in graph_layer else def_graphs[selected_team]
    network_html = plot_pyvis_graph(display_graph, selected_player)
    components.html(network_html, height=470)

with col2:
    st.subheader("2. Best Replacements")
    if results.empty:
        st.warning("No similar players found in this position.")
    else:
        pos_group = features_df[features_df['player_name'] == selected_player]['broad_position'].values[0]
        target_archetype = features_df[features_df['player_name'] == selected_player]['archetype'].values[0]
        
        st.markdown(f"**Target Archetype:** `{target_archetype}`")
        st.markdown(f"**Top Matches (Filtered by `{pos_group}`):**")
        for i, (_, row) in enumerate(results.iterrows()):
            st.markdown(f"{i+1}. **{row['player_name']}** (Similarity: `{row['similarity_score']:.3f}` | `{row['archetype']}`)")
            
        st.markdown("---")
        
        best_match = results.iloc[0]['player_name']
        match_team = results.iloc[0]['team_name']
        
        target_data = features_df[features_df['player_name'] == selected_player]
        match_data = features_df[features_df['player_name'] == best_match]
        
        tab1, tab2, tab3 = st.tabs(["Radar Profile", "Stat Comparison", "Scout & Ego Graphs"])
        
        with tab1:
            st.markdown("#### Tactical & Technical Profile")
            st.markdown("Combines network shape with physical/technical tabular attributes.")
            fig_radar = plot_radar_chart(target_data, match_data)
            st.plotly_chart(fig_radar, use_container_width=True)
            
        with tab2:
            st.markdown("#### Key Graph Metrics")
            fig_bar_xt = plot_stat_comparison_bar(
                target_data['total_xt_generated'].values[0], match_data['total_xt_generated'].values[0],
                selected_player, best_match, "Total xT Generated (Passing Danger)"
            )
            st.plotly_chart(fig_bar_xt, use_container_width=True)
            
            fig_bar_def = plot_stat_comparison_bar(
                target_data['def_pagerank'].values[0], match_data['def_pagerank'].values[0],
                selected_player, best_match, "Defensive PageRank (Synergy Influence)"
            )
            st.plotly_chart(fig_bar_def, use_container_width=True)
            
            fig_bar1 = plot_stat_comparison_bar(
                target_data['pagerank'].values[0], match_data['pagerank'].values[0],
                selected_player, best_match, "Passing PageRank (Influence)"
            )
            st.plotly_chart(fig_bar1, use_container_width=True)
            
            fig_bar2 = plot_stat_comparison_bar(
                target_data['clustering'].values[0], match_data['clustering'].values[0],
                selected_player, best_match, "Clustering Coefficient (Triangles)"
            )
            st.plotly_chart(fig_bar2, use_container_width=True)
            
        with tab3:
            st.info(generate_scout_summary(target_data, match_data))
            
            st.markdown(f"#### Direct {graph_layer} Circles")
            st.markdown("Isolates the player to show only their immediate connections.")
            
            ego_col1, ego_col2 = st.columns(2)
            with ego_col1:
                st.markdown(f"**Target:** `{selected_player}`")
                display_ego_target = graphs[selected_team] if "Passing" in graph_layer else def_graphs[selected_team]
                ego_html_1 = plot_ego_graph(display_ego_target, selected_player, height="260px")
                components.html(ego_html_1, height=275)
                
            with ego_col2:
                st.markdown(f"**Match:** `{best_match}`")
                display_ego_match = graphs[match_team] if "Passing" in graph_layer else def_graphs[match_team]
                ego_html_2 = plot_ego_graph(display_ego_match, best_match, height="260px")
                components.html(ego_html_2, height=275)
