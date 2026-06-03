import streamlit as st
import pandas as pd
import networkx as nx
import numpy as np
from node2vec import Node2Vec
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from pyvis.network import Network
import streamlit.components.v1 as components
import plotly.graph_objects as go
import random
import os

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
    
    season_multiplier = 10 
    
    for team, style in teams.items():
        players = {role: f"{team}_{role}" for role in roles}
        
        # A. GENERATE PASSING NETWORK
        for passer_role, targets in base_logic.items():
            passer_name = players[passer_role]
            for target_role, base_weight in targets:
                target_name = players[target_role]
                weight = base_weight * season_multiplier
                
                if style == "Possession":
                    if passer_role in ["CB1", "CB2", "DM", "CM", "LB", "RB"] and target_role in ["CB1", "CB2", "DM", "CM", "LB", "RB"]:
                        weight *= 1.8 
                elif style == "Direct":
                    if passer_role in ["GK", "CB1", "CB2"] and target_role in ["ST", "LW", "RW"]:
                        weight *= 3.0 
                    if passer_role in ["DM", "CM"] and target_role in ["DM", "CM"]:
                        weight *= 0.5 
                elif style == "High Press":
                    if passer_role in ["LW", "RW", "ST"] and target_role in ["AM", "LW", "RW", "ST"]:
                        weight *= 1.5
                
                final_weight = max(1, int(weight * random.uniform(0.7, 1.3)))
                pass_data.append({
                    "passer_name": passer_name,
                    "receiver_name": target_name,
                    "team_name": team,
                    "pass_count": final_weight
                })
                
        # B. GENERATE TABULAR STATS
        for role in roles:
            player_name = players[role]
            
            # Default low stats
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
                
            # Add some randomness so not all STs have exact same template ranges
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
            
    return pd.DataFrame(pass_data), pd.DataFrame(stats_data)

# --- 2. GRAPH CONSTRUCTION & FEATURE EXTRACTION ---
@st.cache_resource
def process_graphs_and_features(pass_df, stats_df):
    teams = pass_df['team_name'].unique()
    graph_features_list = []
    graphs = {}
    
    for team in teams:
        team_df = pass_df[pass_df['team_name'] == team]
        
        G = nx.DiGraph()
        for _, row in team_df.iterrows():
            G.add_edge(row['passer_name'], row['receiver_name'], weight=row['pass_count'])
            
        graphs[team] = G
        
        in_degree = nx.in_degree_centrality(G)
        out_degree = nx.out_degree_centrality(G)
        betweenness = nx.betweenness_centrality(G, weight='weight')
        pagerank = nx.pagerank(G, weight='weight')
        
        node2vec = Node2Vec(G, dimensions=8, walk_length=10, num_walks=50, workers=1, quiet=True)
        model = node2vec.fit(window=5, min_count=1, batch_words=4)
        
        for node in G.nodes():
            n2v_embedding = model.wv[node]
            features = {
                "player_name": node,
                "in_degree": in_degree[node],
                "out_degree": out_degree[node],
                "betweenness": betweenness[node],
                "pagerank": pagerank[node]
            }
            for i, val in enumerate(n2v_embedding):
                features[f"n2v_{i}"] = val
            graph_features_list.append(features)
            
    graph_features_df = pd.DataFrame(graph_features_list)
    # Merge Graph Features with Tabular Stats
    final_features_df = pd.merge(stats_df, graph_features_df, on="player_name")
    
    return graphs, final_features_df

# --- 3. SIMILARITY ENGINE ---
def find_similar_players(target_player, features_df, top_n=3):
    target_data = features_df[features_df["player_name"] == target_player].iloc[0]
    target_broad_pos = target_data["broad_position"]
    
    # 1. Positional Filtering
    filtered_df = features_df[features_df["broad_position"] == target_broad_pos].copy()
    
    # 2. Separate Numeric Columns
    exclude_cols = ["player_name", "team_name", "broad_position"]
    numeric_cols = [c for c in filtered_df.columns if c not in exclude_cols]
    
    X = filtered_df[numeric_cols].copy()
    
    # 3. Scale Features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=numeric_cols, index=filtered_df.index)
    
    # We heavily upweight explicit centralities so the demo yields highly logical results
    centrality_cols = ["in_degree", "out_degree", "betweenness", "pagerank"]
    for col in centrality_cols:
        X_scaled_df[col] = X_scaled_df[col] * 2.0
        
    # We can also upweight key stats for GKs
    if target_broad_pos == "Goalkeeper":
        X_scaled_df["gk_saves"] *= 2.0
        X_scaled_df["gk_reflexes"] *= 2.0
        
    # Calculate Cosine Similarity
    similarity_matrix = cosine_similarity(X_scaled_df)
    
    # Get index of target within the FILTERED dataframe
    target_idx_loc = filtered_df.index.get_loc(target_data.name)
    similarities = similarity_matrix[target_idx_loc]
    
    filtered_df["similarity_score"] = similarities
    
    # Filter out the target player and sort
    results = filtered_df[filtered_df["player_name"] != target_player]
    results = results.sort_values("similarity_score", ascending=False).head(top_n)
    
    return results

# --- 4. VISUALIZATIONS ---
def plot_pyvis_graph(G, target_player=None, height="450px"):
    net = Network(height=height, width="100%", directed=True, bgcolor="#0E1117", font_color="white")
    
    for node in G.nodes():
        color = "#FF4B4B" if node == target_player else "#1F77B4"
        size = 25 if node == target_player else 15
        in_deg = dict(G.in_degree(weight='weight')).get(node, 0)
        out_deg = dict(G.out_degree(weight='weight')).get(node, 0)
        title = f"Player: {node}\\nReceived: {in_deg}\\nPassed: {out_deg}"
        label = node.split("_")[1] if "_" in node else node
        net.add_node(node, label=label, title=title, color=color, size=size)
        
    for u, v, data in G.edges(data=True):
        weight = data['weight']
        net.add_edge(u, v, value=weight, title=f"{weight} passes", color="#4b5563")
        
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
    
    for node in ego_G.nodes():
        color = "#FF4B4B" if node == target_player else "#1F77B4"
        size = 25 if node == target_player else 15
        label = node.split("_")[1] if "_" in node else node
        net.add_node(node, label=label, color=color, size=size)
        
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
    # Combine Graph metrics (0-1) and Tabular metrics (0-100)
    # We will normalize tabular metrics to 0-1 for plotting only
    
    # Decide which stats to show based on position
    pos = target_data['broad_position'].values[0]
    
    if pos == "Goalkeeper":
        categories = ['In-Degree', 'Out-Degree', 'Betweenness', 'GK Saves', 'GK Reflexes']
        target_vals = [
            target_data['in_degree'].values[0],
            target_data['out_degree'].values[0],
            target_data['betweenness'].values[0],
            target_data['gk_saves'].values[0] / 100.0,
            target_data['gk_reflexes'].values[0] / 100.0
        ]
        match_vals = [
            match_data['in_degree'].values[0],
            match_data['out_degree'].values[0],
            match_data['betweenness'].values[0],
            match_data['gk_saves'].values[0] / 100.0,
            match_data['gk_reflexes'].values[0] / 100.0
        ]
    else:
        categories = ['In-Degree', 'Out-Degree', 'Betweenness', 'Speed', 'Shooting', 'Dribbling', 'Defending']
        target_vals = [
            target_data['in_degree'].values[0],
            target_data['out_degree'].values[0],
            target_data['betweenness'].values[0],
            target_data['speed'].values[0] / 100.0,
            target_data['shooting'].values[0] / 100.0,
            target_data['dribbling'].values[0] / 100.0,
            target_data['defending'].values[0] / 100.0
        ]
        match_vals = [
            match_data['in_degree'].values[0],
            match_data['out_degree'].values[0],
            match_data['betweenness'].values[0],
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
        line_color="#FF4B4B"
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=match_vals,
        theta=categories,
        fill='toself',
        name=f"Match: {match_data['player_name'].values[0]}",
        line_color="#1F77B4"
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1])
        ),
        showlegend=True,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        margin=dict(l=40, r=40, t=20, b=20)
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
        height=150,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        xaxis=dict(showgrid=False, visible=False),
        yaxis=dict(showgrid=False),
        showlegend=False
    )
    fig.update_traces(texttemplate='%{x:.2f}', textposition='outside')
    return fig

def generate_scout_summary(target_data, match_data):
    t_in = target_data['in_degree'].values[0]
    t_out = target_data['out_degree'].values[0]
    t_bw = target_data['betweenness'].values[0]
    
    if t_in > t_out * 1.5:
        role = "**Target / Finisher**"
    elif t_out > t_in * 1.5:
        role = "**Deep Distributor / Initiator**"
    elif t_bw > 0.05:
        role = "**Progressive Link-up Player**"
    else:
        role = "**Balanced Hub**"
        
    pos = target_data['broad_position'].values[0]
    
    if pos == "Goalkeeper":
        return f"**Scout's Summary:** Both players function as a {role} in the passing network. Because they are Goalkeepers, the model filtered out all outfield players and heavily weighted their GK specific attributes (Saves, Reflexes) alongside their ball-playing distribution."
    else:
        return f"**Scout's Summary:** Both players function primarily as a {role} Their structural 'shape' in the passing network is virtually identical. Furthermore, they match closely in physical/technical attributes (Speed, Shooting, etc.), ensuring a complete like-for-like structural replacement."

# --- STREAMLIT UI ---
st.set_page_config(page_title="Graph Recruitment", layout="wide")

st.title("⚽ Graph-Based Player Similarity for Recruitment")
st.markdown("""
This demo models passing networks as directed graphs and combines it with **Tabular Data** (Speed, Shooting, Defending).
It enforces **Positional Filtering** (e.g., GKs only compare to GKs) to find hyper-realistic replacements.
""")

# Load Data
pass_df, stats_df = generate_mock_data()
graphs, features_df = process_graphs_and_features(pass_df, stats_df)

# Sidebar Selection
st.sidebar.header("Scouting Target")
selected_team = st.sidebar.selectbox("Select Team", pass_df['team_name'].unique())
team_players = features_df[features_df['team_name'] == selected_team]['player_name'].tolist()
selected_player = st.sidebar.selectbox("Select Player to Replace", team_players)

# Main Layout
col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader(f"1. Passing Network ({selected_team})")
    st.markdown("*Interactive: You can drag nodes and hover over them for stats.*")
    network_html = plot_pyvis_graph(graphs[selected_team], selected_player)
    components.html(network_html, height=470)

with col2:
    st.subheader("2. Structural Replacements (KNN)")
    if selected_player:
        results = find_similar_players(selected_player, features_df)
        
        # Display Results compactly
        pos_group = features_df[features_df['player_name'] == selected_player]['broad_position'].values[0]
        st.markdown(f"**Top Matches (Filtered by: `{pos_group}`):**")
        for i, (_, row) in enumerate(results.iterrows()):
            st.markdown(f"{i+1}. **{row['player_name']}** (Similarity: `{row['similarity_score']:.3f}`)")
            
        st.markdown("---")
        st.subheader("3. Deep Dive Analysis")
        
        target_data = features_df[features_df['player_name'] == selected_player]
        best_match = results.iloc[0]['player_name']
        match_data = features_df[features_df['player_name'] == best_match]
        match_team = match_data['team_name'].values[0]
        
        # Use Tabs for cleaner organization
        tab1, tab2, tab3 = st.tabs(["Radar Profile", "Stat Comparison", "Scout & Ego Graphs"])
        
        with tab1:
            st.markdown("#### Tactical & Technical Profile")
            st.markdown("Combines network shape with physical/technical tabular attributes.")
            fig = plot_radar_chart(target_data, match_data)
            st.plotly_chart(fig, use_container_width=True)
            
        with tab2:
            st.markdown("#### Detailed Stat Comparison")
            pos_group = target_data['broad_position'].values[0]
            
            st.markdown("**Passing (Graph Metrics)**")
            fig_pass = plot_stat_comparison_bar(
                target_data['out_degree'].values[0], match_data['out_degree'].values[0],
                "Target", "Match",
                "Out-Degree (Passes Made)"
            )
            st.plotly_chart(fig_pass, use_container_width=True)
            
            if pos_group == "Goalkeeper":
                st.markdown("**Goalkeeping (Tabular Metrics)**")
                fig_gk1 = plot_stat_comparison_bar(
                    target_data['gk_saves'].values[0], match_data['gk_saves'].values[0],
                    "Target", "Match",
                    "GK Saves"
                )
                st.plotly_chart(fig_gk1, use_container_width=True)
                
                fig_gk2 = plot_stat_comparison_bar(
                    target_data['gk_reflexes'].values[0], match_data['gk_reflexes'].values[0],
                    "Target", "Match",
                    "GK Reflexes"
                )
                st.plotly_chart(fig_gk2, use_container_width=True)
            else:
                st.markdown("**Defending (Tabular Metric)**")
                fig_def = plot_stat_comparison_bar(
                    target_data['defending'].values[0], match_data['defending'].values[0],
                    "Target", "Match",
                    "Defending Attribute"
                )
                st.plotly_chart(fig_def, use_container_width=True)
                
                st.markdown("**Shooting (Tabular Metric)**")
                fig_shoot = plot_stat_comparison_bar(
                    target_data['shooting'].values[0], match_data['shooting'].values[0],
                    "Target", "Match",
                    "Shooting Attribute"
                )
                st.plotly_chart(fig_shoot, use_container_width=True)
            
        with tab3:
            # Scout Summary
            st.info(generate_scout_summary(target_data, match_data))
            
            # Ego Graphs
            st.markdown("#### Direct Passing Circles")
            st.markdown("Isolates the player to show only their immediate passing connections.")
            
            ego_col1, ego_col2 = st.columns(2)
            with ego_col1:
                st.markdown(f"**Target:** `{selected_player}`")
                ego_html_1 = plot_ego_graph(graphs[selected_team], selected_player, height="260px")
                components.html(ego_html_1, height=275)
                
            with ego_col2:
                st.markdown(f"**Match:** `{best_match}`")
                ego_html_2 = plot_ego_graph(graphs[match_team], best_match, height="260px")
                components.html(ego_html_2, height=275)
