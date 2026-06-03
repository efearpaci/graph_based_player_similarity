# Graph-Based Player Similarity for Football Recruitment (Demo)

## ⚠️ Disclaimer
**This repository is a Proof-of-Concept Demo.** It uses entirely **fake, procedurally generated statistics** to demonstrate the methodology of the thesis. No real-world player data is used in this repository. The passing volumes, tactical styles, and physical attributes are generated via Python to simulate a realistic environment.

## Project Overview
Traditional football scouting heavily relies on independent, tabular metrics (e.g., passes completed, tackles won). However, football is a complex, continuous multi-agent system. This project models a football match as a **directed, weighted graph** to capture the spatial and relational context of player interactions.

By representing players as nodes and their passes as edges, we can apply Network Science and Graph Neural Networks (Node2Vec) to understand a player's **structural and tactical role** within a team.

### Features
This Streamlit dashboard demonstrates a **Hybrid Similarity Engine**:
1. **Graph Topological Features:** Calculates In-Degree, Out-Degree, Betweenness, and PageRank to determine a player's structural shape (e.g., "Deep Lying Playmaker" vs "Target Man").
2. **Latent Graph Embeddings:** Uses Node2Vec to compress complex passing sequences into vector representations.
3. **Tabular Metrics:** Combines the graph data with traditional physical/technical statistics (Speed, Shooting, Defending) for a complete holistic profile.
4. **Strict Positional Filtering:** Ensures that the K-Nearest Neighbors (KNN) similarity search only compares players within the same broad positional grouping (e.g., Goalkeepers are only compared to Goalkeepers).

### Visualizations
* **Interactive Passing Networks:** Built with `pyvis` to explore a team's passing structure dynamically.
* **Ego Graphs:** Isolates individual players to compare their immediate passing circles side-by-side.
* **Football Manager (FM) Style Radar Charts:** Plots both normalized tactical shape and technical attributes on a single polygon.

## How to Run Locally

### Requirements
```bash
pip install -r requirements.txt
```

### Run the App
```bash
streamlit run app.py
```
