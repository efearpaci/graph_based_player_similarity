# ScoutGraph — Graph-Based Player Similarity for Football Recruitment

## Data
This project runs on the **public Wyscout/Pappalardo Soccer Match Event Dataset**
(Pappalardo et al., 2019, *Scientific Data* — [figshare](https://figshare.com/collections/Soccer_match_event_dataset/4415000)):
every recorded event of the **2017/18 season of the Premier League, La Liga, Serie A,
Bundesliga and Ligue 1** (~2,600 players, ~1.9M passes). Expected Threat values use
Karun Singh's public 12×8 xT grid.

## Project Overview
Traditional football scouting relies on independent, tabular metrics (passes completed, tackles won). But football is a relational system: a player's value depends on *how* they connect with teammates. This project models each team's season as **directed, weighted graphs** (xT-weighted passing network + defensive co-action network) and combines **graph-topological features**, **Node2Vec style embeddings**, and **per-90 event stats** into a scout-configurable, cross-league similarity search.

## Features
- **Cross-league player search** — ~1,600 eligible players (≥900 minutes); results always restricted to the same position group, with a league filter.
- **Scout-selectable parameters** — choose which feature groups define "similar" (Attacking, Creation & Passing, Defending, Physique, Goalkeeping, Passing Network, Defensive Synergy, Style Embeddings) and weight each group.
- **Cosine vs Euclidean** — same *style* regardless of output level, or same style *and* output level.
- **Explainability** — per-group similarity breakdown showing *why* two players match.
- **Radar profiles** — percentile-based (within position, Big-5-wide) comparison.
- **Network view** — season passing / defensive networks with nodes at each player's real average on-ball position.
- **Percentile-based archetypes** — Target Man, Deep-Lying Playmaker, Sweeper Keeper…

## Pipeline
| Step | Command | What it does |
|---|---|---|
| 1 | `python download_data.py` | Downloads the raw Wyscout dumps (~90 MB zipped) into `data/raw/` |
| 2 | `python build_dataset.py --leagues all` | ETL: reconstructs pass receivers, builds season networks, computes per-90 stats & xT → `data/processed/*.parquet` |
| 3 | `python train_metric.py --sweep` *(optional)* | Triplet hyperparameter sweep (mining × margin × dim, multi-seed) selected on a **validation fold** (55/15/30 time split) |
| 4 | `python train_metric.py --final` *(optional)* | Trains the best config on train+val, exports learned-metric embeddings |
| 5 | `python learn_weights.py` *(optional)* | Learns per-position feature-group weights on the validation fold → app slider defaults |
| 6 | `streamlit run app.py` | The dashboard. First launch computes Node2Vec embeddings for all 98 teams (~5 min, cached to parquet) |

## Architecture
| File | Responsibility |
|---|---|
| `download_data.py` | Fetch raw data from figshare |
| `build_dataset.py` | ETL: events → players/pass-edges/def-edges tables |
| `real_data.py` | Data loader + position refinement (avg-position-based roles) |
| `engine.py` | Graph construction, centralities, Node2Vec, similarity search |
| `styles.py` | Purple/black UI theme (CSS + card components) |
| `app.py` | Streamlit dashboard |
| `mock_data.py` | (Legacy) procedural fake-data generator used before the real-data switch |

## Evaluation
`python build_dataset.py --leagues all --halves` then `python evaluation.py` runs three experiments
(results shown on the app's **Validation** page):
1. **Identity retrieval** (split-half reliability, Davis et al. 2024) — a player's first-half profile should retrieve the same player from the second half, compared across feature configurations.
2. **Substitution pairs** (Yılmaz & Öğüdücü 2022) — real same-position substitutions as weak interchangeability labels.
3. **Role coherence** — do a player's top-5 neighbours share their refined role label?

Headline findings: per-90 event stats are a strong identity fingerprint (top-5 retrieval 37% from pools of ~400 vs ~1% random). Naive Node2Vec fails across graphs because independently trained embedding spaces are unaligned (top-5 1.9%, role lift ×0.97). **GraphWave** (Donnat et al. 2018) — implemented in `engine.graphwave_embeddings` — fixes this by construction: deterministic heat-wavelet role signatures are comparable across teams (cross-team GK–GK cosine ≈ 0.999) and beat Node2Vec on every metric (top-5 4.0% vs 1.9%, role lift ×1.14 vs ×0.97). Structure encodes *role* rather than *identity*, so tabular stats remain the strongest individual fingerprint — the two are complementary, which is why the app combines them with scout-controlled weights.

**Deep metric learning** (`train_metric.py`): a triplet network trained on real substitution pairs with a leakage-safe **55/15/30 time split** (hyperparameters selected on validation only; test touched once). The tuned model (semi-hard negative mining, margin 0.2, dim 64; val 61.3%±0.8 over 5 seeds) **wins the held-out substitution task** — hit@1 26.6% / hit@3 60.3% vs 22.1% / 56.5% for the best engineered metric and 15.4% / 45.8% random — while scoring mid-pack on identity retrieval (10.4% top-5). The lesson: *metrics encode objectives* — the learned metric optimises interchangeability, engineered cosine over per-90 stats optimises identity, and the right choice depends on the scouting question.

**Learned group weights** (`learn_weights.py`): per-position feature-group weights optimised on the validation fold transfer to the test fold — Midfielder hit@3 42.2%→48.9%, Defender 50.6%→56.5%, Forward 69.5%→70.6% vs all-ones weights. Two interpretable findings: *Creation & Passing* is the strongest interchangeability signal for every outfield position (weight 3.0), and GraphWave gets weight 0 for this task — structural role barely discriminates within a same-squad, same-position candidate set (everyone there shares the role), even though it separates roles across the league. A **GraphWave size sweep** (`tune_graphwave.py`, 16/32/64 dims) showed the embedding is insensitive to size on ~30-node team graphs; the 32-dim default stays.

## Method notes
- **Receiver reconstruction:** Wyscout doesn't record pass receivers; for accurate passes the receiver is taken as the next event of the same team (standard in the passing-network literature).
- **Eligibility:** similarity search pools only players with ≥900 minutes (≈10 full matches), following Davis et al. (2024).
- **Limitations:** no physical/tracking data (speed, distance) — the Physique group covers height/weight only; market values are not included in the dataset.
