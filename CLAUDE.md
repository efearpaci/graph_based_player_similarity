# ScoutGraph — Graph-Based Player Similarity for Football Recruitment (MSc thesis)

Act as a senior data scientist / football-analytics thesis advisor. Core thesis: football
is relational — modeling team seasons as directed, weighted graphs finds players with
similar *tactical roles* across teams and leagues, which tabular stats alone miss.

## Commands
- `streamlit run app.py` — dashboard (main page + 📊 Validation page)
- `python download_data.py` — fetch raw Wyscout dumps (~1 GB, gitignored)
- `python build_dataset.py --leagues all [--halves]` — ETL → `data/processed/*.parquet`
- `python train_metric.py [--sweep|--variance|--final]` — triplet net; sweep selects on
  the VALIDATION fold (55/15/30 time split), final trains train+val & exports
- `python learn_weights.py` — per-position group weights (val-optimised → app defaults)
- `python tune_graphwave.py` — GraphWave size sweep (result: size-insensitive, keep 32)
- `python evaluation.py` — 3 experiments → `evaluation_results.json` (needs `--halves` build)

## Architecture
| File | Responsibility |
|---|---|
| `build_dataset.py` | ETL: receiver reconstruction, xT (Karun Singh grid), per-90 stats, network edges |
| `real_data.py` | Loads processed parquets; refines roles from avg pitch position |
| `engine.py` | Graphs, centralities, Node2Vec + GraphWave embeddings, weighted-group KNN |
| `evaluation.py` | Identity retrieval (split-half), substitution pairs, role coherence |
| `app.py` / `styles.py` / `pages/` | Streamlit UI (purple/black theme) |
| `mock_data.py` | Legacy fake-data generator (pre-real-data; unused by app) |

## Data & method facts
- Dataset: Wyscout/Pappalardo 2017/18 Big-5 leagues (figshare, CC BY 4.0). Swappable.
- Pass receivers are NOT in the raw data — reconstructed as next same-team event.
- Search pool: eligible = ≥900 min (≥450 per half in split-half builds).
- Similarity: z-score within position pool, group weights ÷ √(group size), cosine/euclidean.

## Established results (evaluation.py, don't re-derive)
- GraphWave beats Node2Vec on every metric (identity top-5 4.0% vs 1.9%; role lift ×1.14
  vs ×0.97). Reason: GraphWave signatures are deterministic → comparable across team
  graphs; independently trained Node2Vec spaces are unaligned. Cross-team GK–GK cosine
  ≈ 0.999, Salah–Ronaldo 0.92.
- Tabular per-90 stats are the strongest *identity* fingerprint (top-5 37.4%); structure
  encodes *role* — complementary, hence weighted hybrid.
- Naive concatenation dilutes (32 GraphWave dims swamp 29 tabular): weight groups.
- Triplet metric (train_metric.py, torch MLP): tuned config = semi-hard mining,
  margin 0.2, dim 64 (sweep on val fold; val 61.3%±0.8 hit@3 over 5 seeds).
  WINS held-out substitution retrieval (test hit@1 26.6% / hit@3 60.3% vs tabular
  22.1%/56.5%, random 15.4%/45.8%) but mid-pack on identity (10.4% top-5):
  metrics encode objectives. Split is 55/15/30 by date (substitution_split.json);
  tune on val ONLY, test is touched once — never retrain/tune on it.
- Learned per-position group weights (learn_weights.py → learned_weights.json,
  app slider defaults): val-optimised, transfers to test (Mid hit@3 42→49,
  Def 51→56, Fwd 69.5→70.6). Creation & Passing weight 3.0 everywhere;
  GraphWave weight 0 within-squad (role doesn't discriminate among squadmates).
  GK kept all-ones (only 3 val trials).
- GraphWave size sweep: 16/32/64 dims ≈ identical → size-insensitive on 30-node
  graphs; GW_POINTS=4 (32 dims) stays.

## Gotchas
- `.streamlit/config.toml` disables the file watcher → **restart** streamlit to pick up
  code changes; reload is not enough.
- `players.json` double-escapes unicode (literal `ü`) — `fix_text()` handles it.
- Mid-season cross-league movers appear once per league under the same `player_id` —
  dedupe before any matrix keyed by player_id (corrupted the eval once).
- Feature caches: `data/processed/graph_features*.parquet` — delete to force recompute
  (~5 min per full run for 98 teams).
- Edge tables are keyed by `player_name` (globally uniquified); join via `player_id`
  when in doubt.
