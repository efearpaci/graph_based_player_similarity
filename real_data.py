"""
Real data layer: loads the processed Wyscout/Pappalardo 2017/18 season tables
produced by build_dataset.py (run download_data.py + build_dataset.py first).

Returns the same three DataFrames the app has always consumed:
    players_df, pass_df, def_df
"""

from pathlib import Path

import pandas as pd

PROCESSED = Path(__file__).parent / "data" / "processed"


def _refine_role_label(row):
    """Wyscout only gives GK/DF/MF/FW — refine using the player's average
    on-ball position (x: 0=own goal, 100=opponent goal; y: 0-100 across)."""
    wide = abs(row["avg_y"] - 50)
    if row["position"] == "Goalkeeper":
        return "Goalkeeper"
    if row["position"] == "Defender":
        return "Full-Back" if wide > 22 else "Centre-Back"
    if row["position"] == "Midfielder":
        if row["avg_x"] < 46:
            return "Defensive Mid"
        if row["avg_x"] > 58:
            return "Attacking Mid"
        return "Central Mid"
    return "Winger" if wide > 18 else "Striker"


def data_available():
    return (PROCESSED / "players.parquet").exists()


def load_real_data():
    players = pd.read_parquet(PROCESSED / "players.parquet")
    pass_df = pd.read_parquet(PROCESSED / "pass_edges.parquet")
    def_df = pd.read_parquet(PROCESSED / "def_edges.parquet")

    players["role_label"] = players.apply(_refine_role_label, axis=1)
    return players, pass_df, def_df
