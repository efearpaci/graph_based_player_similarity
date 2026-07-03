"""
Step 2: ETL — convert raw Wyscout event streams into the three tables the app
consumes. Run after download_data.py:

    python build_dataset.py                  # England only (fast validation)
    python build_dataset.py --leagues all    # all five leagues

Outputs (data/processed/):
    players.parquet     one row per player-season (identity + per-90 stats)
    pass_edges.parquet  season passing network per team (count + xT weights)
    def_edges.parquet   defensive co-action network per team

Conventions of the Wyscout data (Pappalardo et al., 2019):
  - positions are 0-100, x toward the opponent goal
  - tag 1801/1802 = accurate/inaccurate, 703/701 = duel won/lost,
    101 goal, 102 own goal, 301 assist, 302 key pass,
    1401 interception, 1501 clearance
  - a pass's receiver is not recorded: for accurate passes we take the next
    event of the same team in the same match (standard reconstruction used in
    the passing-network literature)
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).parent / "data" / "raw"
OUT = Path(__file__).parent / "data" / "processed"

LEAGUES = {
    "England": "Premier League", "Spain": "La Liga", "Italy": "Serie A",
    "Germany": "Bundesliga", "France": "Ligue 1",
}

MIN_MINUTES = 900          # eligibility threshold for the search pool (Davis et al.)
DEF_WINDOW_SEC = 20        # window linking teammates' defensive actions
PROGRESSIVE_DX = 20        # forward progress (0-100 scale) to count as progressive

ACCURATE, INACCURATE = 1801, 1802
GOAL, OWN_GOAL, ASSIST, KEY_PASS = 101, 102, 301, 302
INTERCEPTION, CLEARANCE, WON = 1401, 1501, 703

XT = np.array(json.load(open(RAW / "xT_grid.json")))  # 8 rows (y) x 12 cols (x)


def xt_value(x, y):
    col = min(11, max(0, int(x / 100 * 12)))
    row = min(7, max(0, int(y / 100 * 8)))
    return XT[row][col]


def fix_text(s):
    """players.json double-escapes unicode: strings contain literal '\\u00fc'.
    The file is pure ASCII, so decoding the escapes directly is safe."""
    if "\\u" not in s:
        return s
    try:
        return s.encode("ascii").decode("unicode_escape")
    except (UnicodeDecodeError, UnicodeEncodeError):
        return s


def flag_emoji(alpha2):
    if not alpha2 or len(alpha2) != 2:
        return "🏳️"
    return "".join(chr(0x1F1E6 + ord(c) - ord("A")) for c in alpha2.upper())


def load_minutes(matches):
    """Minutes played per (playerId, teamId) from lineups & substitutions."""
    minutes = Counter()
    for m in matches:
        for team_id, td in m["teamsData"].items():
            f = td.get("formation") or {}
            subs = f.get("substitutions") or []
            subs = subs if isinstance(subs, list) else []
            sub_in = {s["playerIn"]: s["minute"] for s in subs}
            sub_out = {s["playerOut"]: s["minute"] for s in subs}
            for p in f.get("lineup") or []:
                minutes[(p["playerId"], int(team_id))] += min(90, sub_out.get(p["playerId"], 90))
            for p in f.get("bench") or []:
                if p["playerId"] in sub_in:
                    start = sub_in[p["playerId"]]
                    end = sub_out.get(p["playerId"], 90)
                    minutes[(p["playerId"], int(team_id))] += max(0, min(90, end) - start)
    return minutes


def process_league(country, players_by_id, team_names, match_ids=None):
    print(f"— {country}: loading events …")
    events = json.load(open(RAW / "events" / f"events_{country}.json"))
    matches = json.load(open(RAW / "matches" / f"matches_{country}.json"))
    if match_ids is not None:
        matches = [m for m in matches if m["wyId"] in match_ids]
        events = [e for e in events if e["matchId"] in match_ids]
    minutes = load_minutes(matches)

    # group events per match (already time-ordered within match in the dump,
    # but sort defensively by period + clock)
    period_order = {"1H": 0, "2H": 1, "E1": 2, "E2": 3, "P": 4}
    by_match = defaultdict(list)
    for e in events:
        by_match[e["matchId"]].append(e)
    for match_events in by_match.values():
        match_events.sort(key=lambda e: (period_order.get(e["matchPeriod"], 9), e["eventSec"]))

    stats = defaultdict(Counter)          # playerId -> stat counters
    pos_sum = defaultdict(lambda: [0.0, 0.0, 0])  # playerId -> [sum_x, sum_y, n]
    pass_edges = Counter()                # (teamId, passer, receiver) -> passes
    xt_edges = defaultdict(float)         # (teamId, passer, receiver) -> xT
    def_edges = Counter()                 # (teamId, p1, p2) -> co-actions
    player_team = defaultdict(Counter)    # playerId -> {teamId: n_events}

    for match_events in by_match.values():
        last_def = {}  # teamId -> (playerId, period, sec) last defensive action
        n = len(match_events)
        for i, e in enumerate(match_events):
            pid, tid = e["playerId"], e["teamId"]
            if pid == 0:
                continue
            tags = {t["id"] for t in e["tags"]}
            name, sub = e["eventName"], e["subEventName"]
            pos = e["positions"]
            s = stats[pid]
            player_team[pid][tid] += 1
            if pos:
                pos_sum[pid][0] += pos[0]["x"]
                pos_sum[pid][1] += pos[0]["y"]
                pos_sum[pid][2] += 1

            # ---- goals / assists / key passes (any event type) ----
            if GOAL in tags and name in ("Shot", "Free Kick"):
                s["goals"] += 1
            if ASSIST in tags:
                s["assists"] += 1
            if KEY_PASS in tags:
                s["key_passes"] += 1
            if INTERCEPTION in tags:
                s["interceptions"] += 1
            if CLEARANCE in tags:
                s["clearances"] += 1

            # ---- passes ----
            if name == "Pass":
                s["passes"] += 1
                if ACCURATE in tags:
                    s["passes_accurate"] += 1
                if sub == "Cross":
                    s["crosses"] += 1
                if sub == "Smart pass":
                    s["smart_passes"] += 1
                if len(pos) > 1 and pos[1]["x"] - pos[0]["x"] >= PROGRESSIVE_DX:
                    s["progressive_passes"] += 1

                # receiver reconstruction + edge weights
                if ACCURATE in tags and i + 1 < n:
                    nxt = match_events[i + 1]
                    if nxt["teamId"] == tid and nxt["playerId"] not in (0, pid):
                        key = (tid, pid, nxt["playerId"])
                        pass_edges[key] += 1
                        if len(pos) > 1:
                            gain = xt_value(pos[1]["x"], pos[1]["y"]) - \
                                   xt_value(pos[0]["x"], pos[0]["y"])
                            xt_edges[key] += max(0.0, gain)

            # ---- shots ----
            elif name == "Shot":
                s["shots"] += 1
                if ACCURATE in tags:
                    s["shots_on_target"] += 1

            # ---- duels ----
            elif name == "Duel":
                if sub == "Ground attacking duel":
                    s["dribble_attempts"] += 1
                    if WON in tags:
                        s["dribbles_won"] += 1
                elif sub == "Ground defending duel":
                    s["def_duel_attempts"] += 1
                    if WON in tags:
                        s["tackles_won"] += 1
                elif sub == "Air duel":
                    s["aerial_attempts"] += 1
                    if WON in tags:
                        s["aerials_won"] += 1

            # ---- goalkeeping ----
            elif name == "Save attempt":
                s["save_attempts"] += 1
                if ACCURATE in tags:
                    s["saves"] += 1
            elif name == "Goalkeeper leaving line":
                s["gk_exits"] += 1

            # ---- defensive co-action network ----
            is_def_action = (
                (name == "Duel" and sub in ("Ground defending duel", "Air duel"))
                or INTERCEPTION in tags or CLEARANCE in tags
            )
            if is_def_action:
                prev = last_def.get(tid)
                here = (pid, e["matchPeriod"], e["eventSec"])
                if prev and prev[0] != pid and prev[1] == here[1] \
                        and here[2] - prev[2] <= DEF_WINDOW_SEC:
                    def_edges[(tid, prev[0], pid)] += 1
                last_def[tid] = here

    return {
        "stats": stats, "pos_sum": pos_sum, "minutes": minutes,
        "pass_edges": pass_edges, "xt_edges": xt_edges, "def_edges": def_edges,
        "player_team": player_team, "matches_per_team": Counter(
            int(tid) for m in matches for tid in m["teamsData"]),
    }


def season_halves(country):
    """Split a league's match ids into first/second half of season by date."""
    matches = json.load(open(RAW / "matches" / f"matches_{country}.json"))
    ordered = sorted(matches, key=lambda m: m["dateutc"])
    mid = len(ordered) // 2
    return ({m["wyId"] for m in ordered[:mid]},
            {m["wyId"] for m in ordered[mid:]})


def main(countries, suffix="", min_minutes=MIN_MINUTES, match_filters=None):
    OUT.mkdir(parents=True, exist_ok=True)
    players_raw = json.load(open(RAW / "players.json"))
    for p in players_raw:
        for k in ("shortName", "firstName", "lastName"):
            p[k] = fix_text(p.get(k) or "")
    players_by_id = {p["wyId"]: p for p in players_raw}
    teams_raw = json.load(open(RAW / "teams.json"))
    team_names = {t["wyId"]: fix_text(t["name"]) for t in teams_raw}

    all_players, all_pass, all_def = [], [], []

    for country in countries:
        r = process_league(country, players_by_id, team_names,
                           match_ids=(match_filters or {}).get(country))
        league = LEAGUES[country]

        # -- resolve display names (unique) --
        pids = set(r["player_team"])
        short = {}
        seen = Counter(players_by_id[p]["shortName"] for p in pids if p in players_by_id)
        for p in pids:
            meta = players_by_id.get(p)
            if not meta:
                continue
            name = meta["shortName"]
            if seen[name] > 1:
                name = f"{meta['firstName'][:1]}. {meta['lastName']}".strip()
                if name == meta["shortName"] or not meta["lastName"]:
                    name = f"{meta['shortName']} ({p})"
            short[p] = name

        # -- team assignment: team with most minutes (fallback: most events) --
        main_team = {}
        for p in pids:
            team_mins = {t: r["minutes"].get((p, t), 0) for t in r["player_team"][p]}
            main_team[p] = max(team_mins, key=lambda t: (team_mins[t], r["player_team"][p][t]))

        # -- team style from passing volume terciles --
        team_passes = Counter()
        for (tid, a, b), c in r["pass_edges"].items():
            team_passes[tid] += c
        vol = {t: team_passes[t] / max(1, r["matches_per_team"][t]) for t in team_passes}
        q1, q2 = np.percentile(list(vol.values()), [33, 66])
        style = {t: ("Direct" if v <= q1 else "Possession" if v >= q2 else "Balanced")
                 for t, v in vol.items()}

        pos_map = {"GK": "Goalkeeper", "DF": "Defender", "MF": "Midfielder", "FW": "Forward"}

        for p in pids:
            meta = players_by_id.get(p)
            if not meta or p not in short:
                continue
            tid = main_team[p]
            mins = sum(r["minutes"].get((p, t), 0) for t in r["player_team"][p])
            s = r["stats"][p]
            p90 = lambda k: round(s[k] / mins * 90, 3) if mins else 0.0
            sx, sy, n = r["pos_sum"][p]
            nat = meta.get("passportArea") or meta.get("birthArea") or {}
            birth = meta.get("birthDate") or "1990-01-01"
            age = 2018 - int(birth[:4])

            all_players.append({
                "player_id": p,
                "player_name": short[p],
                "full_name": f"{meta['firstName']} {meta['lastName']}".strip(),
                "team_name": team_names.get(tid, str(tid)),
                "league": league,
                "team_style": style.get(tid, "Balanced"),
                "position": pos_map.get(meta["role"]["code2"], "Midfielder"),
                "role_label": meta["role"]["name"],
                "age": age,
                "nationality": nat.get("name", "?"),
                "flag": flag_emoji(nat.get("alpha2code", "")),
                "foot": meta.get("foot", "?"),
                "height": meta.get("height", 0),
                "weight": meta.get("weight", 0),
                "minutes": int(mins),
                "eligible": mins >= min_minutes,
                "avg_x": round(sx / n, 1) if n else 50.0,
                "avg_y": round(sy / n, 1) if n else 50.0,
                # ---- per-90 stats ----
                "goals": p90("goals"), "shots": p90("shots"),
                "shots_on_target": p90("shots_on_target"),
                "dribbles_won": p90("dribbles_won"),
                "assists": p90("assists"), "key_passes": p90("key_passes"),
                "smart_passes": p90("smart_passes"), "crosses": p90("crosses"),
                "progressive_passes": p90("progressive_passes"),
                "passes": p90("passes"),
                "pass_accuracy": round(100 * s["passes_accurate"] / s["passes"], 1) if s["passes"] else 0.0,
                "tackles_won": p90("tackles_won"), "interceptions": p90("interceptions"),
                "clearances": p90("clearances"), "aerials_won": p90("aerials_won"),
                "aerial_win_rate": round(100 * s["aerials_won"] / s["aerial_attempts"], 1) if s["aerial_attempts"] else 0.0,
                "saves": p90("saves"),
                "save_rate": round(100 * s["saves"] / s["save_attempts"], 1) if s["save_attempts"] else 0.0,
                "gk_exits": p90("gk_exits"),
            })

        for (tid, a, b), c in r["pass_edges"].items():
            all_pass.append({
                "team_name": team_names.get(tid, str(tid)), "league": league,
                "passer_id": a, "receiver_id": b,
                "pass_count": c, "total_xt": round(r["xt_edges"][(tid, a, b)], 4),
            })
        for (tid, a, b), c in r["def_edges"].items():
            all_def.append({
                "team_name": team_names.get(tid, str(tid)), "league": league,
                "passer_id": a, "receiver_id": b,
                "synergy_score": c,
            })

    players_df = pd.DataFrame(all_players)
    # names must be globally unique across leagues for the app's name-keyed joins
    dup = players_df["player_name"].duplicated(keep=False)
    players_df.loc[dup, "player_name"] = (
        players_df.loc[dup, "player_name"] + " (" + players_df.loc[dup, "team_name"] + ")")
    id_to_name = dict(zip(players_df["player_id"], players_df["player_name"]))

    def finalize_edges(rows):
        df = pd.DataFrame(rows)
        df["passer_name"] = df["passer_id"].map(id_to_name)
        df["receiver_name"] = df["receiver_id"].map(id_to_name)
        return df.dropna(subset=["passer_name", "receiver_name"]) \
                 .drop(columns=["passer_id", "receiver_id"])

    pass_df = finalize_edges(all_pass)
    def_df = finalize_edges(all_def)

    players_df.to_parquet(OUT / f"players{suffix}.parquet", index=False)
    pass_df.to_parquet(OUT / f"pass_edges{suffix}.parquet", index=False)
    def_df.to_parquet(OUT / f"def_edges{suffix}.parquet", index=False)

    print(f"\n✓ wrote {len(players_df)} players "
          f"({players_df.eligible.sum()} eligible ≥{min_minutes} min), "
          f"{len(pass_df)} pass edges, {len(def_df)} defensive edges [suffix='{suffix}']")
    print(f"  → {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--leagues", default="England",
                    help="'England', comma-list, or 'all'")
    ap.add_argument("--halves", action="store_true",
                    help="additionally build first/second half-season datasets "
                         "(_h1/_h2 suffixes) for the identity-retrieval evaluation")
    args = ap.parse_args()
    countries = list(LEAGUES) if args.leagues == "all" else args.leagues.split(",")

    if args.halves:
        h1 = {c: season_halves(c)[0] for c in countries}
        h2 = {c: season_halves(c)[1] for c in countries}
        print("=== first half of season ===")
        main(countries, suffix="_h1", min_minutes=450, match_filters=h1)
        print("\n=== second half of season ===")
        main(countries, suffix="_h2", min_minutes=450, match_filters=h2)
    else:
        main(countries)
