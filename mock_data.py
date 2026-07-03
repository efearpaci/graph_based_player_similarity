"""
Mock data layer for the Graph-Based Player Similarity demo.

Everything here is procedurally generated fake data. This module is the ONLY
place that produces raw data — when we later switch to a real dataset
(e.g. StatsBomb Open Data / Wyscout), we only need to replace this module
while keeping the same three output DataFrames:

    players_df : one row per player (identity + tabular stats)
    pass_df    : directed passing edges per team (pass_count, total_xt)
    def_df     : directed defensive-synergy edges per team (synergy_score)
"""

import random
import pandas as pd

SEED = 42

TEAMS = {
    "Man City": "Possession", "Arsenal": "Possession", "Liverpool": "High Press",
    "Aston Villa": "Balanced", "Tottenham": "High Press", "Chelsea": "Possession",
    "Newcastle": "Balanced", "Man United": "Direct", "West Ham": "Direct",
    "Crystal Palace": "Direct", "Brighton": "Possession", "Bournemouth": "Balanced",
    "Fulham": "Balanced", "Wolves": "Direct", "Everton": "Direct",
    "Brentford": "Direct", "Nottm Forest": "Direct", "Luton": "Direct",
    "Burnley": "Possession", "Sheff Utd": "Direct",
}

ROLES = ["GK", "CB1", "CB2", "LB", "RB", "DM", "CM", "AM", "LW", "RW", "ST"]

ROLE_TO_POSITION = {
    "GK": "Goalkeeper",
    "CB1": "Defender", "CB2": "Defender", "LB": "Defender", "RB": "Defender",
    "DM": "Midfielder", "CM": "Midfielder", "AM": "Midfielder",
    "LW": "Forward", "RW": "Forward", "ST": "Forward",
}

ROLE_LABELS = {
    "GK": "Goalkeeper", "CB1": "Centre-Back (L)", "CB2": "Centre-Back (R)",
    "LB": "Left-Back", "RB": "Right-Back", "DM": "Defensive Mid",
    "CM": "Central Mid", "AM": "Attacking Mid", "LW": "Left Wing",
    "RW": "Right Wing", "ST": "Striker",
}

FIRST_NAMES = [
    "Marco", "Luka", "Jamal", "Enzo", "Kai", "Rodri", "Bruno", "Declan", "Phil",
    "Martin", "Bukayo", "Gabriel", "Victor", "Rasmus", "Darwin", "Cody", "Moises",
    "Alexis", "Eberechi", "Morgan", "Ollie", "Dominic", "Jarrod", "Anthony",
    "Casemiro", "Andre", "Joao", "Pedro", "Diogo", "Ruben", "Bernardo", "Nicolas",
    "Mateo", "Federico", "Lautaro", "Julian", "Emiliano", "Cristian", "Alejandro",
    "Ivan", "Milos", "Dusan", "Sergej", "Piotr", "Jakub", "Matty", "Callum",
    "Reece", "Trent", "Kobbie", "Adam", "Ethan", "Lewis", "Harvey", "Tyler",
    "Amadou", "Ibrahima", "Moussa", "Cheick", "Yves", "Sofyan", "Achraf",
]

LAST_NAMES = [
    "Silva", "Fernandes", "Rodriguez", "Martinez", "Kovacic", "Petrovic", "Diallo",
    "Traore", "Keita", "Mensah", "Okafor", "Yamamoto", "Larsson", "Johansson",
    "Andersen", "Nielsen", "Vermeer", "De Vries", "Jansen", "Dubois", "Moreau",
    "Lefevre", "Rossi", "Ferrari", "Romano", "Esposito", "Costa", "Oliveira",
    "Santos", "Pereira", "Alvarez", "Gonzalez", "Herrera", "Castillo", "Vargas",
    "Muller", "Schmidt", "Wagner", "Becker", "Hoffmann", "Kowalski", "Nowak",
    "Zielinski", "Horvat", "Novak", "Popescu", "Ionescu", "Dimitrov", "Petrov",
    "Yilmaz", "Kaya", "Demir", "Aksoy", "Osei", "Boateng", "Asante", "Nkemdirim",
    "Walker-Smith", "O'Connor", "Gallagher", "Docherty", "Whitfield", "Barnes",
    "Sterling-Cole", "Ashworth", "Redmond", "Calloway", "Winters", "Vance",
]

NATIONALITIES = [
    ("England", "🏴󠁧󠁢󠁥󠁮󠁧󠁿"), ("Brazil", "🇧🇷"), ("France", "🇫🇷"), ("Spain", "🇪🇸"),
    ("Portugal", "🇵🇹"), ("Germany", "🇩🇪"), ("Argentina", "🇦🇷"), ("Netherlands", "🇳🇱"),
    ("Italy", "🇮🇹"), ("Belgium", "🇧🇪"), ("Croatia", "🇭🇷"), ("Serbia", "🇷🇸"),
    ("Denmark", "🇩🇰"), ("Sweden", "🇸🇪"), ("Poland", "🇵🇱"), ("Turkey", "🇹🇷"),
    ("Ghana", "🇬🇭"), ("Nigeria", "🇳🇬"), ("Senegal", "🇸🇳"), ("Morocco", "🇲🇦"),
    ("Japan", "🇯🇵"), ("Uruguay", "🇺🇾"), ("Colombia", "🇨🇴"), ("Ecuador", "🇪🇨"),
]

# Passing volume logic between roles (who passes to whom, base weight)
PASS_LOGIC = {
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
    "ST": [("AM", 5), ("LW", 5), ("RW", 5)],
}

# Defensive synergy logic (who defends alongside whom)
DEF_LOGIC = {
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
    "ST": [("AM", 15), ("LW", 5), ("RW", 5)],
}


def _rand(rng, lo, hi):
    return rng.randint(lo, hi)


def _generate_stats(rng, role, style):
    """Position-specific tabular stats on a 1-99 scale (per-90 style ratings)."""
    # Baseline for everyone
    s = {
        # Attacking
        "goals": _rand(rng, 5, 25), "xg": _rand(rng, 5, 25),
        "shots_on_target": _rand(rng, 10, 30), "dribbles": _rand(rng, 15, 40),
        "big_chances": _rand(rng, 5, 25),
        # Creation & Passing
        "assists": _rand(rng, 10, 30), "xa": _rand(rng, 10, 30),
        "key_passes": _rand(rng, 15, 35), "progressive_passes": _rand(rng, 20, 45),
        "pass_accuracy": _rand(rng, 55, 75),
        # Defending
        "tackles": _rand(rng, 15, 35), "interceptions": _rand(rng, 15, 35),
        "blocks": _rand(rng, 10, 30), "aerial_duels": _rand(rng, 20, 45),
        "clearances": _rand(rng, 10, 30),
        # Physical
        "speed": _rand(rng, 40, 65), "stamina": _rand(rng, 50, 75),
        "strength": _rand(rng, 40, 65), "agility": _rand(rng, 45, 70),
        # Goalkeeping (near-zero for outfielders)
        "saves": _rand(rng, 1, 8), "reflexes": _rand(rng, 1, 8),
        "gk_distribution": _rand(rng, 1, 8), "claims": _rand(rng, 1, 8),
        "one_v_one": _rand(rng, 1, 8),
    }

    if role == "GK":
        s.update({
            "saves": _rand(rng, 70, 95), "reflexes": _rand(rng, 70, 95),
            "gk_distribution": _rand(rng, 55, 90), "claims": _rand(rng, 60, 92),
            "one_v_one": _rand(rng, 60, 92),
            "clearances": _rand(rng, 40, 70), "pass_accuracy": _rand(rng, 50, 85),
            "speed": _rand(rng, 30, 55), "strength": _rand(rng, 60, 85),
        })
    elif role in ("CB1", "CB2"):
        s.update({
            "tackles": _rand(rng, 65, 90), "interceptions": _rand(rng, 68, 92),
            "blocks": _rand(rng, 65, 90), "aerial_duels": _rand(rng, 70, 95),
            "clearances": _rand(rng, 70, 95), "strength": _rand(rng, 70, 92),
            "pass_accuracy": _rand(rng, 68, 90), "progressive_passes": _rand(rng, 35, 70),
        })
    elif role in ("LB", "RB"):
        s.update({
            "tackles": _rand(rng, 60, 85), "interceptions": _rand(rng, 58, 82),
            "speed": _rand(rng, 72, 94), "stamina": _rand(rng, 75, 95),
            "dribbles": _rand(rng, 45, 70), "key_passes": _rand(rng, 40, 65),
            "progressive_passes": _rand(rng, 45, 75), "xa": _rand(rng, 30, 55),
            "assists": _rand(rng, 25, 55),
        })
    elif role == "DM":
        s.update({
            "tackles": _rand(rng, 62, 88), "interceptions": _rand(rng, 65, 90),
            "pass_accuracy": _rand(rng, 78, 95), "progressive_passes": _rand(rng, 60, 88),
            "stamina": _rand(rng, 70, 92), "strength": _rand(rng, 60, 85),
            "key_passes": _rand(rng, 35, 60),
        })
    elif role == "CM":
        s.update({
            "pass_accuracy": _rand(rng, 75, 93), "progressive_passes": _rand(rng, 62, 90),
            "key_passes": _rand(rng, 50, 78), "assists": _rand(rng, 40, 68),
            "xa": _rand(rng, 40, 68), "dribbles": _rand(rng, 50, 75),
            "stamina": _rand(rng, 72, 94), "tackles": _rand(rng, 40, 68),
            "goals": _rand(rng, 25, 50),
        })
    elif role == "AM":
        s.update({
            "key_passes": _rand(rng, 68, 94), "assists": _rand(rng, 60, 90),
            "xa": _rand(rng, 62, 92), "dribbles": _rand(rng, 60, 88),
            "big_chances": _rand(rng, 55, 85), "goals": _rand(rng, 40, 70),
            "xg": _rand(rng, 40, 68), "agility": _rand(rng, 65, 90),
            "pass_accuracy": _rand(rng, 70, 88),
        })
    elif role in ("LW", "RW"):
        s.update({
            "dribbles": _rand(rng, 70, 95), "speed": _rand(rng, 78, 96),
            "goals": _rand(rng, 50, 80), "xg": _rand(rng, 48, 78),
            "assists": _rand(rng, 45, 75), "xa": _rand(rng, 45, 75),
            "key_passes": _rand(rng, 50, 78), "big_chances": _rand(rng, 50, 80),
            "agility": _rand(rng, 70, 93), "shots_on_target": _rand(rng, 45, 75),
        })
    elif role == "ST":
        s.update({
            "goals": _rand(rng, 68, 95), "xg": _rand(rng, 68, 95),
            "shots_on_target": _rand(rng, 65, 92), "big_chances": _rand(rng, 62, 92),
            "aerial_duels": _rand(rng, 45, 80), "strength": _rand(rng, 55, 85),
            "speed": _rand(rng, 55, 88), "dribbles": _rand(rng, 45, 75),
        })

    # Team-style flavour
    if style == "Possession":
        s["pass_accuracy"] = min(99, s["pass_accuracy"] + 6)
        s["progressive_passes"] = min(99, s["progressive_passes"] + 5)
    elif style == "Direct":
        s["aerial_duels"] = min(99, s["aerial_duels"] + 6)
        s["strength"] = min(99, s["strength"] + 4)
    elif style == "High Press":
        s["stamina"] = min(99, s["stamina"] + 6)
        s["tackles"] = min(99, s["tackles"] + 4)

    return s


def _market_value(rng, stats, age, position):
    """Fake market value in millions, loosely tied to quality and age."""
    if position == "Goalkeeper":
        quality = (stats["saves"] + stats["reflexes"] + stats["one_v_one"]) / 3
    elif position == "Defender":
        quality = (stats["tackles"] + stats["interceptions"] + stats["aerial_duels"]) / 3
    elif position == "Midfielder":
        quality = (stats["pass_accuracy"] + stats["key_passes"] + stats["progressive_passes"]) / 3
    else:
        quality = (stats["goals"] + stats["xg"] + stats["dribbles"]) / 3

    age_factor = 1.35 if age <= 23 else (1.0 if age <= 28 else 0.55)
    value = (quality ** 1.6) / 28 * age_factor * rng.uniform(0.8, 1.25)
    return round(max(0.5, value), 1)


def generate_mock_data(seed=SEED):
    """Returns (players_df, pass_df, def_df)."""
    rng = random.Random(seed)

    # Unique fake names for every player
    n_players = len(TEAMS) * len(ROLES)
    names = set()
    while len(names) < n_players:
        names.add(f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}")
    names = sorted(names)
    rng.shuffle(names)
    name_iter = iter(names)

    season_multiplier = 10
    players, pass_rows, def_rows = [], [], []

    for team, style in TEAMS.items():
        role_to_name = {}
        for role in ROLES:
            name = next(name_iter)
            role_to_name[role] = name
            position = ROLE_TO_POSITION[role]
            age = rng.randint(18, 34)
            nation, flag = rng.choice(NATIONALITIES)
            stats = _generate_stats(rng, role, style)

            players.append({
                "player_name": name,
                "team_name": team,
                "team_style": style,
                "role": role,
                "role_label": ROLE_LABELS[role],
                "position": position,
                "age": age,
                "nationality": nation,
                "flag": flag,
                "market_value": _market_value(rng, stats, age, position),
                **stats,
            })

        # Passing network edges
        for passer_role, targets in PASS_LOGIC.items():
            for target_role, base_weight in targets:
                weight = base_weight * season_multiplier
                xt_multiplier = 0.01
                if style == "Possession":
                    if passer_role in ["CB1", "CB2", "DM", "CM", "LB", "RB"] and \
                       target_role in ["CB1", "CB2", "DM", "CM", "LB", "RB"]:
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

                final_weight = max(1, int(weight * rng.uniform(0.7, 1.3)))
                total_xt = final_weight * xt_multiplier * rng.uniform(0.8, 1.2)

                pass_rows.append({
                    "passer_name": role_to_name[passer_role],
                    "receiver_name": role_to_name[target_role],
                    "passer_role": passer_role,
                    "receiver_role": target_role,
                    "team_name": team,
                    "pass_count": final_weight,
                    "total_xt": total_xt,
                })

        # Defensive synergy edges
        for def_role, targets in DEF_LOGIC.items():
            for target_role, base_weight in targets:
                weight = base_weight * season_multiplier
                if style == "High Press" and def_role in ["ST", "LW", "RW", "AM"]:
                    weight *= 2.0
                elif style == "Possession" and def_role in ["CB1", "CB2", "GK"]:
                    weight *= 1.5

                def_rows.append({
                    "passer_name": role_to_name[def_role],
                    "receiver_name": role_to_name[target_role],
                    "passer_role": def_role,
                    "receiver_role": target_role,
                    "team_name": team,
                    "synergy_score": max(1, int(weight * rng.uniform(0.5, 1.5))),
                })

    return pd.DataFrame(players), pd.DataFrame(pass_rows), pd.DataFrame(def_rows)
