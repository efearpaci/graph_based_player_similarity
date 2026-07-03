"""Custom CSS for the purple/black scouting UI."""

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"], .stApp, [data-testid="stSidebar"] {
    font-family: 'Inter', sans-serif;
}

h1, h2, h3, h4, .hero-title, .card-name, .metric-big {
    font-family: 'Space Grotesk', sans-serif !important;
}

.stApp {
    background:
        radial-gradient(1200px 500px at 85% -10%, rgba(139, 92, 246, 0.14), transparent 60%),
        radial-gradient(900px 400px at -10% 10%, rgba(217, 70, 239, 0.08), transparent 55%),
        #0a0812;
}

[data-testid="stSidebar"] {
    background: #0d0a16;
    border-right: 1px solid #241b38;
}
[data-testid="stSidebar"] hr { border-color: #241b38; }

/* ---------- hero ---------- */
.hero {
    padding: 1.6rem 2rem;
    border-radius: 20px;
    background: linear-gradient(120deg, #1a0f2e 0%, #130c22 55%, #0f0a1c 100%);
    border: 1px solid #2e2249;
    margin-bottom: 1.2rem;
    position: relative;
    overflow: hidden;
}
.hero::after {
    content: "";
    position: absolute;
    top: -60%; right: -10%;
    width: 55%; height: 220%;
    background: radial-gradient(closest-side, rgba(168, 85, 247, 0.22), transparent);
    pointer-events: none;
}
.hero-title {
    font-size: 2rem;
    font-weight: 700;
    margin: 0 0 0.3rem 0;
    background: linear-gradient(90deg, #e9d5ff 0%, #a855f7 55%, #d946ef 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.5px;
}
.hero-sub { color: #9d93b8; font-size: 0.95rem; margin: 0; max-width: 720px; }

.section-label {
    font-family: 'Space Grotesk', sans-serif;
    text-transform: uppercase;
    letter-spacing: 2.5px;
    font-size: 0.72rem;
    color: #a855f7;
    font-weight: 600;
    margin: 0.4rem 0 0.6rem 0;
}

/* ---------- target profile card ---------- */
.profile-card {
    background: linear-gradient(135deg, #191227 0%, #120d1e 100%);
    border: 1px solid #32254f;
    border-radius: 18px;
    padding: 1.3rem 1.6rem;
    margin-bottom: 0.6rem;
}
.profile-top { display: flex; align-items: center; gap: 0.9rem; flex-wrap: wrap; }
.avatar {
    width: 58px; height: 58px;
    border-radius: 16px;
    background: linear-gradient(135deg, #7c3aed, #d946ef);
    display: flex; align-items: center; justify-content: center;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.35rem; font-weight: 700; color: #fff;
    box-shadow: 0 4px 24px rgba(168, 85, 247, 0.35);
}
.card-name { font-size: 1.45rem; font-weight: 700; color: #f3eeff; margin: 0; }
.card-meta { color: #9d93b8; font-size: 0.88rem; margin-top: 2px; }

.pill {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.4px;
    margin-right: 6px;
    margin-top: 6px;
}
.pill-purple { background: rgba(168, 85, 247, 0.16); color: #c084fc; border: 1px solid rgba(168, 85, 247, 0.4); }
.pill-pink   { background: rgba(217, 70, 239, 0.13); color: #e879f9; border: 1px solid rgba(217, 70, 239, 0.35); }
.pill-dim    { background: rgba(157, 147, 184, 0.1);  color: #9d93b8; border: 1px solid rgba(157, 147, 184, 0.3); }

/* ---------- match cards ---------- */
.match-card {
    background: #14101f;
    border: 1px solid #2a2040;
    border-radius: 16px;
    padding: 1rem 1.1rem;
    height: 100%;
    transition: border-color .2s, transform .2s, box-shadow .2s;
    position: relative;
}
.match-card:hover {
    border-color: #a855f7;
    transform: translateY(-3px);
    box-shadow: 0 10px 34px rgba(168, 85, 247, 0.18);
}
.match-rank {
    position: absolute; top: 0.8rem; right: 0.9rem;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.78rem; font-weight: 700; color: #4c3a70;
}
.match-name { font-family: 'Space Grotesk', sans-serif; font-size: 1.02rem; font-weight: 600; color: #f3eeff; margin: 0 0 1px 0; }
.match-team { color: #9d93b8; font-size: 0.8rem; margin-bottom: 0.55rem; }
.sim-badge {
    display: inline-block;
    background: linear-gradient(90deg, #7c3aed, #d946ef);
    color: white;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 0.92rem;
    padding: 3px 12px;
    border-radius: 10px;
    margin-bottom: 0.5rem;
}
.sim-bar-track {
    background: #241b38; border-radius: 99px; height: 5px; margin-top: 0.55rem;
}
.sim-bar-fill {
    background: linear-gradient(90deg, #7c3aed, #d946ef);
    height: 5px; border-radius: 99px;
}

/* ---------- streamlit widget restyling ---------- */
.stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: 1px solid #241b38; }
.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 10px 10px 0 0;
    color: #9d93b8;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 500;
}
.stTabs [aria-selected="true"] { color: #e9d5ff !important; }

div[data-testid="stExpander"] {
    border: 1px solid #241b38;
    border-radius: 12px;
    background: #100c1a;
}

[data-testid="stMetric"] {
    background: #14101f;
    border: 1px solid #2a2040;
    border-radius: 14px;
    padding: 0.7rem 1rem;
}
[data-testid="stMetricLabel"] { color: #9d93b8; }

::-webkit-scrollbar { width: 9px; height: 9px; }
::-webkit-scrollbar-track { background: #0d0a16; }
::-webkit-scrollbar-thumb { background: #32254f; border-radius: 99px; }
::-webkit-scrollbar-thumb:hover { background: #a855f7; }

#MainMenu, footer { visibility: hidden; }
</style>
"""


def hero(title, subtitle):
    return f"""
    <div class="hero">
        <p class="hero-title">{title}</p>
        <p class="hero-sub">{subtitle}</p>
    </div>
    """


def profile_card(p):
    initials = "".join(w[0] for w in p["player_name"].replace(".", "").split()[:2]).upper()
    body = f"{int(p['height'])} cm · {p['foot']}-footed" if p.get("height") else ""
    return f"""
    <div class="profile-card">
      <div class="profile-top">
        <div class="avatar">{initials}</div>
        <div>
          <p class="card-name">{p['player_name']}</p>
          <p class="card-meta">{p['flag']} {p['nationality']} &nbsp;·&nbsp; {p['team_name']}
             ({p['league']}) &nbsp;·&nbsp; Age {p['age']} &nbsp;·&nbsp;
             {int(p['minutes']):,} min &nbsp;·&nbsp; {body}</p>
        </div>
      </div>
      <div style="margin-top:0.55rem;">
        <span class="pill pill-purple">{p['role_label']}</span>
        <span class="pill pill-pink">{p['archetype']}</span>
        <span class="pill pill-dim">{p['team_style']} system</span>
      </div>
    </div>
    """


def match_card(p, rank):
    sim_pct = max(0.0, min(1.0, p["similarity"])) * 100
    return f"""
    <div class="match-card">
      <div class="match-rank">#{rank}</div>
      <p class="match-name">{p['player_name']}</p>
      <p class="match-team">{p['flag']} {p['team_name']} · {p['league']}</p>
      <span class="sim-badge">{sim_pct:.1f}%</span>
      <div style="margin-top:0.3rem;">
        <span class="pill pill-pink">{p['archetype']}</span>
        <span class="pill pill-purple">{p['role_label']}</span>
      </div>
      <p class="card-meta" style="margin-top:0.5rem;">Age {p['age']} · {int(p['minutes']):,} min</p>
      <div class="sim-bar-track"><div class="sim-bar-fill" style="width:{sim_pct:.1f}%"></div></div>
    </div>
    """
