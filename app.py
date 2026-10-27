import os
import re
from html import escape as esc

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from ipl_engine import IPLEngine

st.set_page_config(page_title="IPL Match Winner Predictor", page_icon="🏏", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "matches.csv")
DELIV_PATH = os.path.join(BASE_DIR, "deliveries.csv")  # Playing 11 / POTM / Live features ke liye

# Internet image (Wikimedia Commons). Chalane ke liye internet chahiye.
HERO_IMG = (
    "https://commons.wikimedia.org/wiki/Special:FilePath/"
    "Cricket_ball_at_Southwater_CC%2C_in_Southwater%2C_West_Sussex%2C_England.jpg?width=500"
)
HERO_PAGE = (
    "https://commons.wikimedia.org/wiki/File:"
    "Cricket_ball_at_Southwater_CC,_in_Southwater,_West_Sussex,_England.jpg"
)

# Purane team names ko naye naam me badalna
TEAM_RENAMES = {
    "Delhi Daredevils": "Delhi Capitals",
    "Deccan Chargers": "Sunrisers Hyderabad",
    "Kings XI Punjab": "Punjab Kings",
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
    "Rising Pune Supergiants": "Rising Pune Supergiant",
}
ACTIVE_TEAMS = [
    "Chennai Super Kings",
    "Mumbai Indians",
    "Kolkata Knight Riders",
    "Royal Challengers Bengaluru",
    "Delhi Capitals",
    "Punjab Kings",
    "Rajasthan Royals",
    "Sunrisers Hyderabad",
    "Gujarat Titans",
    "Lucknow Super Giants",
]
SHORT = {
    "Chennai Super Kings": "CSK",
    "Mumbai Indians": "MI",
    "Kolkata Knight Riders": "KKR",
    "Royal Challengers Bengaluru": "RCB",
    "Delhi Capitals": "DC",
    "Punjab Kings": "PBKS",
    "Rajasthan Royals": "RR",
    "Sunrisers Hyderabad": "SRH",
    "Gujarat Titans": "GT",
    "Lucknow Super Giants": "LSG",
}

# Alag-alag Kaggle datasets me column ke naam thode alag hote hain, unhe yahan match kiya jaata hai
ALIASES = {
    "team1": ["team1"],
    "team2": ["team2"],
    "toss_winner": ["tosswinner"],
    "toss_decision": ["tossdecision"],
    "venue": ["venue"],
    "winner": ["winner", "winningteam", "matchwinner"],
}
FEATURES = ["team1", "team2", "venue", "toss_decision", "toss_by_team1"]

# ------------------------------------------------------------------ STYLE
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Rajdhani:wght@600;700&display=swap');
:root {
    --gold: #fcd34d; --amber: #f59e0b; --night: #050b14; --line: rgba(255,255,255,.12);
    --card: rgba(255,255,255,.055); --txt: #f1f5f9; --muted: #94a3b8;
}
html, body, .stApp, [data-testid="stAppViewContainer"] { font-family: 'Inter', system-ui, sans-serif; }
.stApp {
    background:
        linear-gradient(rgba(255,255,255,.028) 1px, transparent 1px) 0 0 / 52px 52px,
        linear-gradient(90deg, rgba(255,255,255,.028) 1px, transparent 1px) 0 0 / 52px 52px,
        radial-gradient(900px 520px at 10% 0%, rgba(16,185,129,.30), transparent 62%),
        radial-gradient(800px 520px at 95% 8%, rgba(245,158,11,.20), transparent 62%),
        radial-gradient(900px 600px at 50% 115%, rgba(59,130,246,.16), transparent 62%),
        #050b14;
    background-attachment: fixed;
}
[data-testid="stAppViewContainer"], .main { background: transparent !important; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 3.2rem; max-width: 1240px; }
h1, h2, h3, h4, label, .stMarkdown p, .stCaption, [data-testid="stMetricLabel"] { color: #f1f5f9 !important; }
[data-testid="stWidgetLabel"] p { color: #cbd5e1 !important; font-weight: 600; }
[data-testid="stCaptionContainer"], .stCaption { color: #94a3b8 !important; }
[data-testid="stMetricValue"] { color: #fcd34d !important; font-family: 'Rajdhani', sans-serif; font-weight: 700; font-size: 2rem; }

/* ---------- sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(8,24,38,.97), rgba(5,11,20,.99));
    border-right: 1px solid var(--line);
}
.brand { display:flex; align-items:center; gap:.7rem; margin:.4rem 0 1rem; }
.brand .logo { width:44px; height:44px; border-radius:14px; display:grid; place-items:center; font-size:1.5rem;
    background: linear-gradient(135deg,#fcd34d,#f59e0b); box-shadow: 0 8px 22px rgba(245,158,11,.4); }
.brand b { font-family:'Rajdhani',sans-serif; font-size:1.35rem; line-height:1.1; display:block; color:#f1f5f9; }
.brand small { color:#94a3b8; }

/* ---------- hero */
@keyframes bounce { 0%,100% { transform: translateY(0) rotate(0deg); } 50% { transform: translateY(-12px) rotate(20deg); } }
.beat { display:inline-block; animation: bounce 1.6s infinite; }
.hero-title {
    font-family:'Rajdhani',sans-serif; font-size: clamp(2.6rem, 5.8vw, 4.8rem) !important; font-weight: 700 !important;
    line-height: 1.0 !important; margin: 0 !important; letter-spacing: .01em;
}
.hero-title .grad, .hero-title .beat { font-size: inherit !important; font-weight: inherit !important; }
.grad {
    background: linear-gradient(90deg, #fcd34d, #fb923c, #f43f5e);
    -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;
}
.subtitle { color: #cbd5e1; font-size: 1.1rem; margin-top: .9rem; max-width: 34rem; line-height: 1.55; }
.chips { display:flex; flex-wrap:wrap; gap:.6rem; margin:1.1rem 0 .4rem; }
.chip { padding:.5rem .95rem; border-radius:14px; background: var(--card); border:1px solid var(--line); backdrop-filter: blur(8px); }
.chip b { display:block; font-family:'Rajdhani',sans-serif; font-size:1.4rem; color:#fcd34d; line-height:1.1; }
.chip small { color:#94a3b8; font-size:.78rem; }
.author {
    display:inline-block; margin-top:.8rem; padding:.3rem .95rem; border-radius:999px; font-weight:600; color:#1a1a1a;
    background: linear-gradient(90deg,#fcd34d,#f59e0b); box-shadow: 0 4px 18px rgba(245,158,11,.4);
}
@keyframes floaty { 0%,100% { transform: translateY(0); } 50% { transform: translateY(-10px); } }
.hero-img {
    display:block; margin:0 auto; width:100%; max-width:320px; border-radius:50%; animation: floaty 4s ease-in-out infinite;
    box-shadow: 0 0 60px rgba(245,158,11,.5), 0 0 120px rgba(245,158,11,.2); border:2px solid rgba(255,255,255,.2);
}
.credit { color:#94a3b8; font-size:.8rem; text-align:center; margin-top:.5rem; }
.credit a { color:#fcd34d; }

/* ---------- tabs */
div[data-baseweb="tab-list"] {
    gap:.35rem; background: rgba(255,255,255,.05); border:1px solid var(--line); padding:.35rem;
    border-radius:18px; backdrop-filter: blur(10px); overflow-x:auto;
}
button[data-baseweb="tab"] { height:auto; padding:.55rem 1.05rem; border-radius:13px; background:transparent; }
button[data-baseweb="tab"] p { color:#cbd5e1 !important; font-weight:600; }
button[data-baseweb="tab"]:hover { background: rgba(255,255,255,.07); }
button[data-baseweb="tab"][aria-selected="true"] {
    background: linear-gradient(90deg,#fcd34d,#f59e0b); box-shadow: 0 6px 18px rgba(245,158,11,.35);
}
button[data-baseweb="tab"][aria-selected="true"] p { color:#1a1a1a !important; }
div[data-baseweb="tab-highlight"], div[data-baseweb="tab-border"] { display:none; }

/* ---------- inputs */
div[data-baseweb="select"] > div, div[data-baseweb="input"] > div, div[data-baseweb="base-input"] {
    background: rgba(255,255,255,.07) !important; border:1px solid var(--line) !important; border-radius:12px !important; color:#f1f5f9 !important;
}
div[data-baseweb="select"] > div:hover, div[data-baseweb="input"] > div:hover { border-color: rgba(252,211,77,.55) !important; }
span[data-baseweb="tag"] { background: linear-gradient(90deg,#f59e0b,#fb923c) !important; border-radius:8px !important; }
span[data-baseweb="tag"] span { color:#1a1a1a !important; font-weight:600; }
div[role="radiogroup"] { gap:.5rem; }
div[role="radiogroup"] label {
    background: rgba(255,255,255,.06); border:1px solid var(--line); border-radius:12px; padding:.35rem .85rem; transition: all .15s ease;
}
div[role="radiogroup"] label:has(input:checked) { border-color:#fcd34d; background: rgba(252,211,77,.14); }
div.stButton > button, div[data-testid="stFormSubmitButton"] button {
    width:100%; border:none; border-radius:14px; padding:.7rem 1rem; font-size:1.05rem; font-weight:700; color:#1a1a1a;
    background: linear-gradient(90deg,#fcd34d,#f59e0b); box-shadow: 0 6px 20px rgba(245,158,11,.4); transition: transform .15s ease;
}
div.stButton > button p, div[data-testid="stFormSubmitButton"] button p { color:#1a1a1a !important; font-weight:700; }
div.stButton > button:hover, div[data-testid="stFormSubmitButton"] button:hover { transform: translateY(-2px) scale(1.01); }
div[data-testid="stForm"] {
    background: rgba(255,255,255,.06); border:1px solid rgba(255,255,255,.14); border-radius:22px; padding:1.6rem; backdrop-filter: blur(10px);
}
[data-testid="stMetric"] { background: var(--card); border:1px solid var(--line); border-radius:16px; padding:.9rem 1.1rem; backdrop-filter: blur(8px); }
[data-testid="stDataFrame"] { border-radius:14px; overflow:hidden; border:1px solid var(--line); }
[data-testid="stExpander"] { background: var(--card); border:1px solid var(--line) !important; border-radius:16px; }
[data-testid="stAlert"] { border-radius:14px; }

/* ---------- section header */
.sec { display:flex; align-items:center; gap:.75rem; margin:1.5rem 0 .7rem; }
.sec .ic { width:40px; height:40px; border-radius:12px; display:grid; place-items:center; font-size:1.2rem;
    background: linear-gradient(135deg, rgba(252,211,77,.25), rgba(251,146,60,.18)); border:1px solid rgba(252,211,77,.35); }
.sec h3 { margin:0 !important; padding:0 !important; font-family:'Rajdhani',sans-serif; font-size:1.7rem !important; font-weight:700 !important; line-height:1.1; }
.sec small { color:#94a3b8; font-size:.9rem; }

/* ---------- result card */
.result-card {
    position:relative; overflow:hidden; border-radius:22px; padding:1.3rem 1.8rem; margin:1rem 0; color:#fff;
    border:1px solid var(--line); border-left:6px solid var(--c,#fcd34d);
    background: linear-gradient(135deg, rgba(255,255,255,.09), rgba(255,255,255,.03));
    box-shadow: 0 18px 50px -22px var(--c,#fcd34d); backdrop-filter: blur(12px);
}
.result-card::after { content: var(--icon,'🏆'); position:absolute; right:1.4rem; top:50%; transform:translateY(-50%); font-size:4.4rem; opacity:.13; }
.result-card .small { color:#e2e8f0; font-size:1rem; }
.result-card .big { font-family:'Rajdhani',sans-serif; font-size:2.7rem; font-weight:700; line-height:1.1; color:#fff; }

/* ---------- VS banner + probability bar */
.vs { display:grid; grid-template-columns: 1fr auto 1fr; align-items:center; gap:1rem; margin:.4rem 0 1rem; }
.vs .side { text-align:center; }
.vs .tname { font-weight:600; color:#e2e8f0; margin-top:.5rem; }
.vs .pct { font-family:'Rajdhani',sans-serif; font-size:2.8rem; font-weight:700; line-height:1.05; }
.vs-pill { font-family:'Rajdhani',sans-serif; font-weight:700; font-size:1.3rem; padding:.45rem .9rem; border-radius:999px;
    background: rgba(255,255,255,.08); border:1px solid var(--line); color:#cbd5e1; }
.badge { margin:0 auto; border-radius:50%; display:grid; place-items:center; font-family:'Rajdhani',sans-serif; font-weight:700;
    letter-spacing:.03em; border:2px solid rgba(255,255,255,.35); }
@keyframes grow { from { width: 0; } }
.pbar { display:flex; height:22px; border-radius:999px; overflow:hidden; background: rgba(255,255,255,.08); border:1px solid var(--line); }
.pbar .seg { display:flex; align-items:center; justify-content:center; overflow:hidden; animation: grow .9s ease-out; }
.pbar .seg span { font-size:.78rem; font-weight:700; color:#0b1220; }
.pbar-lab { display:flex; justify-content:space-between; margin:.35rem .3rem 1rem; font-weight:600; font-size:.9rem; }

/* ---------- podium */
.podium { display:grid; grid-template-columns: repeat(3, 1fr); gap:.9rem; margin:.5rem 0 1rem; }
.pod { border-radius:20px; padding:1rem 1.1rem; border:1px solid var(--line); border-top:4px solid var(--c);
    background: linear-gradient(160deg, rgba(255,255,255,.09), rgba(255,255,255,.03)); box-shadow: 0 14px 40px -22px var(--c); transition: transform .2s ease; }
.pod:hover { transform: translateY(-4px); }
.pod .medal { font-size:1.6rem; }
.pod .pn { font-weight:700; font-size:1.05rem; color:#f1f5f9; margin-top:.2rem; }
.pod .pt { color:#94a3b8; font-size:.85rem; }
.pod .pp { font-family:'Rajdhani',sans-serif; font-size:2.1rem; font-weight:700; color:var(--c); line-height:1.1; margin-top:.3rem; }
.pod .pmini { height:6px; border-radius:99px; background: rgba(255,255,255,.1); margin-top:.4rem; overflow:hidden; }
.pod .pmini i { display:block; height:100%; background: var(--c); border-radius:99px; }

/* ---------- live scoreboard */
.scoreboard { display:grid; grid-template-columns: 1.1fr 1fr; gap:1.2rem; padding:1.2rem 1.6rem; margin:.6rem 0 1rem; border-radius:22px;
    background: linear-gradient(135deg, rgba(8,24,38,.9), rgba(11,61,46,.55)); border:1px solid var(--line); box-shadow: 0 18px 50px -24px #000; }
.live { display:inline-flex; align-items:center; gap:.45rem; font-weight:700; font-size:.8rem; color:#fecaca; background: rgba(239,68,68,.16);
    border:1px solid rgba(239,68,68,.4); padding:.15rem .6rem; border-radius:999px; }
@keyframes pulse { 0%,100% { opacity:1; transform:scale(1); } 50% { opacity:.35; transform:scale(1.5); } }
.live .dot { width:8px; height:8px; border-radius:50%; background:#ef4444; animation: pulse 1.3s infinite; }
.sb-team { font-weight:700; font-size:1.1rem; margin-top:.5rem; }
.sb-score { font-family:'Rajdhani',sans-serif; font-size:4.2rem; font-weight:700; line-height:1; color:#fff; }
.sb-score span { font-size:2.4rem; color:#94a3b8; }
.sb-ov { color:#94a3b8; }
.sb-right { display:grid; grid-template-columns: 1fr 1fr; gap:.6rem; align-content:center; }
.kv { background: rgba(255,255,255,.06); border:1px solid var(--line); border-radius:14px; padding:.5rem .8rem; }
.kv small { display:block; color:#94a3b8; font-size:.75rem; }
.kv b { font-family:'Rajdhani',sans-serif; font-size:1.5rem; color:#fcd34d; }

/* ---------- sidebar menu (left navigation) */
[data-testid="stSidebar"] div[role="radiogroup"] { flex-direction: column; gap:.4rem; }
[data-testid="stSidebar"] div[role="radiogroup"] label { width:100%; padding:.65rem .95rem; border-radius:14px; }
[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child { display:none; }
[data-testid="stSidebar"] div[role="radiogroup"] label:hover { border-color: rgba(252,211,77,.5); }
[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
    background: linear-gradient(90deg,#fcd34d,#f59e0b); border-color: transparent; box-shadow: 0 6px 18px rgba(245,158,11,.35);
}
[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p { color:#1a1a1a !important; font-weight:700; }
[data-testid="stSidebar"] div[role="radiogroup"] label p { font-weight:600; }

.footer { text-align:center; color:#94a3b8; margin-top:2.5rem; font-size:.9rem; }
@media (max-width: 720px) {
    .podium, .scoreboard { grid-template-columns: 1fr; }
    .vs .pct { font-size:2.1rem; }
    .sb-score { font-size:3.3rem; }
}
@media (prefers-reduced-motion: reduce) { .beat, .hero-img, .live .dot, .pbar .seg { animation: none; } }
</style>
""",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------ UI HELPERS
TEAM_COLORS = {
    "Chennai Super Kings": "#fdb913", "Mumbai Indians": "#2b7fff", "Kolkata Knight Riders": "#8b5cf6",
    "Royal Challengers Bengaluru": "#ef4444", "Delhi Capitals": "#3b82f6", "Punjab Kings": "#f43f5e",
    "Rajasthan Royals": "#ec4899", "Sunrisers Hyderabad": "#fb923c", "Gujarat Titans": "#38bdf8",
    "Lucknow Super Giants": "#14b8a6",
}


def _rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def rgba(h, a):
    r, g, b = _rgb(h)
    return f"rgba({r},{g},{b},{a})"


def _dist(x, y):
    return sum((p - q) ** 2 for p, q in zip(_rgb(x), _rgb(y))) ** 0.5


def team_color(team):
    return TEAM_COLORS.get(team, "#94a3b8")


def pair_colors(a, b):
    """Dono teams ke rang. Agar kaafi milte-julte hon to doosri team ko alag rang milta hai."""
    ca, cb = team_color(a), team_color(b)
    if _dist(ca, cb) < 120:
        for alt in ("#22c55e", "#f59e0b", "#a78bfa", "#38bdf8"):
            if _dist(ca, alt) >= 120:
                cb = alt
                break
    return ca, cb


def put(markup):
    st.markdown(markup, unsafe_allow_html=True)


def section(title, icon="✨", sub=""):
    small = f"<small>{esc(sub)}</small>" if sub else ""
    put(f'<div class="sec"><div class="ic">{icon}</div><div><h3>{esc(title)}</h3>{small}</div></div>')


def badge(team, color, size=64):
    r, g, b = _rgb(color)
    ink = "#0b1220" if (0.299 * r + 0.587 * g + 0.114 * b) > 150 else "#ffffff"
    code = esc(SHORT.get(team, team[:3].upper()))
    return (f'<div class="badge" style="width:{size}px;height:{size}px;color:{ink};'
            f'background:linear-gradient(135deg,{color},{rgba(color, .6)});'
            f'box-shadow:0 8px 28px {rgba(color, .45)};font-size:{size * 0.32:.0f}px">{code}</div>')


def result_card(label, big, sub, color="#fcd34d", icon="🏆"):
    put(f'<div class="result-card" style="--c:{color};--icon:\'{icon}\'">'
        f'<div class="small">{esc(label)}</div><div class="big">{esc(big)}</div>'
        f'<div class="small">{esc(sub)}</div></div>')


def prob_bar(a, pa, b, ca, cb):
    wa = pa * 100
    wb = 100 - wa
    put('<div class="pbar">'
        f'<div class="seg" style="width:{wa:.1f}%;background:linear-gradient(90deg,{rgba(ca, .85)},{ca})"><span>{wa:.0f}%</span></div>'
        f'<div class="seg" style="width:{wb:.1f}%;background:linear-gradient(90deg,{cb},{rgba(cb, .85)})"><span>{wb:.0f}%</span></div>'
        '</div>'
        f'<div class="pbar-lab"><span style="color:{ca}">{esc(a)}</span><span style="color:{cb}">{esc(b)}</span></div>')


def vs_banner(a, b, pa, ca, cb):
    put('<div class="vs">'
        f'<div class="side">{badge(a, ca)}<div class="tname">{esc(a)}</div>'
        f'<div class="pct" style="color:{ca}">{pa * 100:.1f}%</div></div>'
        '<div><div class="vs-pill">VS</div></div>'
        f'<div class="side">{badge(b, cb)}<div class="tname">{esc(b)}</div>'
        f'<div class="pct" style="color:{cb}">{(1 - pa) * 100:.1f}%</div></div></div>')


def podium(top3, ra, ca, cb):
    medals = ["🥇", "🥈", "🥉"]
    cards = ""
    for i, (_, r) in enumerate(top3.iterrows()):
        c = ca if r["Team"] == ra else cb
        cards += (f'<div class="pod" style="--c:{c}"><div class="medal">{medals[i]}</div>'
                  f'<div class="pn">{esc(str(r["Player"]))}</div>'
                  f'<div class="pt">{esc(SHORT.get(r["Team"], r["Team"]))}</div>'
                  f'<div class="pp">{r["POTM %"]:.1f}%</div>'
                  f'<div class="pmini"><i style="width:{min(r["POTM %"] * 5, 100):.0f}%"></i></div></div>')
    put(f'<div class="podium">{cards}</div>')


def scoreboard(team, color, score, wkts, balls, items):
    kv = "".join(f'<div class="kv"><small>{esc(k)}</small><b>{esc(str(v))}</b></div>' for k, v in items)
    put('<div class="scoreboard"><div>'
        '<div class="live"><span class="dot"></span>LIVE</div>'
        f'<div class="sb-team" style="color:{color}">{esc(team)}</div>'
        f'<div class="sb-score">{score}<span>/{wkts}</span></div>'
        f'<div class="sb-ov">{balls // 6}.{balls % 6} overs</div></div>'
        f'<div class="sb-right">{kv}</div></div>')


# ------------------------------------------------------------------ DATA
@st.cache_data
def load_data(path_or_file):
    return pd.read_csv(path_or_file)


def prepare_data(raw: pd.DataFrame):
    """Column ke naam sahi karke data saaf karta hai. (clean_df, missing_columns) deta hai."""
    clean = {c: re.sub(r"[^a-z0-9]", "", str(c).lower()) for c in raw.columns}
    rename = {}
    for target, names in ALIASES.items():
        for col, c in clean.items():
            if c in names:
                rename[col] = target
                break
    df = raw.rename(columns=rename)
    missing = [k for k in ALIASES if k not in df.columns]
    if missing:
        return None, missing

    df = df[list(ALIASES)].copy()
    for c in ["team1", "team2", "toss_winner", "winner"]:
        df[c] = df[c].astype(str).str.strip().replace(TEAM_RENAMES)
    df["venue"] = df["venue"].astype(str).str.split(",").str[0].str.strip()
    df["toss_decision"] = df["toss_decision"].astype(str).str.strip().str.lower()
    # Sirf wahi matches jinka result nikla (no result / tie hata do)
    df = df[(df["winner"] == df["team1"]) | (df["winner"] == df["team2"])]
    return df.reset_index(drop=True), []


def make_xy(df: pd.DataFrame):
    X = pd.DataFrame(
        {
            "team1": df["team1"],
            "team2": df["team2"],
            "venue": df["venue"],
            "toss_decision": df["toss_decision"],
            "toss_by_team1": (df["toss_winner"] == df["team1"]).astype(int).astype(str),
        }
    )
    y = (df["winner"] == df["team1"]).astype(int)
    return X, y


def swapped(df: pd.DataFrame):
    s = df.copy()
    s["team1"], s["team2"] = df["team2"], df["team1"]
    return s


@st.cache_resource
def train_model(df: pd.DataFrame):
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    # Team1/Team2 ka order matter na kare isliye data ki ulti copy bhi jodte hain
    train_aug = pd.concat([train_df, swapped(train_df)])
    test_aug = pd.concat([test_df, swapped(test_df)])

    model = Pipeline(
        [
            ("prep", ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), FEATURES)])),
            ("clf", LogisticRegression(C=0.5, max_iter=2000)),
        ]
    )
    model.fit(*make_xy(train_aug))
    Xt, yt = make_xy(test_aug)
    acc = accuracy_score(yt, model.predict(Xt))
    return model, acc


def win_probability(model, a, b, venue, toss_winner, decision):
    """Team A ke jeetne ki probability (dono orders ka average)."""
    row_ab = pd.DataFrame(
        [{"team1": a, "team2": b, "venue": venue, "toss_decision": decision,
          "toss_by_team1": str(int(toss_winner == a))}]
    )
    row_ba = pd.DataFrame(
        [{"team1": b, "team2": a, "venue": venue, "toss_decision": decision,
          "toss_by_team1": str(int(toss_winner == b))}]
    )
    p1 = model.predict_proba(row_ab)[0][1]
    p2 = model.predict_proba(row_ba)[0][1]
    return float((p1 + (1 - p2)) / 2)


def team_table(df):
    teams = pd.unique(df[["team1", "team2"]].values.ravel())
    rows = []
    for t in teams:
        played = int(((df["team1"] == t) | (df["team2"] == t)).sum())
        won = int((df["winner"] == t).sum())
        rows.append({"team": t, "played": played, "won": won,
                     "win_pct": round(100 * won / played, 1) if played else 0.0})
    return pd.DataFrame(rows).sort_values("win_pct", ascending=False).reset_index(drop=True)


def head_to_head(df, a, b):
    m = df[((df["team1"] == a) & (df["team2"] == b)) | ((df["team1"] == b) & (df["team2"] == a))]
    return len(m), int((m["winner"] == a).sum()), int((m["winner"] == b).sum())


def venue_record(df, team, venue):
    m = df[((df["team1"] == team) | (df["team2"] == team)) & (df["venue"] == venue)]
    return len(m), int((m["winner"] == team).sum())


# ------------------------------------------------------------------ CHARTS
def style_fig(fig, height=380):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0", family="Inter, sans-serif"), height=height, margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(bgcolor="rgba(255,255,255,0.08)", bordercolor="rgba(255,255,255,.12)", borderwidth=1),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,.07)", zerolinecolor="rgba(255,255,255,.12)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,.07)", zerolinecolor="rgba(255,255,255,.12)")
    return fig


def cricket_ball_3d():
    """Mouse se ghumne wali 3D cricket ball."""
    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, np.pi, 40)
    U, V = np.meshgrid(u, v)
    X = np.cos(U) * np.sin(V)
    Y = np.sin(U) * np.sin(V)
    Z = np.cos(V)

    fig = go.Figure(
        go.Surface(
            x=X, y=Y, z=Z, surfacecolor=Z, showscale=False, hoverinfo="skip",
            colorscale=[[0, "#7f1d1d"], [0.5, "#b91c1c"], [1, "#ef4444"]],
            lighting=dict(ambient=0.45, diffuse=0.9, specular=0.6, roughness=0.4),
        )
    )
    th = np.linspace(0, 2 * np.pi, 200)
    fig.add_trace(
        go.Scatter3d(
            x=1.01 * np.cos(th), y=1.01 * np.sin(th), z=np.zeros_like(th),
            mode="lines", line=dict(color="#f8fafc", width=6),
            hoverinfo="skip", showlegend=False,
        )
    )
    ts = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    for zoff in (-0.09, 0.09):
        r = np.sqrt(1 - zoff**2) * 1.012
        fig.add_trace(
            go.Scatter3d(
                x=r * np.cos(ts), y=r * np.sin(ts), z=np.full_like(ts, zoff),
                mode="markers", marker=dict(size=2.5, color="#f1f5f9"),
                hoverinfo="skip", showlegend=False,
            )
        )
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
            aspectmode="data", camera=dict(eye=dict(x=1.5, y=1.2, z=0.9)),
        ),
        paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=0, b=0), height=420,
    )
    return fig


def teams_3d(tt):
    tt = tt[tt["played"] >= 10]
    fig = go.Figure(
        go.Scatter3d(
            x=tt["played"], y=tt["won"], z=tt["win_pct"],
            mode="markers+text", text=[SHORT.get(t, t) for t in tt["team"]],
            textposition="top center", textfont=dict(color="white", size=12),
            marker=dict(
                size=np.clip(tt["played"] / 8, 6, 22), color=tt["win_pct"],
                colorscale="YlOrRd", opacity=0.9, showscale=True,
                colorbar=dict(title="Win %"),
            ),
            customdata=tt["team"],
            hovertemplate="%{customdata}<br>Matches: %{x}<br>Wins: %{y}<br>Win %: %{z}<extra></extra>",
        )
    )
    fig.update_layout(
        scene=dict(
            xaxis_title="Matches khele", yaxis_title="Jeete", zaxis_title="Win %",
            bgcolor="rgba(0,0,0,0)",
        ),
        paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white"),
        height=560, margin=dict(l=0, r=0, t=10, b=0),
    )
    return fig


# ------------------------------------------------------------------ LOAD DATA
if os.path.exists(DATA_PATH):
    raw = load_data(DATA_PATH)
else:
    uploaded = st.file_uploader("matches.csv upload karo", type="csv")
    if uploaded is None:
        st.info("app.py ke same folder me matches.csv rakho ya yahan upload karo.")
        st.stop()
    raw = load_data(uploaded)

df, missing = prepare_data(raw)
if missing:
    st.error(
        "Dataset me ye columns chahiye: team1, team2, toss_winner, toss_decision, venue, winner. "
        f"Ye nahi mile: {', '.join(missing)}"
    )
    st.stop()
if len(df) < 50:
    st.error("Dataset me bahut kam matches hain. IPL ka poora matches.csv use karo.")
    st.stop()

model, accuracy = train_model(df)
tt = team_table(df)

all_teams = sorted(pd.unique(df[["team1", "team2"]].values.ravel()))
team_choices = [t for t in ACTIVE_TEAMS if t in all_teams]
if len(team_choices) < 2:
    team_choices = all_teams
venues = df["venue"].value_counts().index.tolist()


@st.cache_resource(show_spinner=False)
def _build_engine(matches_path, deliv_path, mtime_m, mtime_d):
    return IPLEngine(pd.read_csv(matches_path), pd.read_csv(deliv_path))


def need_engine(show=True):
    """Playing 11 / POTM / Live tabs ka engine. Pehli baar train hone me kuch seconds lagte hain."""
    if not (os.path.exists(DATA_PATH) and os.path.exists(DELIV_PATH)):
        if show:
            st.info("Is feature ke liye `matches.csv` aur `deliveries.csv` dono app.py ke same folder me chahiye.")
        return None
    try:
        with st.spinner("Player aur live-match models ban rahe hain (pehli baar kuch seconds lagte hain)..."):
            return _build_engine(DATA_PATH, DELIV_PATH,
                                 os.path.getmtime(DATA_PATH), os.path.getmtime(DELIV_PATH))
    except Exception as exc:
        if show:
            st.error(f"Engine load nahi ho paya: {exc}")
        return None


COLOR_A, COLOR_B = "#f59e0b", "#22c55e"

# ------------------------------------------------------------------ SIDEBAR
st.sidebar.markdown(
    '<div class="brand"><div class="logo">🏏</div><div><b>IPL Predictor</b><small>ML se match ka andaza</small></div></div>',
    unsafe_allow_html=True,
)
PAGES = ["🏏 Predict", "🧢 Playing 11 + POTM", "⏱️ Live Match", "📊 Team Analysis", "🌐 3D Cricket", "ℹ️ About"]
page = st.sidebar.radio("Menu", PAGES, key="nav", label_visibility="collapsed")
st.sidebar.markdown("---")
st.sidebar.metric("Base model accuracy", f"{accuracy * 100:.1f}%")
st.sidebar.metric("Matches in dataset", len(df))
st.sidebar.markdown("---")
st.sidebar.caption("Tabs me Playing 11, Player of the Match aur Live prediction bhi hai.")
st.sidebar.caption("By Govind Khatile")

# ------------------------------------------------------------------ HERO
left, right = st.columns([7, 4])
with left:
    st.markdown(
        '<div class="hero-title"><span class="beat">🏏</span> '
        '<span class="grad">IPL Match<br>Winner Predictor</span></div>'
        '<div class="subtitle">Teams, Playing 11, toss aur venue batao. Machine Learning batayega kaun jeetega, '
        'Player of the Match kaun banega, aur live score se match ka haal.</div>'
        f'<div class="chips"><div class="chip"><b>{len(df):,}</b><small>matches</small></div>'
        f'<div class="chip"><b>{df["venue"].nunique()}</b><small>stadiums</small></div>'
        f'<div class="chip"><b>{len(team_choices)}</b><small>active teams</small></div>'
        f'<div class="chip"><b>{accuracy * 100:.1f}%</b><small>base model accuracy</small></div></div>'
        '<span class="author">Developed by Govind Khatile</span>',
        unsafe_allow_html=True,
    )
with right:
    st.markdown(
        f'<img class="hero-img" src="{HERO_IMG}" alt="Cricket ball" '
        f'onerror="this.style.display=\'none\'">'
        f'<div class="credit">Photo: <a href="{HERO_PAGE}" target="_blank">Wikimedia Commons</a></div>',
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------------ TAB 1: PREDICT
if page == PAGES[0]:
    section("Match ki details", "🏏")

    with st.form("match_form"):
        c1, c2 = st.columns(2)
        with c1:
            team_a = st.selectbox("Team A", team_choices, index=0)
            venue = st.selectbox("Venue (stadium)", venues, index=0)
            toss_by = st.radio("Toss kisne jeeta?", ["Team A", "Team B"], horizontal=True)
        with c2:
            team_b = st.selectbox("Team B", team_choices, index=1)
            decision = st.radio(
                "Toss jeetne wali team ne kya chuna?",
                ["bat", "field"],
                format_func=lambda x: "Pehle Batting 🏏" if x == "bat" else "Pehle Bowling 🥎",
                horizontal=True,
            )
        submitted = st.form_submit_button("🏏 Predict Winner")

    if submitted:
        if team_a == team_b:
            st.session_state.pop("pred", None)
            st.error("Dono teams alag-alag chuno.")
        else:
            toss_winner = team_a if toss_by == "Team A" else team_b
            p = win_probability(model, team_a, team_b, venue, toss_winner, decision)
            st.session_state["pred"] = {
                "a": team_a, "b": team_b, "venue": venue, "p": p,
                "toss": toss_winner, "decision": decision,
            }

    if "pred" in st.session_state:
        r = st.session_state["pred"]
        a, b, pa = r["a"], r["b"], r["p"]
        winner = a if pa >= 0.5 else b
        conf = max(pa, 1 - pa) * 100

        ca, cb = pair_colors(a, b)
        section("Prediction", "🏆", "Model ke hisaab se jeetne ke chances")
        vs_banner(a, b, pa, ca, cb)
        prob_bar(a, pa, b, ca, cb)
        result_card("Jeetne ke zyada chance:", winner, f"{conf:.1f}% chance (model ke hisaab se)", team_color(winner))

        n, wa, wb = head_to_head(df, a, b)
        va_n, va_w = venue_record(df, a, r["venue"])
        vb_n, vb_w = venue_record(df, b, r["venue"])
        m1, m2, m3 = st.columns(3)
        m1.metric("🤝 Head-to-Head", f"{wa} - {wb}", f"{n} matches ({SHORT.get(a, a)} vs {SHORT.get(b, b)})", delta_color="off")
        m2.metric(f"🏟️ {SHORT.get(a, a)} is venue par", f"{va_w}/{va_n} jeete", delta_color="off")
        m3.metric(f"🏟️ {SHORT.get(b, b)} is venue par", f"{vb_w}/{vb_n} jeete", delta_color="off")
        st.caption(
            "Ye purane data par based ek ML andaza hai. Cricket me form, injury, pitch aur "
            "us din ka khel bhi matter karta hai, isliye asli result alag ho sakta hai."
        )

# ------------------------------------------------------------------ TAB: PLAYING 11 + POTM
if page == PAGES[1]:
    section("Playing 11 + conditions se Winner aur Player of the Match", "🧢")
    eng = need_engine()
    if eng is not None:
        xc1, xc2 = st.columns(2)
        xa = xc1.selectbox("Team A", team_choices, index=0, key="xi_team_a")
        xb = xc2.selectbox("Team B", team_choices, index=1, key="xi_team_b")
        xv1, xv2, xv3 = st.columns([2, 1, 1])
        xv = xv1.selectbox("Venue (stadium)", venues, index=0, key="xi_venue")
        xt = xv2.radio("Toss kisne jeeta?", ["Team A", "Team B"], horizontal=True, key="xi_toss")
        xd = xv3.radio(
            "Toss ke baad?", ["bat", "field"], horizontal=True, key="xi_dec",
            format_func=lambda x: "Batting 🏏" if x == "bat" else "Bowling 🥎",
        )
        vn, v_avg, v_bf = eng.venue_info(xv)
        st.caption(
            f"🏟️ {xv}: {vn} matches · pehli innings ka average score {v_avg:.0f} · "
            f"pehle batting karne wali team {v_bf:.0f}% matches jeeti."
        )

        if xa == xb:
            st.warning("Dono teams alag-alag chuno.")
        else:
            xp1, xp2 = st.columns(2)
            with xp1:
                pick_a = st.multiselect(
                    f"{xa} ka Playing 11", eng.player_options(xa), default=eng.default_xi(xa),
                    max_selections=11, key=f"xi_pick_a_{xa}",
                )
            with xp2:
                pick_b = st.multiselect(
                    f"{xb} ka Playing 11", eng.player_options(xb), default=eng.default_xi(xb),
                    max_selections=11, key=f"xi_pick_b_{xb}",
                )
            st.caption(
                "Default me har team ke pichle season me sabse zyada khelne wale 11 players hain. "
                "Apna asli Playing 11 chuno. Dataset (2008–2024) me jo player nahi hai, uski rating average maani jaati hai."
            )

            if st.button("🏏 Predict (Playing 11 ke saath)", type="primary", key="xi_btn"):
                if len(pick_a) != 11 or len(pick_b) != 11:
                    st.error(f"Dono teams me exactly 11 players chuno (abhi {len(pick_a)} aur {len(pick_b)} hain).")
                elif set(pick_a) & set(pick_b):
                    st.error("Ek hi player dono teams me nahi ho sakta: " + ", ".join(sorted(set(pick_a) & set(pick_b))))
                else:
                    toss_w = xa if xt == "Team A" else xb
                    xpa, sa, sb = eng.win_prob(xa, xb, pick_a, pick_b, xv, toss_w, xd)
                    st.session_state["xi_pred"] = {
                        "a": xa, "b": xb, "pa": xpa, "sa": sa, "sb": sb,
                        "base": win_probability(model, xa, xb, xv, toss_w, xd),
                        "potm": eng.potm_predict(xa, xb, pick_a, pick_b, xpa),
                        "venue": xv,
                    }

            xr = st.session_state.get("xi_pred")
            if xr:
                ra, rb, rpa = xr["a"], xr["b"], xr["pa"]
                rwin = ra if rpa >= 0.5 else rb
                xca, xcb = pair_colors(ra, rb)
                section("Prediction", "🏆", "Playing 11, toss aur venue ke hisaab se")
                vs_banner(ra, rb, rpa, xca, xcb)
                prob_bar(ra, rpa, rb, xca, xcb)
                result_card("Jeetne ke zyada chance (Playing 11 ke hisaab se):", rwin,
                            f"{max(rpa, 1 - rpa) * 100:.1f}% chance (model ke hisaab se)", team_color(rwin))

                st.caption(
                    f"Bina Playing 11 ke (sirf team + toss + venue wale model se) {SHORT.get(ra, ra)} ke jeetne ka chance: "
                    f"{xr['base'] * 100:.1f}%. IPL me pehle se winner batana bahut mushkil hai, isliye "
                    "probability aksar 40–60% ke beech hi aati hai."
                )

                sa, sb = xr["sa"], xr["sb"]
                strength = pd.DataFrame(
                    {
                        "Pehlu": ["Batting taakat (runs/inning, average se zyada)",
                             "Wicket lene ki taakat (wickets/innings, average se zyada)",
                             "Economy edge (average se kam runs per over)"],
                        SHORT.get(ra, ra): [round(sa[0], 1), round(sa[1], 2), round(sa[2], 1)],
                        SHORT.get(rb, rb): [round(sb[0], 1), round(sb[1], 2), round(sb[2], 1)],
                    }
                )
                st.dataframe(strength, hide_index=True)

                section("Player of the Match kaun ban sakta hai?", "⭐", "Playing 11 ke players ki rating se")
                pt = xr["potm"]
                podium(pt.head(3), ra, xca, xcb)
                top8 = pt.head(8).iloc[::-1]
                pbar = go.Figure(
                    go.Bar(
                        x=top8["POTM %"], y=top8["Player"], orientation="h",
                        marker_color=[xca if t == ra else xcb for t in top8["Team"]],
                        text=[f"{v:.1f}%" for v in top8["POTM %"]], textposition="outside",
                    )
                )
                pbar.update_layout(xaxis_title="Player of the Match banne ka chance (%)")
                st.plotly_chart(style_fig(pbar, 400), key="xi_potm_bar")
                with st.expander("Saare 22 players ki rating aur POTM %"):
                    st.dataframe(pt, hide_index=True)
                st.caption(
                    "POTM ka andaza player ke career batting/bowling record aur uski team ke jeetne ke chance se aata hai "
                    "(Player of the Match zyadatar jeetne wali team se hota hai). Match ke din ka asli performance "
                    "pehle se pata nahi hota, isliye ye sirf sabse likely naam batata hai."
                )

# ------------------------------------------------------------------ TAB: LIVE MATCH
if page == PAGES[2]:
    section("Half-match / Live prediction", "⏱️")
    st.caption("Match ke beech ka score daalo, model batayega ki ab kaun jeet raha hai.")
    eng = need_engine()
    if eng is not None:
        inn = st.radio("Kaun si innings chal rahi hai?", ["2nd innings (chase)", "1st innings"],
                       horizontal=True, key="live_inn")

        def live_bar(label_a, pa, label_b, key):
            lca, lcb = pair_colors(label_a, label_b)
            prob_bar(label_a, pa, label_b, lca, lcb)

        if inn.startswith("2nd"):
            lc1, lc2 = st.columns(2)
            chase = lc1.selectbox("Chase karne wali team", team_choices, index=0, key="lv_chase")
            defend = lc2.selectbox("Pehle batting karne wali team", team_choices, index=1, key="lv_defend")
            ln = st.columns(5)
            target = ln[0].number_input("Target", min_value=1, max_value=300, value=180, key="lv2_target")
            score = ln[1].number_input("Abhi ka score", min_value=0, max_value=300, value=90, key="lv2_score")
            wk = ln[2].number_input("Wickets gire", min_value=0, max_value=10, value=3, key="lv2_wk")
            ov = ln[3].number_input("Overs poore", min_value=0, max_value=19, value=10, key="lv2_ov")
            bl = ln[4].number_input("Is over ki balls", min_value=0, max_value=5, value=0, key="lv2_bl")
            balls = int(ov) * 6 + int(bl)

            if chase == defend:
                st.warning("Dono teams alag-alag chuno.")
            elif balls == 0:
                st.info("Kam se kam 1 ball khel chuki ho tab prediction milti hai.")
            elif score >= target:
                st.success(f"{chase} target pura kar chuki hai. {chase} jeet gayi!")
            elif wk >= 10:
                st.error(f"{chase} all out. {defend} jeet gayi.")
            else:
                pc = eng.live_chase(int(target), int(score), int(wk), balls)
                lwin = chase if pc >= 0.5 else defend
                runs_left, balls_left = int(target - score), 120 - balls
                scoreboard(chase, team_color(chase), int(score), int(wk), balls, [
                    ("Target", int(target)), ("Runs chahiye", runs_left),
                    ("Balls bachi", balls_left), ("Wickets haath me", 10 - int(wk)),
                    ("Current RR", f"{score * 6 / balls:.2f}"), ("Required RR", f"{runs_left * 6 / balls_left:.2f}"),
                ])
                result_card("Abhi jeetne ke zyada chance:", lwin,
                            f"{max(pc, 1 - pc) * 100:.1f}% chance (model ke hisaab se)", team_color(lwin))
                live_bar(chase, pc, defend, "lv2_bar")
                if balls + 6 <= 120:
                    runs_axis = list(range(0, 25))
                    fig_sw = go.Figure()
                    for w_lost, name, col in [(0, "Agle over me wicket nahi", COLOR_B), (1, "1 wicket gira", "#f43f5e")]:
                        ys = [eng.live_chase(int(target), int(score) + r, int(wk) + w_lost, balls + 6) * 100
                              for r in runs_axis]
                        fig_sw.add_trace(go.Scatter(x=runs_axis, y=ys, mode="lines+markers", name=name,
                                                    line=dict(color=col)))
                    fig_sw.update_layout(xaxis_title="Agle over me banaye gaye runs",
                                         yaxis_title=f"{chase} ke jeetne ka chance (%)",
                                         yaxis=dict(range=[0, 100]), legend=dict(orientation="h", y=-0.35))
                    st.markdown("**Agla over match badal sakta hai?**")
                    st.plotly_chart(style_fig(fig_sw, 340), key="lv2_whatif")
        else:
            lc1, lc2 = st.columns(2)
            bat1 = lc1.selectbox("Abhi batting kar rahi team", team_choices, index=0, key="lv_bat1")
            bowl1 = lc2.selectbox("Bowling team", team_choices, index=1, key="lv_bowl1")
            lv = st.selectbox("Venue (stadium)", venues, index=0, key="lv1_venue")
            ln = st.columns(4)
            score1 = ln[0].number_input("Abhi ka score", min_value=0, max_value=300, value=80, key="lv1_score")
            wk1 = ln[1].number_input("Wickets gire", min_value=0, max_value=10, value=2, key="lv1_wk")
            ov1 = ln[2].number_input("Overs poore", min_value=0, max_value=19, value=10, key="lv1_ov")
            bl1 = ln[3].number_input("Is over ki balls", min_value=0, max_value=5, value=0, key="lv1_bl")
            balls1 = int(ov1) * 6 + int(bl1)

            if bat1 == bowl1:
                st.warning("Dono teams alag-alag chuno.")
            elif balls1 == 0:
                st.info("Kam se kam 1 ball khel chuki ho tab prediction milti hai.")
            elif wk1 >= 10:
                st.error(f"{bat1} all out ho chuki hai.")
            else:
                proj, mae, p1 = eng.live_first(lv, int(score1), int(wk1), balls1)
                _, v_avg1, _ = eng.venue_info(lv)
                scoreboard(bat1, team_color(bat1), int(score1), int(wk1), balls1, [
                    ("Projected score", f"{proj:.0f}"), ("Andaza range", f"{max(proj - mae, score1):.0f}–{proj + mae:.0f}"),
                    ("Venue average", f"{v_avg1:.0f}"), ("Current RR", f"{score1 * 6 / balls1:.2f}"),
                ])
                live_bar(bat1, p1, bowl1, "lv1_bar")
                st.caption(f"Upar ka bar batata hai ki abhi ke score par {bat1} ke match jeetne ka chance kitna hai.")

        st.caption(
            "Ye model IPL ki pichli 2008–2024 ki ball-by-ball history (20 over wale matches) se seekha hai. "
            "2nd innings ka prediction 1st innings se zyada bharosemand hota hai, kyunki target pata hota hai."
        )

# ------------------------------------------------------------------ TAB 2: ANALYSIS
if page == PAGES[3]:
    section("Teams ka record", "📊")
    toss_win_pct = (df["toss_winner"] == df["winner"]).mean() * 100
    k1, k2, k3 = st.columns(3)
    k1.metric("Total matches", len(df))
    k2.metric("Toss jeetne wali team jeeti", f"{toss_win_pct:.1f}%")
    k3.metric("Alag-alag stadium", df["venue"].nunique())

    shown = tt[tt["played"] >= 10].sort_values("win_pct")
    fig_w = go.Figure(
        go.Bar(
            x=shown["win_pct"], y=shown["team"], orientation="h",
            marker=dict(color=shown["win_pct"], colorscale="YlOrRd"),
            text=[f"{v}%" for v in shown["win_pct"]], textposition="outside",
        )
    )
    fig_w.update_layout(xaxis_title="Jeet ka percentage (%)")
    st.plotly_chart(style_fig(fig_w, 460), key="win_pct_chart")

    section("Head-to-Head", "🤝")
    h1, h2 = st.columns(2)
    ha = h1.selectbox("Team 1", team_choices, index=0, key="h2h_a")
    hb = h2.selectbox("Team 2", team_choices, index=1, key="h2h_b")
    if ha == hb:
        st.info("Dono alag teams chuno.")
    else:
        n, wa, wb = head_to_head(df, ha, hb)
        if n == 0:
            st.info("In dono teams ke beech dataset me koi match nahi mila.")
        else:
            donut = go.Figure(
                go.Pie(labels=[ha, hb], values=[wa, wb], hole=0.55,
                       marker=dict(colors=list(pair_colors(ha, hb))), textinfo="label+value")
            )
            donut.update_layout(showlegend=False)
            st.plotly_chart(style_fig(donut, 340), key="h2h_donut")
            st.caption(f"Kul {n} matches: {ha} {wa} jeeti, {hb} {wb} jeeti.")

    section("Sabse zyada matches wale stadium", "🏟️")
    top_v = df["venue"].value_counts().head(10).sort_values()
    fig_v = go.Figure(go.Bar(x=top_v.values, y=top_v.index, orientation="h",
                             marker=dict(color="#fb923c")))
    st.plotly_chart(style_fig(fig_v, 380), key="venue_chart")

# ------------------------------------------------------------------ TAB 3: 3D
if page == PAGES[4]:
    section("3D Cricket Ball", "🌐")
    st.caption("🖱️ Mouse se ghumao aur zoom karo")
    st.plotly_chart(cricket_ball_3d(), key="ball3d")

    section("Teams ka 3D view", "🧊")
    st.caption("Har bubble ek team hai: matches khele, kitne jeete aur jeet ka %. Mouse se ghumao.")
    st.plotly_chart(teams_3d(tt), key="teams3d")

# ------------------------------------------------------------------ TAB 4: ABOUT
if page == PAGES[5]:
    section("About", "ℹ️")
    st.markdown(
        """
- **Model:** Logistic Regression (IPL `matches.csv` par train hua)
- **Input:** Team A, Team B, venue, toss kisne jeeta aur toss ka faisla
- **Output:** Dono teams ke jeetne ka percentage
- **Dataset:** Kaggle ka IPL matches dataset (`matches.csv`)
- **Photo:** Wikimedia Commons se. Credit dena zaroori hai.
- **Note:** Ye project sirf seekhne/demo ke liye hai. Isse betting ya paisa lagane ke liye use mat karo.
"""
    )

    st.markdown("#### Naye features")
    st.markdown(
        """
- **Playing 11 + conditions:** `deliveries.csv` se har player ki career batting/bowling rating banti hai (kam data wale players ki rating average ki taraf khichti hai). Dono Playing 11 ki taakat, toss aur venue ka record Logistic Regression ko diya jaata hai.
- **Player of the Match:** Har player ki rating aur uski team ke jeetne ke chance se probability nikalti hai, aur 22 players me normalize hoti hai.
- **Live Match:** `deliveries.csv` ki har ball ki state (score, wickets, balls, run rate, target) par Gradient Boosting model. 2nd innings me jeetne ka chance, 1st innings me projected score aur jeetne ka chance milta hai.
"""
    )
    eng_about = need_engine(show=False)
    if eng_about is not None:
        em = eng_about.metrics
        st.markdown("#### Model ki asli accuracy (purane matches par train, naye par test)")
        st.markdown(
            f"""
- **Winner (Playing 11 + toss + venue):** {em['win_cv_acc'] * 100:.1f}% (time-series CV). Sirf team+venue+toss wale model ka {em['win_base_acc'] * 100:.1f}%, coin flip 50%. Matlab pehle se winner batana IPL me lagbhag 50-50 jaisa hi hai, Playing 11 se sirf thodi madad milti hai.
- **Player of the Match:** sahi player sabse upar {em['potm_top1'] * 100:.1f}% matches me aur top 3 me {em['potm_top3'] * 100:.1f}% matches me (random andaza: {em['potm_rand1'] * 100:.1f}% aur {em['potm_rand3'] * 100:.1f}%).
- **Live, 2nd innings (chase):** {em['live2_acc'] * 100:.1f}% accuracy (AUC {em['live2_auc']:.2f}), alag rakhe gaye {em['live_test_matches']} naye matches par.
- **Live, 1st innings:** jeetne ka andaza {em['live1_acc'] * 100:.1f}% accuracy (AUC {em['live1_auc']:.2f}); final score ka average error lagbhag {em['live1_mae']:.0f} runs.
"""
        )

st.markdown('<div class="footer">Made with 🏏 by Govind Khatile · sirf seekhne/demo ke liye, betting ke liye nahi</div>', unsafe_allow_html=True)