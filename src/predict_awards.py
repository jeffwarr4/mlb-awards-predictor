# src/predict_awards.py
# ---------------------------------------------------------------
# Scores current-season players and outputs per league per award:
#
#   predictions/
#     top10_al_mvp_latest.csv      ← upload this tab to Google Sheet
#     top10_nl_mvp_latest.csv
#     top10_al_cy_latest.csv
#     top10_nl_cy_latest.csv
#     top10_al_mvp_{ts}.csv        ← timestamped archive
#     ...
#     top5_flat_mvp.csv            ← Canva Bulk Create (2 rows: AL, NL)
#     top5_flat_cy.csv
#     .prev_top10_mvp.csv          ← hidden snapshot for next-run movement
#     .prev_top10_cy.csv
#
# Key output columns:
#   rank        — 1–10
#   rank_delta  — numeric (+3 rose, -2 fell, "" = NEW)  ← JS reads this directly
#   movement    — display string "↑3", "↓2", "–", "NEW" ← human-readable backup
#   Name, Team, lgID, <prob>, <stat cols>, WinPct
# ---------------------------------------------------------------

import warnings; warnings.filterwarnings("ignore")
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
_stinger = Path(r"C:\Users\jeffw\OneDrive\DevProj\stinger-assets")
if _stinger.exists():
    sys.path.insert(0, str(_stinger))
import json, re, urllib.request
import requests
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

# ── Config ────────────────────────────────────────────────────────
CURRENT_YEAR = 2026

TEAM_LEAGUE: Dict[str, str] = {
    "NYY":"AL","BOS":"AL","TOR":"AL","BAL":"AL","TBR":"AL",
    "CLE":"AL","MIN":"AL","DET":"AL","KCR":"AL","CHW":"AL",
    "HOU":"AL","SEA":"AL","OAK":"AL","TEX":"AL","LAA":"AL",
    "LAD":"NL","SFG":"NL","SDP":"NL","ARI":"NL","COL":"NL",
    "CHC":"NL","STL":"NL","CIN":"NL","MIL":"NL","PIT":"NL",
    "ATL":"NL","NYM":"NL","PHI":"NL","WSN":"NL","MIA":"NL",
}
TEAM_ABBR_CANON = {
    "WSH":"WSN","TBD":"TBR","KCA":"KCR","ANA":"LAA","TB":"TBR","KC":"KCR",
    "SF":"SFG","SD":"SDP","WSN":"WSN","LAA":"LAA","LAD":"LAD","SFG":"SFG",
    "SDP":"SDP","NYY":"NYY","BOS":"BOS","TOR":"TOR","BAL":"BAL","TBR":"TBR",
    "CLE":"CLE","MIN":"MIN","DET":"DET","CHW":"CHW","HOU":"HOU","SEA":"SEA",
    "OAK":"OAK","TEX":"TEX","ATL":"ATL","PHI":"PHI","MIA":"MIA","NYM":"NYM",
    "CHC":"CHC","STL":"STL","CIN":"CIN","MIL":"MIL","PIT":"PIT","ARI":"ARI",
    "COL":"COL","KCR":"KCR",
}
MLB_STATS_BASE = "https://statsapi.mlb.com/api/v1"
FG_PROJ_BASE   = "https://www.fangraphs.com/api/projections"

# The FanGraphs actuals leaderboard (/api/leaders/major-league/data) is behind
# Cloudflare's JS challenge — unreachable by any requests-based approach.
# /api/projections is unprotected and serves Steamer (updated weekly in-season).
_FG_SESSION = requests.Session()
_FG_SESSION.headers.update({
    "User-Agent":      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                       " (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Referer":         "https://www.fangraphs.com/",
    "Accept":          "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
})

# Full team names exactly as returned by MLB Stats API → canonical abbreviation.
# Used by get_winpct, get_batting_stats, and get_pitching_stats.
TEAM_NAME_TO_ABBR: Dict[str, str] = {
    # Full names — used by the /stats endpoint
    "New York Yankees":"NYY","Boston Red Sox":"BOS","Toronto Blue Jays":"TOR",
    "Baltimore Orioles":"BAL","Tampa Bay Rays":"TBR",
    "Cleveland Guardians":"CLE","Minnesota Twins":"MIN","Detroit Tigers":"DET",
    "Kansas City Royals":"KCR","Chicago White Sox":"CHW",
    "Houston Astros":"HOU","Seattle Mariners":"SEA","Texas Rangers":"TEX",
    "Los Angeles Angels":"LAA","Oakland Athletics":"OAK","Sacramento Athletics":"OAK",
    "Los Angeles Dodgers":"LAD","San Francisco Giants":"SFG","San Diego Padres":"SDP",
    "Arizona Diamondbacks":"ARI","Colorado Rockies":"COL",
    "Chicago Cubs":"CHC","St. Louis Cardinals":"STL","Cincinnati Reds":"CIN",
    "Milwaukee Brewers":"MIL","Pittsburgh Pirates":"PIT",
    "Atlanta Braves":"ATL","New York Mets":"NYM","Philadelphia Phillies":"PHI",
    "Washington Nationals":"WSN","Miami Marlins":"MIA",
    # Short names — used by the /standings endpoint
    "Yankees":"NYY","Red Sox":"BOS","Blue Jays":"TOR","Orioles":"BAL","Rays":"TBR",
    "Guardians":"CLE","Twins":"MIN","Tigers":"DET","Royals":"KCR","White Sox":"CHW",
    "Astros":"HOU","Mariners":"SEA","Rangers":"TEX","Angels":"LAA","Athletics":"OAK",
    "Dodgers":"LAD","Giants":"SFG","Padres":"SDP","D-backs":"ARI","Diamondbacks":"ARI",
    "Rockies":"COL","Cubs":"CHC","Cardinals":"STL","Reds":"CIN","Brewers":"MIL",
    "Pirates":"PIT","Braves":"ATL","Mets":"NYM","Phillies":"PHI",
    "Nationals":"WSN","Marlins":"MIA",
}

MVP_STAT_COLS  = ["bat_WAR_fg","bat_wRC_plus","bat_OPS","HR","RBI","WinPct"]
CY_STAT_COLS   = ["pit_WAR_fg","pit_FIP","pit_Kpct","SO_pit","IP","WinPct"]
MVP_CANVA_COLS = ["bat_WAR_fg","bat_wRC_plus","bat_OPS","HR","RBI"]
CY_CANVA_COLS  = ["pit_WAR_fg","pit_FIP","pit_Kpct","SO_pit","IP"]


# ── Helpers ───────────────────────────────────────────────────────
def _mlb_fetch(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read())

def get_batting_stats(year: int) -> pd.DataFrame:
    """Fetch season batting stats from the MLB Stats API (never blocked)."""
    url = (f"{MLB_STATS_BASE}/stats"
           f"?stats=season&group=hitting&season={year}&playerPool=All&limit=2000")
    try:
        data = _mlb_fetch(url)
    except Exception as e:
        print(f"  MLB API batting failed: {e}")
        return pd.DataFrame()

    rows = []
    for sp in data["stats"][0]["splits"]:
        s = sp["stat"]
        abbr = TEAM_NAME_TO_ABBR.get(sp.get("team", {}).get("name", ""))
        if not abbr:
            continue
        rows.append({
            "Name":     sp["player"]["fullName"],
            "mlbam_id": sp["player"]["id"],
            "Team": abbr,
            "G":   s.get("gamesPlayed", 0),
            "AB":  s.get("atBats", 0),
            "H":   s.get("hits", 0),
            "HR":  s.get("homeRuns", 0),
            "RBI": s.get("rbi", 0),
            "BB":  s.get("baseOnBalls", 0),
            "SO":  s.get("strikeOuts", 0),
            "SB":  s.get("stolenBases", 0),
            "CS":  s.get("caughtStealing", 0),
            "R":   s.get("runs", 0),
            "2B":  s.get("doubles", 0),
            "3B":  s.get("triples", 0),
            "HBP": s.get("hitByPitch", 0),
            "SF":  s.get("sacFlies", 0),
            "OBP": float(s.get("obp", 0) or 0),
            "SLG": float(s.get("slg", 0) or 0),
            "OPS": float(s.get("ops", 0) or 0),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # wRC+ proxy: OPS scaled to league median — used only as fallback if FanGraphs
    # projections are unavailable; overwritten by get_fg_projections() in build_features.
    qual = df[df["AB"] >= 50]
    lg_ops = qual["OPS"].median() if not qual.empty else 0.72
    df["wRC+"] = (df["OPS"] / lg_ops * 100).round(1) if lg_ops > 0 else 100.0
    df["WAR"] = 0.0
    return df


def get_pitching_stats(year: int) -> pd.DataFrame:
    """Fetch season pitching stats from the MLB Stats API (never blocked).

    FIP, K%, BB%, ERA- are computed from raw counts.
    xFIP approximated as FIP (no HR/FB rate without batted-ball data).
    WAR is set to 0 (not available from this API).
    """
    FIP_CONST = 3.20  # league-average FIP constant; normalizes FIP to ERA scale

    url = (f"{MLB_STATS_BASE}/stats"
           f"?stats=season&group=pitching&season={year}&playerPool=All&limit=2000")
    try:
        data = _mlb_fetch(url)
    except Exception as e:
        print(f"  MLB API pitching failed: {e}")
        return pd.DataFrame()

    rows = []
    for sp in data["stats"][0]["splits"]:
        s = sp["stat"]
        abbr = TEAM_NAME_TO_ABBR.get(sp.get("team", {}).get("name", ""))
        if not abbr:
            continue
        ip  = float(str(s.get("inningsPitched", "0.0") or "0.0"))
        bf  = int(s.get("battersFaced", 0) or 0)
        so  = int(s.get("strikeOuts", 0))
        bb  = int(s.get("baseOnBalls", 0))
        hr  = int(s.get("homeRuns", 0))
        hbp = int(s.get("hitBatsmen", 0))
        er  = int(s.get("earnedRuns", 0))
        fip = (13*hr + 3*(bb+hbp) - 2*so) / ip + FIP_CONST if ip > 0 else np.nan
        rows.append({
            "Name":     sp["player"]["fullName"],
            "mlbam_id": sp["player"]["id"],
            "Team": abbr,
            "IP":  ip,
            "G":   int(s.get("gamesPlayed", 0)),
            "GS":  int(s.get("gamesStarted", 0)),
            "SO":  so,
            "BB":  bb,
            "ER":  er,
            "HR":  hr,
            "SV":  int(s.get("saves", 0)),
            "W":   int(s.get("wins", 0)),
            "L":   int(s.get("losses", 0)),
            "H":   int(s.get("hits", 0)),
            "WAR": 0.0,   # overwritten by get_fg_projections() in build_features
            "FIP": round(fip, 2) if not np.isnan(fip) else np.nan,
            "K%":  round(so / bf, 4) if bf > 0 else 0.0,
            "BB%": round(bb / bf, 4) if bf > 0 else 0.0,
            "xFIP": round(fip, 2) if not np.isnan(fip) else np.nan,
            "_era": er * 9 / ip if ip > 0 else np.nan,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # ERA- = (pitcher ERA / league ERA) * 100; lower is better
    qual_era = df.loc[df["IP"] >= 10, "_era"].dropna()
    lg_era = qual_era.mean() if not qual_era.empty else 4.20
    df["ERA-"] = (df["_era"] / lg_era * 100).where(df["_era"].notna()).round(1)
    df.drop(columns=["_era"], inplace=True)
    return df


def get_fg_projections() -> tuple:
    """Fetch Steamer projections from FanGraphs /api/projections (unprotected endpoint).

    Returns (bat_df, pit_df) each indexed by xMLBAMID (= MLB Stats API player.id).
    Steamer is updated weekly in-season blending actuals + projections, so WAR and
    wRC+ here are more informative than early-season accumulated totals.
    Falls back to empty DataFrames silently if the endpoint is unreachable.
    """
    from config import fg_projection_type
    proj_type = fg_projection_type()
    print(f"  FG projection type: {proj_type}")

    bat_df = pd.DataFrame()
    pit_df = pd.DataFrame()
    try:
        r = _FG_SESSION.get(
            f"{FG_PROJ_BASE}?pos=all&stats=bat&type={proj_type}&team=0&lg=all&players=0",
            timeout=15)
        r.raise_for_status()
        rows = r.json()
        bat_df = pd.DataFrame([
            {"mlbam_id": int(row["xMLBAMID"]),
             "fg_WAR_bat": float(row.get("WAR") or 0),
             "fg_wRC_plus": float(row.get("wRC+") or 0)}
            for row in rows
            if row.get("xMLBAMID") and row.get("League") in ("AL", "NL")
        ])
        print(f"  FG projections (bat): {len(bat_df)} rows")
    except Exception as e:
        print(f"  FG bat projections unavailable: {e}")

    try:
        r = _FG_SESSION.get(
            f"{FG_PROJ_BASE}?pos=all&stats=pit&type={proj_type}&team=0&lg=all&players=0",
            timeout=15)
        r.raise_for_status()
        rows = r.json()
        pit_df = pd.DataFrame([
            {"mlbam_id": int(row["xMLBAMID"]),
             "fg_WAR_pit": float(row.get("WAR") or 0),
             "fg_FIP":     float(row.get("FIP") or 0),
             "fg_Kpct":    float(row.get("K%") or 0),
             "fg_BBpct":   float(row.get("BB%") or 0)}
            for row in rows
            if row.get("xMLBAMID") and row.get("League") in ("AL", "NL")
        ])
        print(f"  FG projections (pit): {len(pit_df)} rows")
    except Exception as e:
        print(f"  FG pit projections unavailable: {e}")

    return bat_df, pit_df


def ip_to_outs(ip):
    if pd.isna(ip): return 0
    w = int(np.floor(ip)); return w * 3 + int(round((ip - w) * 10))

def pad(df, cols):
    for c in cols:
        if c not in df.columns: df[c] = 0


# ── Win % — MLB Stats API ─────────────────────────────────────────
def get_winpct(year: int) -> pd.DataFrame:
    try:
        url = (f"{MLB_STATS_BASE}/standings"
               f"?leagueId=103,104&season={year}&standingsTypes=regularSeason")
        data = _mlb_fetch(url)
        rows = []
        for rec in data.get("records", []):
            for tr in rec.get("teamRecords", []):
                abbr = TEAM_NAME_TO_ABBR.get(tr["team"].get("name", "").strip())
                if not abbr:
                    continue
                w, l = tr.get("wins", 0), tr.get("losses", 0)
                rows.append({"Team": abbr, "WinPct": w / (w + l) if w + l else 0.5})
        df = pd.DataFrame(rows)
        if not df.empty:
            return df
    except Exception as e:
        print(f"  WinPct fetch failed: {e}")

    print("  WinPct fallback 0.500")
    return pd.DataFrame({"Team": list(TEAM_LEAGUE), "WinPct": [0.5] * len(TEAM_LEAGUE)})


# ── Feature builder ───────────────────────────────────────────────
def build_features(year: int) -> pd.DataFrame:
    print(f"Fetching {year} stats from MLB Stats API ...")
    bat = get_batting_stats(year)
    pit = get_pitching_stats(year)
    print(f"  batters: {len(bat)}  pitchers: {len(pit)}")
    fg_bat, fg_pit = get_fg_projections()

    if bat.empty:
        bat = pd.DataFrame(columns=[
            "Name","Team","G","AB","H","HR","RBI","BB","SO","SB","CS",
            "R","2B","3B","HBP","SF","WAR","wRC+","OPS","OBP","SLG"
        ])
    if pit.empty:
        pit = pd.DataFrame(columns=[
            "Name","Team","IP","G","GS","SO","BB","ER","HR","SV",
            "W","L","H","WAR","FIP","K%","BB%","ERA-","xFIP"
        ])
    for df in (bat, pit):
        if "Team" in df.columns:
            # Teams from MLB API are already canonical; TEAM_ABBR_CANON handles
            # any legacy abbreviations that may appear in archived data
            df["Team"] = df["Team"].map(
                lambda x: TEAM_ABBR_CANON.get(str(x).strip(), str(x).strip()))
    if not bat.empty: bat["lgID"] = bat["Team"].map(TEAM_LEAGUE)
    if not pit.empty: pit["lgID"] = pit["Team"].map(TEAM_LEAGUE)

    bm = {"G":"G_bat","AB":"AB","H":"H","HR":"HR","RBI":"RBI","BB":"BB","SO":"SO",
          "SB":"SB","CS":"CS","R":"R","2B":"2B","3B":"3B","HBP":"HBP","SF":"SF"}
    for s in bm:
        if s not in bat.columns: bat[s] = 0
    bat.rename(columns=bm, inplace=True)
    # bat_* columns are FanGraphs-named display aliases; OBP/SLG/OPS keep their
    # original names because that is what the trained model feature columns expect.
    for s, d in [("WAR","bat_WAR_fg"),("wRC+","bat_wRC_plus"),
                 ("OPS","bat_OPS"),("OBP","bat_OBP"),("SLG","bat_SLG")]:
        if s in bat.columns:
            bat[d] = pd.to_numeric(bat[s], errors="coerce")
    # Ensure Lahman-named model features are numeric and present
    for col in ("OBP", "SLG", "OPS"):
        if col in bat.columns:
            bat[col] = pd.to_numeric(bat[col], errors="coerce")
        else:
            bat[col] = 0.0

    pad(pit, ["IP"])
    pit["IP"] = pd.to_numeric(pit["IP"], errors="coerce").fillna(0)
    pit["IPouts"] = pit["IP"].apply(ip_to_outs)
    pm = {"G":"G_pit","GS":"GS","SO":"SO_pit","BB":"BB_pit","ER":"ER",
          "HR":"HR_pit","SV":"SV","W":"W_pit","L":"L_pit","H":"H_pit"}
    for s in pm:
        if s not in pit.columns: pit[s] = 0
    pit.rename(columns=pm, inplace=True)
    for s,d in [("WAR","pit_WAR_fg"),("FIP","pit_FIP"),("K%","pit_Kpct"),
                ("BB%","pit_BBpct"),("ERA-","pit_ERA_minus"),("xFIP","pit_xFIP")]:
        if s in pit.columns: pit[d] = pd.to_numeric(pit[s], errors="coerce")

    bk = [c for c in ["Name","mlbam_id","Team","lgID","G_bat","AB","H","HR","RBI","BB","SO",
          "SB","CS","R","2B","3B","HBP","SF",
          "OBP","SLG","OPS",              # Lahman names — used directly by model
          "bat_WAR_fg","bat_wRC_plus","bat_OPS","bat_OBP","bat_SLG"]
          if c in bat.columns]
    pk = [c for c in ["Name","mlbam_id","Team","lgID","IP","IPouts","G_pit","GS","SO_pit",
          "BB_pit","ER","HR_pit","SV","W_pit","L_pit","H_pit","pit_WAR_fg",
          "pit_FIP","pit_Kpct","pit_BBpct","pit_ERA_minus","pit_xFIP"]
          if c in pit.columns]

    cur = pd.merge(bat[bk], pit[pk], on=["Name","Team"], how="outer", suffixes=("_bat","_pit"))

    # Reconcile lgID from both sides after the outer merge
    if "lgID_bat" in cur.columns and "lgID_pit" in cur.columns:
        cur["lgID"] = cur["lgID_bat"].fillna(cur["lgID_pit"])
        cur.drop(columns=["lgID_bat","lgID_pit"], inplace=True)
    elif "lgID_bat" in cur.columns:
        cur.rename(columns={"lgID_bat":"lgID"}, inplace=True)
    elif "lgID_pit" in cur.columns:
        cur.rename(columns={"lgID_pit":"lgID"}, inplace=True)

    # Reconcile mlbam_id (same player may appear in both bat and pit after outer merge)
    if "mlbam_id_bat" in cur.columns and "mlbam_id_pit" in cur.columns:
        cur["mlbam_id"] = cur["mlbam_id_bat"].fillna(cur["mlbam_id_pit"])
        cur.drop(columns=["mlbam_id_bat","mlbam_id_pit"], inplace=True)
    elif "mlbam_id_bat" in cur.columns:
        cur.rename(columns={"mlbam_id_bat":"mlbam_id"}, inplace=True)
    elif "mlbam_id_pit" in cur.columns:
        cur.rename(columns={"mlbam_id_pit":"mlbam_id"}, inplace=True)

    # If lgID still missing (both bat and pit were empty), derive from Team
    if "lgID" not in cur.columns:
        cur["lgID"] = cur["Team"].map(TEAM_LEAGUE)

    # Drop rows where league is unknown (e.g. team not in TEAM_LEAGUE dict)
    cur = cur[cur["lgID"].isin(["AL","NL"])].copy()

    # ── Overlay FanGraphs Steamer projections (WAR, wRC+, FIP, K%, BB%) ──────
    # Joined on integer MLBAM ID — no name-matching fragility.
    # Only overwrites where a projection exists; MLB Stats API values are kept
    # as fallback for players missing from Steamer (e.g. call-ups mid-season).
    if not fg_bat.empty and "mlbam_id" in cur.columns:
        cur["mlbam_id"] = pd.to_numeric(cur["mlbam_id"], errors="coerce")
        fg_bat["mlbam_id"] = pd.to_numeric(fg_bat["mlbam_id"], errors="coerce")
        cur = cur.merge(fg_bat, on="mlbam_id", how="left")
        cur["bat_WAR_fg"]   = cur["fg_WAR_bat"].where(cur["fg_WAR_bat"].notna(), cur["bat_WAR_fg"])
        cur["bat_wRC_plus"] = cur["fg_wRC_plus"].where(cur["fg_wRC_plus"].notna(), cur["bat_wRC_plus"])
        cur.drop(columns=["fg_WAR_bat","fg_wRC_plus"], inplace=True)

    if not fg_pit.empty and "mlbam_id" in cur.columns:
        fg_pit["mlbam_id"] = pd.to_numeric(fg_pit["mlbam_id"], errors="coerce")
        cur = cur.merge(fg_pit, on="mlbam_id", how="left")
        cur["pit_WAR_fg"] = cur["fg_WAR_pit"].where(cur["fg_WAR_pit"].notna(), cur["pit_WAR_fg"])
        cur["pit_FIP"]    = cur["fg_FIP"].where(cur["fg_FIP"].notna(), cur["pit_FIP"])
        cur["pit_Kpct"]   = cur["fg_Kpct"].where(cur["fg_Kpct"].notna(), cur["pit_Kpct"])
        cur["pit_BBpct"]  = cur["fg_BBpct"].where(cur["fg_BBpct"].notna(), cur["pit_BBpct"])
        cur.drop(columns=["fg_WAR_pit","fg_FIP","fg_Kpct","fg_BBpct"], inplace=True)

    cur = cur.merge(get_winpct(year), on="Team", how="left")
    cur["WinPct"] = pd.to_numeric(cur.get("WinPct"), errors="coerce").fillna(0.5)
    pad(cur, ["G_fld","PO","A","E","DP","FieldPct"])
    nc = cur.select_dtypes(include=[np.number]).columns
    cur[nc] = cur[nc].replace([np.inf,-np.inf], np.nan).fillna(0)
    return cur
    

# ── Movement ──────────────────────────────────────────────────────
def add_movement(df: pd.DataFrame, prev_path: Path) -> pd.DataFrame:
    """
    rank_delta : int if returning player (+up/-down), "" if NEW
    movement   : display string "↑3", "↓2", "–", "NEW"
    Mirrors the NBA sheet's rank_delta column so the same JS logic works.
    """
    df = df.copy()
    lookup = {}
    if prev_path.exists():
        prev = pd.read_csv(prev_path)
        lookup = {(str(r["Name"]).strip(), str(r["lgID"]).strip()): int(r["rank"])
                  for _, r in prev.iterrows() if pd.notna(r.get("rank"))}

    deltas, displays = [], []
    for _, row in df.iterrows():
        key = (str(row["Name"]).strip(), str(row["lgID"]).strip())
        if key not in lookup:
            deltas.append("");  displays.append("NEW")
        else:
            d = lookup[key] - int(row["rank"])
            deltas.append(d)
            displays.append("↑"+str(d) if d>0 else ("↓"+str(abs(d)) if d<0 else "–"))

    df["rank_delta"] = deltas
    df["movement"]   = displays
    return df


# ── Score + rank ──────────────────────────────────────────────────
def load_model(task_name: str, models_dir: Path) -> Tuple:
    from config import CHAMPION_MODEL
    model_file = CHAMPION_MODEL.get(task_name, "model_logreg.joblib")
    model_path = models_dir / task_name / model_file
    # fallback to whichever exists
    if not model_path.exists():
        for f in ["model_logreg.joblib", "model_randomforest.joblib"]:
            if (models_dir / task_name / f).exists():
                model_path = models_dir / task_name / f
                break
    model     = joblib.load(model_path)
    feat_cols = joblib.load(models_dir / task_name / "feature_columns.joblib")
    return model, feat_cols

def score(df, model, feats, col):
    if df.empty:
        out = df.copy(); out[col] = pd.Series(dtype=float); return out
    X = df.copy()
    for c in feats:
        if c not in X.columns: X[c] = 0
    X = X[feats].replace([np.inf,-np.inf], np.nan).fillna(0)
    out = df.copy(); out[col] = model.predict_proba(X)[:,1]; return out


def _headshot_key(name: str) -> str:
    """Filename slug for stinger-assets: firstname_lastname (no team, no accents).
    Must match safe_filename() in sync_espn_headshots.py.
    e.g. 'José Ramírez' → 'jose_ramirez',  'Jazz Chisholm Jr.' → 'jazz_chisholm_jr'
    """
    import unicodedata as _ud
    n = _ud.normalize("NFD", name).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", n.lower())).strip("_")


def _player_key(name: str, team: str) -> str:
    """Stable slug for Sheets lookups: firstname_lastname_team (all lowercase).
    e.g. 'Aaron Judge', 'NYY'        → 'aaron_judge_nyy'
         'Jazz Chisholm Jr.', 'NYY'  → 'jazz_chisholm_jr_nyy'
         'José Ramírez', 'CLE'       → 'jose_ramirez_cle'
    """
    import unicodedata
    # Decompose accented chars (é → e + combining accent) then drop non-ASCII
    normalized = unicodedata.normalize("NFD", name)
    ascii_name = normalized.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-z0-9]+", "_", ascii_name.lower()).strip("_")
    return f"{slug}_{team.lower()}"


def top10(df, prob, stats):
    base_cols = ["mlbam_id", "Name", "Team", "lgID"]
    keep = [c for c in base_cols + [prob] + stats if c in df.columns]
    r = df[keep].dropna(subset=["lgID"]).copy()
    r = r[r["lgID"].isin(["AL", "NL"])]
    frames = []
    for lg in ["AL", "NL"]:
        sub = (r[r["lgID"] == lg]
               .sort_values(prob, ascending=False)
               .head(10)
               .copy())
        sub["rank"] = range(1, len(sub) + 1)
        sub.insert(0, "player_key", sub.apply(
            lambda r: _player_key(r["Name"], r["Team"]), axis=1))
        frames.append(sub)
    return pd.concat(frames, ignore_index=True)


# ── Canva flat CSV ─────────────────────────────────────────────────
def build_flat(df, stat_cols, n=5):
    rows = []
    for lg, grp in df.groupby("lgID"):
        grp = grp.sort_values("rank").head(n)
        row = {"League": lg}
        for i, (_, p) in enumerate(grp.iterrows(), 1):
            row[f"P{i}_Rank"] = int(p["rank"])
            row[f"P{i}_Name"] = p["Name"]
            row[f"P{i}_Team"] = p["Team"]
            for col in stat_cols:
                key = col.replace("%","pct").replace("+","plus").replace("-","minus")
                v = p.get(col,"")
                if isinstance(v, float):
                    if col in ("bat_OPS","bat_OBP","bat_SLG"): v = f"{v:.3f}"
                    elif "Kpct" in col or "BBpct" in col:
                        v = f"{v*100:.1f}%" if v<1 else f"{v:.1f}%"
                    elif col in ("pit_FIP","pit_xFIP"): v = f"{v:.2f}"
                    else: v = f"{v:.1f}"
                row[f"P{i}_{key}"] = v
        rows.append(row)
    return pd.DataFrame(rows)


# ── Main ──────────────────────────────────────────────────────────
def main(year=CURRENT_YEAR, outdir=None, models_dir=None, timestamp=None):
    outdir     = Path(outdir or f"predictions/{year}")
    models_dir = Path(models_dir or "models")
    outdir.mkdir(parents=True, exist_ok=True)
    ts = timestamp or datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    prev_mvp = outdir / ".prev_top10_mvp.csv"
    prev_cy  = outdir / ".prev_top10_cy.csv"

    cur = build_features(year)
    if cur.empty:
        print(f"No player data returned for {year} — season may not have started yet.")
        return {}
    print("cur columns:", [c for c in cur.columns])
    print("lgID present:", "lgID" in cur.columns)
    print("lgID values:", cur["lgID"].value_counts().to_dict() if "lgID" in cur.columns else "MISSING")
    mm, mf = load_model("MVP_top5",  models_dir)
    cm, cf = load_model("CY_top5",   models_dir)
    # Minimum activity filters — keeps Skubal in MVP, removes Judge from CY
    # Any pitcher with at least 1 inning is CY eligible
    # Any batter with at least 1 PA OR pitcher with MVP-caliber WAR is MVP eligible
    cur_mvp = cur[
    (cur["AB"] > 0) |                    # position players
    (cur["bat_WAR_fg"] >= 1.5)           # Ohtani tier (future-proof)
    ].copy()

    cur_cy = cur[
        (cur["IPouts"] > 0)                  # anyone who has actually pitched
    ].copy()

    t10_mvp = add_movement(top10(score(cur_mvp,mm,mf,"MVP_prob"),"MVP_prob",MVP_STAT_COLS), prev_mvp)
    t10_cy  = add_movement(top10(score(cur_cy, cm,cf,"CY_prob"), "CY_prob", CY_STAT_COLS),  prev_cy)

    try:
        from sync_espn_headshots import add_players_if_new
        candidates = list(dict.fromkeys(t10_mvp["Name"].tolist() + t10_cy["Name"].tolist()))
        new_names = add_players_if_new(candidates)
        if new_names:
            print(f"\nHeadshot list: added {len(new_names)} new player(s): {', '.join(new_names)}")
        else:
            print("\nHeadshot list: no new players to add.")
    except Exception as exc:
        print(f"\n[WARN] Could not update headshot fetch list: {exc}")

    wp = get_winpct(year)
    print("WinPct sample:")
    print(wp.sort_values("WinPct", ascending=False).head(10).to_string(index=False))

    # Save snapshots for next run
    t10_mvp[["Name","lgID","rank"]].to_csv(prev_mvp, index=False)
    t10_cy[["Name","lgID","rank"]].to_csv(prev_cy,   index=False)

    # Per-league CSVs — one file per league per award
    saved = {}
    for award, df, prob in [("mvp",t10_mvp,"MVP_prob"),("cy",t10_cy,"CY_prob")]:
        stats = MVP_STAT_COLS if award=="mvp" else CY_STAT_COLS
        for lg in ["AL","NL"]:
            sub  = df[df["lgID"]==lg].copy()
            base = f"top10_{lg.lower()}_{award}"
            sub.to_csv(outdir/f"{base}_{ts}.csv",  index=False)   # timestamped archive
            sub.to_csv(outdir/f"{base}_latest.csv", index=False)  # Google Sheet import
            saved[f"{lg}_{award}"] = str(outdir/f"{base}_latest.csv")
            # Keep only the 4 most recent archives; delete older ones
            archives = sorted(outdir.glob(f"{base}_2*.csv"))  # timestamp prefix is a year
            for old in archives[:-4]:
                old.unlink()
            show = ["rank","rank_delta","movement","Name","Team",prob]+stats
            print(f"\n=== {lg} {'MVP' if award=='mvp' else 'Cy Young'} ===")
            print(sub[[c for c in show if c in sub.columns]].to_string(index=False))

    # Canva flat CSVs
    build_flat(t10_mvp, MVP_CANVA_COLS).to_csv(outdir/"top5_flat_mvp.csv", index=False)
    build_flat(t10_cy,  CY_CANVA_COLS).to_csv(outdir/"top5_flat_cy.csv",  index=False)
    saved["flat_mvp"] = str(outdir/"top5_flat_mvp.csv")
    saved["flat_cy"]  = str(outdir/"top5_flat_cy.csv")

    # Headshot candidates — stable path fetched weekly by stinger-assets workflow
    candidates_path = Path("predictions/mlb_candidates.csv")
    seen: set[str] = set()
    rows = []
    for name in dict.fromkeys(t10_mvp["Name"].tolist() + t10_cy["Name"].tolist()):
        key = _headshot_key(name)
        if key not in seen:
            rows.append({"sport": "MLB", "player_name": name, "player_key": key})
            seen.add(key)
    pd.DataFrame(rows).to_csv(candidates_path, index=False)
    saved["mlb_candidates"] = str(candidates_path)
    print(f"  Headshot candidates → {candidates_path} ({len(rows)} players)")

    print(f"\n✅  All outputs → {outdir.resolve()}")
    return saved


if __name__ == "__main__":
    main()
