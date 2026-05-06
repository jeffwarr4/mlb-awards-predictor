# src/pull_fg_current.py
# ---------------------------------------------------------------
# Pulls current-season FanGraphs actuals (type=8 dashboard stats)
# for all players using a team-by-team approach to bypass the
# 30-row API hard cap on the global leaderboard endpoint.
#
# Output (written to data/raw/fg_exports/):
#   fg_bat_{year}.csv   — MLBAMID, WAR, wRC+, AVG, OBP, SLG, OPS, Name, Team
#   fg_pit_{year}.csv   — MLBAMID, WAR, FIP, xFIP, K%, BB%, ERA, Name, Team
#
# These files are consumed by predict_awards.load_fg_exports(), which
# is the highest-priority data source for current-season FG stats.
#
# Run:
#   python src/pull_fg_current.py           # current year
#   python src/pull_fg_current.py --year 2025
#   python src/pull_fg_current.py --force   # re-download even if cached
# ---------------------------------------------------------------

import argparse
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import FG_EXPORT_DIR, CURRENT_YEAR

# FanGraphs assigns internal team IDs 1-30 (all MLB franchises confirmed).
# Iterating all 30 is the only reliable way to get every player when the
# global leaderboard endpoint returns at most 30 rows per request.
_TEAM_IDS = list(range(1, 31))

_SLEEP_BETWEEN_TEAMS = 10.0  # seconds between team requests — runs overnight, no rush
_MAX_RETRIES = 2
_RETRY_SLEEP = 60            # back off 60s on transient errors
_BLOCK_SLEEP  = 300          # back off 5 min if FanGraphs returns 403 (IP block)

_FG_URL = "https://www.fangraphs.com/api/leaders/major-league/data"
_FG_HEADERS = {
    "User-Agent":      ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/124.0.0.0 Safari/537.36"),
    "Referer":         "https://www.fangraphs.com/",
    "Accept":          "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
}

_NAME_HTML_RE = re.compile(r"<[^>]+>")

# Columns to keep in the final CSVs (extras are silently dropped if absent)
_BAT_KEEP = ["MLBAMID", "playerid", "Name", "Team", "WAR", "wRC+",
             "AVG", "OBP", "SLG", "OPS", "PA"]
_PIT_KEEP = ["MLBAMID", "playerid", "Name", "Team", "WAR", "ERA",
             "FIP", "xFIP", "K%", "BB%", "IP"]


def _clean_name(s: str) -> str:
    return _NAME_HTML_RE.sub("", str(s)).strip()


def _fetch_team(session: requests.Session, stats: str, team_id: int, year: int) -> list:
    """Fetch type=8 stats for one team. Returns list of row dicts."""
    params = {
        "pos": "all", "stats": stats, "lg": "all",
        "qual": 1, "type": 8,
        "season": year, "month": 0, "season1": year, "ind": 0,
        "team": team_id, "rost": 0, "age": 0,
        "filter": "", "players": 0, "startdate": "", "enddate": "",
        "page": "1_100",
    }
    for attempt in range(_MAX_RETRIES):
        try:
            r = session.get(_FG_URL, params=params, timeout=30)
            if r.status_code == 403:
                # IP temporarily blocked — back off and try once more
                if attempt < _MAX_RETRIES - 1:
                    print(f"    team {team_id}: 403 received — backing off {_BLOCK_SLEEP}s")
                    time.sleep(_BLOCK_SLEEP)
                    continue
                print(f"    team {team_id}: still 403 after backoff — skipping")
                return []
            r.raise_for_status()
            payload = r.json()
            rows = payload.get("data", []) if isinstance(payload, dict) else payload
            return rows or []
        except Exception as exc:
            if attempt < _MAX_RETRIES - 1:
                print(f"    team {team_id} attempt {attempt+1} failed ({exc}) — retrying")
                time.sleep(_RETRY_SLEEP)
            else:
                print(f"    team {team_id} FAILED: {exc}")
                return []
    return []


def _pull_all_teams(session: requests.Session, stats: str, year: int) -> pd.DataFrame:
    """Pull all 30 teams and return combined, deduplicated DataFrame."""
    all_rows = []
    for team_id in _TEAM_IDS:
        rows = _fetch_team(session, stats, team_id, year)
        all_rows.extend(rows)
        time.sleep(_SLEEP_BETWEEN_TEAMS)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)

    if "Name" in df.columns:
        df["Name"] = df["Name"].apply(_clean_name)

    # FanGraphs uses "xMLBAMID" in the API JSON; rename to match load_fg_exports()
    if "xMLBAMID" in df.columns:
        df = df.rename(columns={"xMLBAMID": "MLBAMID"})

    # Traded players appear once per team; keep the row with the highest WAR
    # (season-total WAR is what predict_awards needs)
    if "playerid" in df.columns and "WAR" in df.columns:
        df["WAR"] = pd.to_numeric(df["WAR"], errors="coerce").fillna(0)
        df = (df.sort_values("WAR", ascending=False)
                .drop_duplicates(subset=["playerid"])
                .reset_index(drop=True))

    return df


def pull_bat(year: int, session: requests.Session) -> pd.DataFrame:
    df = _pull_all_teams(session, "bat", year)
    if df.empty:
        return df
    out = df[[c for c in _BAT_KEEP if c in df.columns]].copy()
    print(f"  batting: {len(out)} players")
    return out


def pull_pit(year: int, session: requests.Session) -> pd.DataFrame:
    df = _pull_all_teams(session, "pit", year)
    if df.empty:
        return df
    out = df[[c for c in _PIT_KEEP if c in df.columns]].copy()
    print(f"  pitching: {len(out)} players")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Pull FanGraphs current-season stats")
    parser.add_argument("--year",  type=int, default=CURRENT_YEAR)
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if output files already exist")
    args = parser.parse_args()

    FG_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    bat_path = FG_EXPORT_DIR / f"fg_bat_{args.year}.csv"
    pit_path = FG_EXPORT_DIR / f"fg_pit_{args.year}.csv"

    if bat_path.exists() and pit_path.exists() and not args.force:
        print(f"Already have {bat_path.name} and {pit_path.name}.")
        print("Pass --force to re-download.")
        return

    session = requests.Session()
    session.headers.update(_FG_HEADERS)

    # Preflight: confirm endpoint is reachable before iterating all 30 teams
    try:
        probe = session.get(_FG_URL, params={
            "pos": "all", "stats": "bat", "lg": "all", "qual": 1, "type": 8,
            "season": args.year, "month": 0, "season1": args.year, "ind": 0,
            "team": 1, "rost": 0, "age": 0,
            "filter": "", "players": 0, "startdate": "", "enddate": "",
            "page": "1_30",
        }, timeout=15)
        if probe.status_code == 403:
            print("ERROR: FanGraphs returned 403 on preflight check.")
            print("IP may be temporarily blocked by Cloudflare. Wait 30-60 minutes and retry.")
            print("This happens when the script is run repeatedly in quick succession.")
            return
        probe.raise_for_status()
        print(f"Preflight OK (status {probe.status_code})")
    except Exception as exc:
        print(f"ERROR: Preflight failed — {exc}")
        return

    mins = len(_TEAM_IDS) * 2 * _SLEEP_BETWEEN_TEAMS / 60
    print(f"Pulling FanGraphs actuals for {args.year} "
          f"(30 teams x 2 stat types, ~{mins:.0f} min at {_SLEEP_BETWEEN_TEAMS:.0f}s/team)...")

    print("\nBatting (type=8):")
    bat_df = pull_bat(args.year, session)
    if not bat_df.empty:
        bat_df.to_csv(bat_path, index=False)
        print(f"  saved -> {bat_path}")
    else:
        print("  no batting data returned")

    print("\nPitching (type=8):")
    pit_df = pull_pit(args.year, session)
    if not pit_df.empty:
        pit_df.to_csv(pit_path, index=False)
        print(f"  saved -> {pit_path}")
    else:
        print("  no pitching data returned")

    if not bat_df.empty and not pit_df.empty:
        print(f"\nDone. Run predict_awards.py to score {args.year}.")


if __name__ == "__main__":
    main()
