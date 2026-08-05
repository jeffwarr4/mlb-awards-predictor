# src/pull_fg_current.py
# ---------------------------------------------------------------
# Pulls current-season FanGraphs actuals (type=8 dashboard stats)
# for ALL players in a single request per stat type.
#
# Output (written to data/raw/fg_exports/):
#   fg_bat_{year}.csv   — same column set as the manual "Export Data" CSV
#   fg_pit_{year}.csv   — same column set as the manual "Export Data" CSV
#
# These files are consumed by predict_awards.load_fg_exports(), which
# is the highest-priority data source for current-season FG stats.
# They are also the committed fallback used when the GitHub Actions
# runner IP is blocked by Cloudflare — so this script must NEVER
# overwrite a good file with a short/empty result. See _validate().
#
# Run:
#   python src/pull_fg_current.py           # current year
#   python src/pull_fg_current.py --year 2025
#   python src/pull_fg_current.py --force   # re-download even if cached
# ---------------------------------------------------------------
#
# HISTORY / WHY THIS LOOKS THE WAY IT DOES
# ----------------------------------------
# The previous version paged with `page="1_100"` and always got back
# exactly 30 rows, which was misread as a 30-row "hard cap" on the
# leaderboard endpoint. It isn't a cap and it isn't a paywall — the
# endpoint simply ignores `page` and takes `pageitems` + `pagenum`.
# With `pageitems` set correctly one request returns the entire
# leaderboard (1352 batters / 780 pitchers for 2026 at qual=0), so the
# old 30-team crawl with a 10s sleep (~10 min) is no longer needed.
# The team-by-team path is retained only as a fallback.

import argparse
import re
import sys
import time
import unicodedata
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import FG_EXPORT_DIR, CURRENT_YEAR

_FG_URL = "https://www.fangraphs.com/api/leaders/major-league/data"
# DO NOT SEND THESE. Kept only so --probe can demonstrate the failure.
#
# This header block was in the original version of this script and is what
# actually caused the "Cloudflare is blocking us" 403s. Sending a hardcoded
# Chrome/124 User-Agent from a Python HTTP client is a far stronger bot
# signal than sending no User-Agent at all: the claimed browser is
# inconsistent with everything else about the connection. Plain `requests`
# with default headers returns 200. Verified by --probe:
#
#   requests            none          200
#   requests            full _FG      403
#   curl_cffi:chrome    referer only  200
#   curl_cffi:chrome    full _FG      403
#
# The endpoint is public — no login, no cookies, no TLS impersonation.
_FG_HEADERS = {
    "User-Agent":      ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/124.0.0.0 Safari/537.36"),
    "Referer":         "https://www.fangraphs.com/leaders/major-league",
    "Accept":          "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
}

_PAGE_ITEMS = 5000           # comfortably above any full-season leaderboard
_TEAM_IDS = list(range(1, 31))
_SLEEP_BETWEEN_TEAMS = 10.0  # only used by the fallback path
_MAX_RETRIES = 2
_RETRY_SLEEP = 15
_BLOCK_SLEEP = 20            # short — a 403 here is systematic, not rate-limiting

# Absolute floors — below this the pull is treated as failed rather than
# written out, so a bad run can't clobber the committed fallback CSVs.
_MIN_ROWS = {"bat": 200, "pit": 150}

_NAME_HTML_RE = re.compile(r"<[^>]+>")

# ── Output schema ────────────────────────────────────────────────
# Column name in the CSV  ->  key in the FanGraphs API JSON.
# Matches the manual "Export Data" download so that a hand-exported
# file and a scripted file are interchangeable downstream.

_BAT_COLS = [
    ("Name", "Name"), ("Team", "Team"), ("G", "G"), ("PA", "PA"),
    ("HR", "HR"), ("R", "R"), ("RBI", "RBI"), ("SB", "SB"),
    ("BB%", "BB%"), ("K%", "K%"), ("ISO", "ISO"), ("BABIP", "BABIP"),
    ("AVG", "AVG"), ("OBP", "OBP"), ("SLG", "SLG"), ("wOBA", "wOBA"),
    ("xwOBA", "xwOBA"), ("wRC+", "wRC+"),
    ("BsR", "BaseRunning"),      # dashboard "BsR" is BaseRunning, NOT wBsR
    ("Off", "Offense"), ("Def", "Defense"),
    ("WAR", "WAR"),
    ("NameASCII", None),         # derived from Name
    ("PlayerId", "playerid"),
    ("MLBAMID", "xMLBAMID"),     # API calls it xMLBAMID; export calls it MLBAMID
]

# NOTE ON COLUMN ORDER (pitching): predict_awards.load_fg_exports() picks the
# strikeout column with  next(c for c in raw.columns if c in ("K%", "K/9")) —
# i.e. whichever appears FIRST in the file wins. Training (merge_fangraphs.py)
# maps FanGraphs "K%" -> pit_Kpct as a true rate (~0.28). The manual dashboard
# export only carries K/9 (~10.5), so live scoring was feeding K/9 into a
# feature trained on K%, ~37x too large, and render_graphics.py then formatted
# it as a percentage. K% and BB% are therefore emitted BEFORE K/9 and BB/9.
_PIT_COLS = [
    ("Name", "Name"), ("Team", "Team"),
    ("W", "W"), ("L", "L"), ("SV", "SV"), ("G", "G"), ("GS", "GS"),
    ("IP", "IP"),
    ("K%", "K%"), ("BB%", "BB%"),     # must precede K/9 and BB/9
    ("K/9", "K/9"), ("BB/9", "BB/9"), ("HR/9", "HR/9"),
    ("BABIP", "BABIP"), ("LOB%", "LOB%"), ("GB%", "GB%"), ("HR/FB", "HR/FB"),
    ("vFA (pi)", "pfxvFA"),           # Pitch Info fastball velocity
    ("ERA", "ERA"), ("ERA-", "ERA-"), ("xERA", "xERA"),
    ("FIP", "FIP"), ("xFIP", "xFIP"),
    ("WAR", "WAR"),
    ("NameASCII", None),
    ("PlayerId", "playerid"),
    ("MLBAMID", "xMLBAMID"),
]

_SCHEMA = {"bat": _BAT_COLS, "pit": _PIT_COLS}


# ── helpers ──────────────────────────────────────────────────────

def _clean_name(s) -> str:
    """FanGraphs returns Name as an <a href=...>Player</a> anchor."""
    return _NAME_HTML_RE.sub("", str(s)).strip()


def _to_ascii(s: str) -> str:
    """'José Ramírez' -> 'Jose Ramirez' (matches the export's NameASCII)."""
    return (unicodedata.normalize("NFKD", str(s))
            .encode("ascii", "ignore").decode("ascii").strip())


def _params(stats: str, year: int, team: int = 0, pageitems: int = _PAGE_ITEMS) -> dict:
    return {
        "pos": "all", "stats": stats, "lg": "all",
        "qual": 0,            # 0 = no minimum PA/IP (was 1)
        "type": 8,            # dashboard
        "season": year, "season1": year,
        "month": 0, "ind": 0,
        "team": team, "rost": 0, "age": 0,
        "filter": "", "players": 0, "startdate": "", "enddate": "",
        "pageitems": pageitems,   # the param that actually works
        "pagenum": 1,             # `page="1_100"` is ignored by the API
    }


class CloudflareBlocked(Exception):
    """Raised when FanGraphs returns 403 — the request never reached the API."""


def _candidate_sessions():
    """Yield (label, session) pairs in preference order.

    Plain `requests` with default headers is first because that is what
    actually works (see --probe). The curl_cffi entries are optional
    insurance in case FanGraphs tightens things later; the script does not
    depend on curl_cffi being installed.
    """
    s = requests.Session()
    yield "requests (default headers)", s

    try:
        from curl_cffi import requests as creq
    except ImportError:
        return

    for target in ("chrome", "safari17_0"):
        try:
            yield f"curl_cffi:{target}", creq.Session(impersonate=target)
        except Exception:
            continue


def make_session(year: int):
    """Return the first session configuration that FanGraphs accepts.

    Costs one tiny 1-row request per candidate. This is self-healing: if
    FanGraphs changes its bot rules, the script tries the alternatives
    instead of failing outright.

    Critically, this sends NO spoofed browser headers. See _FG_HEADERS.
    """
    for label, session in _candidate_sessions():
        try:
            r = session.get(_FG_URL, params=_params("bat", year, pageitems=1),
                            timeout=30)
            if r.status_code == 200:
                print(f"  http: {label}")
                return session
            print(f"  http: {label} -> {r.status_code}, trying next")
        except Exception as exc:
            print(f"  http: {label} -> {type(exc).__name__}, trying next")
        time.sleep(1)

    raise CloudflareBlocked("no working client configuration")


def _get(session, params: dict) -> dict:
    """GET with a short retry. Raises CloudflareBlocked on a 403.

    A 403 here means the TLS fingerprint was rejected — it is not rate
    limiting, so sleeping and retrying the same way accomplishes nothing.
    Fail fast and loudly rather than burning minutes in backoff.
    """
    for attempt in range(_MAX_RETRIES):
        try:
            r = session.get(_FG_URL, params=params, timeout=60)
            if r.status_code == 403:
                if attempt < _MAX_RETRIES - 1:
                    print(f"    403 (Cloudflare) — one quick retry in {_BLOCK_SLEEP}s")
                    time.sleep(_BLOCK_SLEEP)
                    continue
                raise CloudflareBlocked("FanGraphs returned 403")
            r.raise_for_status()
            return r.json()
        except CloudflareBlocked:
            raise
        except Exception as exc:
            if attempt < _MAX_RETRIES - 1:
                print(f"    attempt {attempt + 1} failed ({exc}) — retrying in {_RETRY_SLEEP}s")
                time.sleep(_RETRY_SLEEP)
            else:
                print(f"    FAILED: {exc}")
    return {}


def _rows_of(payload) -> list:
    if isinstance(payload, dict):
        return payload.get("data", []) or []
    return payload or []


def _shape(rows: list, stats: str) -> pd.DataFrame:
    """Project raw API rows onto the manual-export column schema."""
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    if "Name" in df.columns:
        df["Name"] = df["Name"].apply(_clean_name)
    if "Team" in df.columns:
        df["Team"] = df["Team"].apply(_clean_name)

    out = pd.DataFrame()
    for csv_col, api_key in _SCHEMA[stats]:
        if csv_col == "NameASCII":
            out[csv_col] = df["Name"].apply(_to_ascii) if "Name" in df.columns else ""
        elif api_key in df.columns:
            out[csv_col] = df[api_key]
        else:
            print(f"    warning: '{csv_col}' (API key '{api_key}') not in response — blank")
            out[csv_col] = pd.NA

    # Traded players appear once per team on the team-by-team fallback path;
    # keep the highest-WAR row (season totals are what predict_awards needs).
    if "PlayerId" in out.columns and "WAR" in out.columns:
        out["WAR"] = pd.to_numeric(out["WAR"], errors="coerce").fillna(0)
        out = (out.sort_values("WAR", ascending=False)
                  .drop_duplicates(subset=["PlayerId"])
                  .reset_index(drop=True))
    return out


def _validate(df: pd.DataFrame, stats: str, expected: int | None) -> bool:
    """Refuse to write anything that looks truncated or broken."""
    n = len(df)
    floor = _MIN_ROWS[stats]
    if n < floor:
        print(f"  REJECTED: only {n} rows (floor {floor}) — refusing to overwrite "
              f"the committed fallback CSV")
        return False
    if expected and n < expected * 0.9:
        print(f"  REJECTED: got {n} rows but API reported totalCount={expected}")
        return False
    if "MLBAMID" in df.columns:
        have_id = pd.to_numeric(df["MLBAMID"], errors="coerce").notna().sum()
        if have_id < n * 0.5:
            print(f"  REJECTED: only {have_id}/{n} rows have MLBAMID — "
                  f"predict_awards joins on this")
            return False
    if expected and n != expected:
        print(f"  note: {n} rows vs totalCount={expected} (dedup of traded players)")
    return True


# ── pull paths ───────────────────────────────────────────────────

def pull_single(session: requests.Session, stats: str, year: int):
    """Preferred path: one request returns the whole leaderboard."""
    payload = _get(session, _params(stats, year))
    rows = _rows_of(payload)
    total = payload.get("totalCount") if isinstance(payload, dict) else None
    if not rows:
        return pd.DataFrame(), total
    print(f"  {stats}: {len(rows)} rows in a single request"
          + (f" (totalCount={total})" if total else ""))
    return _shape(rows, stats), total


def pull_by_team(session: requests.Session, stats: str, year: int):
    """Fallback: 30 requests, one per team, slow enough to avoid a block."""
    print(f"  {stats}: falling back to team-by-team "
          f"(~{len(_TEAM_IDS) * _SLEEP_BETWEEN_TEAMS / 60:.0f} min)")
    all_rows = []
    for team_id in _TEAM_IDS:
        rows = _rows_of(_get(session, _params(stats, year, team=team_id, pageitems=500)))
        all_rows.extend(rows)
        time.sleep(_SLEEP_BETWEEN_TEAMS)
    print(f"  {stats}: {len(all_rows)} rows from team crawl")
    return _shape(all_rows, stats), None


def pull(session: requests.Session, stats: str, year: int) -> pd.DataFrame:
    df, total = pull_single(session, stats, year)
    if not df.empty and _validate(df, stats, total):
        return df
    df, total = pull_by_team(session, stats, year)
    if not df.empty and _validate(df, stats, total):
        return df
    return pd.DataFrame()


# ── diagnostics ──────────────────────────────────────────────────

def probe(year: int) -> int:
    """Try a matrix of client configurations and report which get a 200.

    Cloudflare's decision depends on the TLS fingerprint AND whether the
    headers are self-consistent with it. Rather than change one variable per
    run, try them all at once and print a table.
    """
    params = _params("bat", year, pageitems=1)

    print("Probing FanGraphs client configurations...\n")
    print(f"  {'client':<22} {'headers':<14} status")
    print("  " + "-" * 46)

    results = []

    # 1. plain requests, with and without the browser-ish header block
    for hdrs, label in ((None, "none"), (_FG_HEADERS, "full _FG")):
        s = requests.Session()
        if hdrs:
            s.headers.update(hdrs)
        try:
            code = s.get(_FG_URL, params=params, timeout=30).status_code
        except Exception as exc:
            code = f"ERR {type(exc).__name__}"
        print(f"  {'requests':<22} {label:<14} {code}")
        results.append(("requests", label, code))

    # 2. curl_cffi across impersonation targets and header strategies
    try:
        from curl_cffi import requests as creq
    except ImportError:
        print("\n  curl_cffi not installed — pip install curl_cffi")
        return 1

    targets = ["chrome", "chrome124", "chrome120", "chrome110",
               "edge101", "safari17_0"]
    strategies = (
        (None, "none"),
        ({"Referer": _FG_HEADERS["Referer"]}, "referer only"),
        (_FG_HEADERS, "full _FG"),
    )

    for target in targets:
        for hdrs, label in strategies:
            try:
                s = creq.Session(impersonate=target)
                if hdrs:
                    s.headers.update(hdrs)
                code = s.get(_FG_URL, params=params, timeout=30).status_code
            except Exception as exc:
                code = f"ERR {type(exc).__name__}"
            print(f"  {'curl_cffi:' + target:<22} {label:<14} {code}")
            results.append((target, label, code))
            time.sleep(1)

    wins = [r for r in results if r[2] == 200]
    print()
    if wins:
        print(f"  {len(wins)} configuration(s) returned 200. Best: "
              f"impersonate='{wins[0][0]}', headers={wins[0][1]}")
        return 0
    print("  No HTTP client configuration succeeded — Cloudflare is running")
    print("  challenge mode. A real browser (Playwright) is required.")
    return 2


# ── main ─────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Pull FanGraphs current-season stats")
    parser.add_argument("--year", type=int, default=CURRENT_YEAR)
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if output files already exist")
    parser.add_argument("--probe", action="store_true",
                        help="Diagnose which client config Cloudflare accepts, then exit")
    args = parser.parse_args()

    if args.probe:
        return probe(args.year)

    FG_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    bat_path = FG_EXPORT_DIR / f"fg_bat_{args.year}.csv"
    pit_path = FG_EXPORT_DIR / f"fg_pit_{args.year}.csv"

    if bat_path.exists() and pit_path.exists() and not args.force:
        print(f"Already have {bat_path.name} and {pit_path.name}. Pass --force to re-download.")
        return 0

    print(f"Pulling FanGraphs {args.year} dashboard actuals (qual=0, all teams)...")

    failures = []

    try:
        session = make_session(args.year)

        print("\nBatting:")
        bat_df = pull(session, "bat", args.year)
        if not bat_df.empty:
            bat_df.to_csv(bat_path, index=False, encoding="utf-8")
            print(f"  saved -> {bat_path}  ({len(bat_df)} rows x {len(bat_df.columns)} cols)")
        else:
            failures.append("batting")
            print(f"  KEPT existing {bat_path.name} (pull failed)")

        print("\nPitching:")
        pit_df = pull(session, "pit", args.year)
        if not pit_df.empty:
            pit_df.to_csv(pit_path, index=False, encoding="utf-8")
            print(f"  saved -> {pit_path}  ({len(pit_df)} rows x {len(pit_df.columns)} cols)")
        else:
            failures.append("pitching")
            print(f"  KEPT existing {pit_path.name} (pull failed)")

    except CloudflareBlocked:
        print("\n" + "=" * 66)
        print("BLOCKED BY CLOUDFLARE (403)")
        print("=" * 66)
        print("Every client configuration was rejected. This is not a rate")
        print("limit — retrying or waiting will not help.")
        print()
        print("Run with --probe to see the full matrix of what FanGraphs")
        print("accepts. Historically the cause was this script sending a")
        print("spoofed Chrome User-Agent; sending no custom headers works.")
        print()
        print("The CSVs in data/raw/fg_exports/ have been left untouched.")
        print("=" * 66)
        return 2

    if failures:
        # Non-zero exit so a scheduled run surfaces the failure instead of
        # silently proceeding with stale data.
        print(f"\nFAILED: {', '.join(failures)} — existing CSVs left untouched.")
        return 1

    print(f"\nDone. Run predict_awards.py to score {args.year}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
