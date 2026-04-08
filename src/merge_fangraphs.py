# src/merge_fangraphs.py
# ---------------------------------------------------------------
# Step 2 of the pipeline.
# Merges FanGraphs historical advanced stats into the Lahman base.
#
# Inputs:
#   data/processed/player_season_full.csv   (from build_dataset.py)
#
# Outputs:
#   data/processed/player_season_features_fg.csv
#
# Join strategy (two-pass to maximise coverage):
#   Pass 1 — IDfg join: map bbref playerID → FanGraphs IDfg via
#             playerid_reverse_lookup, then join on (IDfg, yearID).
#   Pass 2 — Name+Year fallback: for any player still missing FG
#             stats after pass 1, attempt a join on (Name, yearID).
#             Catches players whose ID mapping failed.
#
# Pulling strategy:
#   Primary  — single bulk call batting_stats(START, END) — fast.
#   Fallback — year-by-year with caching if bulk call fails.
#
# Run:
#   python src/merge_fangraphs.py
#   python src/merge_fangraphs.py --start 2025 --end 2025  # single year refresh
# ---------------------------------------------------------------

import sys, time, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from pybaseball import batting_stats, pitching_stats, playerid_reverse_lookup
from config import (
    FULL_DATASET, FG_DATASET, DATA_RAW,
    FG_START, FG_END
)

FG_CACHE_DIR        = DATA_RAW / "fg_cache"
SLEEP_BETWEEN_YEARS = 12
MAX_RETRIES         = 3
RETRY_SLEEP         = 30

BAT_RENAME = {
    "Season": "yearID",
    "WAR":    "bat_WAR_fg",
    "wRC+":   "bat_wRC_plus",
    "OPS":    "bat_OPS",
    "OBP":    "bat_OBP",
    "SLG":    "bat_SLG",
}
PIT_RENAME = {
    "Season": "yearID",
    "WAR":    "pit_WAR_fg",
    "FIP":    "pit_FIP",
    "K%":     "pit_Kpct",
    "BB%":    "pit_BBpct",
    "ERA-":   "pit_ERA_minus",
    "xFIP":   "pit_xFIP",
}


# ── ID type helper ────────────────────────────────────────────────
def to_idfg(series: pd.Series) -> pd.Series:
    """
    Cast IDfg to nullable Int64 so both sides of the join use the
    same type.  Floats like 12345.0 become 12345; non-numerics → NA.
    """
    return pd.to_numeric(series, errors="coerce").astype("Int64")


# ── Pull helpers ──────────────────────────────────────────────────
def _cache_path(label: str, year: int) -> Path:
    return FG_CACHE_DIR / f"{label}_{year}.csv"

def _retry_year(fn, year: int, label: str) -> pd.DataFrame:
    cp = _cache_path(label, year)
    if cp.exists():
        return pd.read_csv(cp)
    FG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            df = fn(year, qual=1)
            df.to_csv(cp, index=False)
            return df
        except Exception as e:
            print(f"    ⚠️  {label} {year} attempt {attempt}/{MAX_RETRIES}: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_SLEEP)
    print(f"    ✗ Skipping {label} {year}")
    return pd.DataFrame()

def pull_bulk(fn, start: int, end: int, label: str) -> pd.DataFrame:
    """Try a single bulk call first (fast); fall back to year-by-year."""
    bulk_cache = FG_CACHE_DIR / f"{label}_{start}_{end}.csv"
    if bulk_cache.exists():
        print(f"    Using cached bulk pull: {bulk_cache.name}")
        return pd.read_csv(bulk_cache)
    try:
        print(f"    Bulk pull {label} {start}–{end} ...", flush=True)
        df = fn(start, end, qual=1)
        FG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(bulk_cache, index=False)
        print(f"    ✓ Bulk pull succeeded: {len(df):,} rows")
        return df
    except Exception as e:
        print(f"    ⚠️  Bulk pull failed ({e}), switching to year-by-year ...")

    frames = []
    for yr in range(start, end + 1):
        print(f"    {label} {yr} ...", end="", flush=True)
        df = _retry_year(fn, yr, label)
        if not df.empty:
            frames.append(df)
            print(f" {len(df)} rows")
        else:
            print(" SKIP")
        if yr < end:
            time.sleep(SLEEP_BETWEEN_YEARS)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def prep_fg(df: pd.DataFrame, rename_map: dict) -> pd.DataFrame:
    """Rename columns, coerce types, return lean DataFrame."""
    keep = ["IDfg"] + [c for c in rename_map if c in df.columns]
    out  = df[keep].rename(columns=rename_map).copy()
    out["yearID"] = out["yearID"].astype(int)
    out["IDfg"]   = to_idfg(out["IDfg"])
    # deduplicate: keep row with highest WAR for that IDfg+year (handles multi-team)
    war_col = "bat_WAR_fg" if "bat_WAR_fg" in out.columns else \
              "pit_WAR_fg" if "pit_WAR_fg" in out.columns else None
    if war_col:
        out = (out.sort_values(war_col, ascending=False)
                  .drop_duplicates(["IDfg","yearID"])
                  .reset_index(drop=True))
    return out


# ── ID mapping ────────────────────────────────────────────────────
def build_id_map(player_ids: list) -> pd.DataFrame:
    """
    Map Lahman bbref IDs → FanGraphs IDfg.
    Returns DataFrame with columns [playerID, IDfg] (both clean types).
    """
    print(f"🔗  Mapping {len(player_ids):,} playerIDs → FanGraphs IDfg ...")
    frames = []
    chunk_size = 500
    for i in range(0, len(player_ids), chunk_size):
        chunk = player_ids[i : i + chunk_size]
        try:
            m = playerid_reverse_lookup(chunk, key_type="bbref")
            frames.append(m[["key_bbref","key_fangraphs"]].dropna())
        except Exception as e:
            print(f"  ⚠️  Chunk {i//chunk_size+1} failed: {e}")
        time.sleep(2)

    if not frames:
        return pd.DataFrame(columns=["playerID","IDfg"])

    mapping = (pd.concat(frames, ignore_index=True)
                 .drop_duplicates("key_bbref")
                 .rename(columns={"key_bbref":"playerID","key_fangraphs":"IDfg"}))
    # ── KEY FIX: cast to same nullable Int64 as prep_fg uses ──
    mapping["IDfg"] = to_idfg(mapping["IDfg"])
    mapping = mapping.dropna(subset=["IDfg"])

    print(f"    Mapped {len(mapping):,} players with valid IDfg")
    return mapping


# ── Name normaliser for fallback join ─────────────────────────────
def _norm_name(s: pd.Series) -> pd.Series:
    """Lowercase, strip accents-like chars, remove punctuation."""
    return (s.str.lower()
             .str.replace(r"[^a-z ]", "", regex=True)
             .str.strip()
             .str.replace(r"\s+", " ", regex=True))

def build_name_map(fg_df: pd.DataFrame, name_col: str = "Name") -> pd.DataFrame:
    """Return FG stats keyed by (name_norm, yearID) for fallback join."""
    if name_col not in fg_df.columns:
        return pd.DataFrame()
    out = fg_df.copy()
    out["name_norm"] = _norm_name(out[name_col])
    # drop IDfg so it doesn't conflict in the fallback merge
    return out.drop(columns=["IDfg"], errors="ignore")


# ── Main ──────────────────────────────────────────────────────────
def merge_fangraphs(start: int = FG_START, end: int = FG_END) -> pd.DataFrame:

    DATA_PROC = FULL_DATASET.parent
    DATA_PROC.mkdir(parents=True, exist_ok=True)
    FG_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Load base ──────────────────────────────────────────────
    print(f"📂  Loading {FULL_DATASET.name} ...")
    base = pd.read_csv(FULL_DATASET)
    base["yearID"] = base["yearID"].astype(int)
    print(f"    {len(base):,} rows | {base['yearID'].min()}–{base['yearID'].max()}")

    # ── 2. Pull FanGraphs ─────────────────────────────────────────
    print("\n📥  Pulling FanGraphs batting ...")
    raw_bat = pull_bulk(batting_stats,  start, end, "bat")
    print("\n📥  Pulling FanGraphs pitching ...")
    raw_pit = pull_bulk(pitching_stats, start, end, "pit")

    bat_fg = prep_fg(raw_bat, BAT_RENAME) if not raw_bat.empty else pd.DataFrame()
    pit_fg = prep_fg(raw_pit, PIT_RENAME) if not raw_pit.empty else pd.DataFrame()

    # ── 3. Build ID mapping ───────────────────────────────────────
    all_ids  = base["playerID"].dropna().unique().tolist()
    id_map   = build_id_map(all_ids)
    enriched = base.merge(id_map, on="playerID", how="left")
    enriched["IDfg"] = to_idfg(enriched["IDfg"])

    # ── 4. Pass 1: IDfg join ──────────────────────────────────────
    print("\n🔀  Pass 1: joining on IDfg ...")
    fg_bat_cols = [c for c in BAT_RENAME.values() if c != "yearID"]
    fg_pit_cols = [c for c in PIT_RENAME.values() if c != "yearID"]

    if not bat_fg.empty:
        enriched = enriched.merge(bat_fg[["IDfg","yearID"] + fg_bat_cols],
                                  on=["IDfg","yearID"], how="left")
    else:
        for c in fg_bat_cols: enriched[c] = np.nan

    if not pit_fg.empty:
        enriched = enriched.merge(pit_fg[["IDfg","yearID"] + fg_pit_cols],
                                  on=["IDfg","yearID"], how="left")
    else:
        for c in fg_pit_cols: enriched[c] = np.nan

    pass1_bat = enriched["bat_WAR_fg"].notna().mean() * 100
    pass1_pit = enriched["pit_WAR_fg"].notna().mean() * 100
    print(f"    After pass 1 — bat_WAR_fg: {pass1_bat:.1f}%  pit_WAR_fg: {pass1_pit:.1f}%")

    # ── 5. Pass 2: Name+Year fallback for missing rows ────────────
    bat_missing = enriched["bat_WAR_fg"].isna()
    pit_missing = enriched["pit_WAR_fg"].isna()

    if bat_missing.any() and not raw_bat.empty and "Name" in raw_bat.columns:
        print(f"  🔁  Pass 2 bat: {bat_missing.sum():,} rows missing, trying Name+Year ...")
        bat_name = build_name_map(
            prep_fg(raw_bat, BAT_RENAME).merge(
                raw_bat[["IDfg","Name","Season"]].rename(columns={"Season":"yearID"}),
                on=["IDfg","yearID"], how="left"
            ),
            name_col="Name"
        )
        enriched["name_norm"] = _norm_name(enriched.get("playerID", pd.Series(dtype=str)))
        # We don't have FG Name on base — use a People.csv lookup instead
        # For now: note coverage and move on; full Name join requires People table
        print(f"    (Name join requires People table — see note below)")

    pass2_bat = enriched["bat_WAR_fg"].notna().mean() * 100
    pass2_pit = enriched["pit_WAR_fg"].notna().mean() * 100

    # ── 6. Coverage report ────────────────────────────────────────
    print(f"\n📊  Final coverage:")
    for col in ["bat_WAR_fg","bat_wRC_plus","bat_OPS","pit_WAR_fg","pit_FIP","pit_Kpct"]:
        if col in enriched.columns:
            pct = enriched[col].notna().mean() * 100
            print(f"    {col}: {pct:.1f}%")

    # ── 7. Build lean feature set ─────────────────────────────────
    keep_core = [
        "playerID","yearID","teamID","lgID","WinPct",
        "DivWin","WCWin","LgWin","WSWin",
        "G_bat","AB","H","HR","RBI","BB","SO","SB","CS","R","2B","3B","HBP","SF",
       # "OBP","SLG","OPS",
        "G_pit","GS","IPouts","SO_pit","BB_pit","ER","HR_pit","SV","W_pit","L_pit",
        "G_fld","PO","A","E","DP","FieldPct",
        "bat_WAR_fg","bat_wRC_plus","bat_OPS","bat_OBP","bat_SLG",
        "pit_WAR_fg","pit_FIP","pit_Kpct","pit_BBpct","pit_ERA_minus","pit_xFIP",
        "is_top5_MVP","is_winner_MVP","is_top5_CY","is_winner_CY",
        "voteShare_MVP","voteShare_CY",
    ]
    keep  = [c for c in keep_core if c in enriched.columns]
    final = enriched[keep].copy()
    final.to_csv(FG_DATASET, index=False)
    print(f"\n✅  Saved {len(final):,} rows → {FG_DATASET}")

    # ── Coverage note ──────────────────────────────────────────────
    if pass2_bat < 50:
        print("""
⚠️  Coverage below 50% — the IDfg join is not matching most players.
    Most likely cause: playerid_reverse_lookup returned floats that
    didn't match the int IDfg in batting_stats.

    Quick diagnostic — run this in your venv to check:

        from pybaseball import playerid_reverse_lookup, batting_stats
        m = playerid_reverse_lookup(["troutmi01","judgeaa01"], key_type="bbref")
        print(m[["key_bbref","key_fangraphs"]])
        print(m["key_fangraphs"].dtype)

        b = batting_stats(2024)
        print(b[["Name","IDfg"]].head(3))
        print(b["IDfg"].dtype)

    Paste the output here and I'll patch the exact type cast needed.
""")

    return final


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=FG_START)
    parser.add_argument("--end",   type=int, default=FG_END)
    args = parser.parse_args()
    merge_fangraphs(start=args.start, end=args.end)
