# src/build_dataset.py
# ---------------------------------------------------------------
# Step 1 of the pipeline.
# Reads the Lahman zip and builds the base training dataset.
#
# Inputs:
#   data/raw/lahman_1871-2025_csv.zip
#
# Outputs:
#   data/processed/player_season_full.csv   ← all merged columns
#
# Run:
#   python src/build_dataset.py
# ---------------------------------------------------------------

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))  # project root on path

import zipfile
import numpy as np
import pandas as pd
from config import (
    LAHMAN_ZIP, LAHMAN_PREFIX, FULL_DATASET,
    TRAIN_START, DATA_PROC
)


def load_lahman(zip_path: Path, prefix: str) -> dict:
    """Load the five core Lahman tables into a dict of DataFrames."""
    print(f"📂  Loading Lahman from {zip_path.name} ...")
    with zipfile.ZipFile(zip_path, "r") as z:
        tables = {}
        for name in ["Batting", "Pitching", "Fielding", "Teams", "AwardsSharePlayers"]:
            tables[name] = pd.read_csv(z.open(prefix + name + ".csv"))
            print(f"    {name}: {len(tables[name]):,} rows")
    return tables


def filter_years(tables: dict, start: int) -> dict:
    year_cols = {
        "Batting": "yearID", "Pitching": "yearID", "Fielding": "yearID",
        "Teams": "yearID", "AwardsSharePlayers": "yearID"
    }
    return {
        name: df[df[year_cols[name]] >= start].copy()
        for name, df in tables.items()
    }


def build_team_winpct(teams: pd.DataFrame) -> pd.DataFrame:
    """One row per (yearID, teamID, lgID) with WinPct."""
    teams = teams.copy()
    teams["WinPct"] = teams["W"] / (teams["W"] + teams["L"])

    # Playoff flags — useful features, include for training
    for col in ["DivWin", "WCWin", "LgWin", "WSWin"]:
        if col in teams.columns:
            teams[col] = (teams[col] == "Y").astype(int)
        else:
            teams[col] = 0

    keep = ["yearID", "teamID", "lgID", "WinPct", "DivWin", "WCWin", "LgWin", "WSWin"]
    return teams[[c for c in keep if c in teams.columns]].copy()


def primary_team(batting: pd.DataFrame, pitching: pd.DataFrame) -> pd.DataFrame:
    """
    For traded players, pick the team where they spent the most time.
    Batters weighted by AB; pitchers by IPouts; fallback to G.
    Returns one row per (playerID, yearID) with teamID.
    """
    bat_wt = batting.groupby(["playerID","yearID","teamID"], as_index=False).agg(
        AB=("AB","sum"), G=("G","sum")
    )
    bat_wt["w"] = bat_wt["AB"].fillna(0).where(bat_wt["AB"]>0, bat_wt["G"])

    pit_wt = pitching.groupby(["playerID","yearID","teamID"], as_index=False).agg(
        IPouts=("IPouts","sum"), G=("G","sum")
    )
    pit_wt["w"] = pit_wt["IPouts"].fillna(0).where(pit_wt["IPouts"]>0, pit_wt["G"])

    bat_primary = (bat_wt.sort_values("w", ascending=False)
                         .drop_duplicates(["playerID","yearID"])
                         [["playerID","yearID","teamID"]]
                         .rename(columns={"teamID":"bat_team"}))

    pit_primary = (pit_wt.sort_values("w", ascending=False)
                         .drop_duplicates(["playerID","yearID"])
                         [["playerID","yearID","teamID"]]
                         .rename(columns={"teamID":"pit_team"}))

    primary = pd.merge(bat_primary, pit_primary, on=["playerID","yearID"], how="outer")
    primary["teamID"] = primary["bat_team"].fillna(primary["pit_team"])
    return primary[["playerID","yearID","teamID"]].dropna(subset=["teamID"])


def aggregate_stats(batting, pitching, fielding) -> pd.DataFrame:
    """Aggregate each table to one row per (playerID, yearID)."""
    bat_agg = batting.groupby(["playerID","yearID"], as_index=False).sum(numeric_only=True)
    pit_agg = pitching.groupby(["playerID","yearID"], as_index=False).sum(numeric_only=True)
    fld_agg = fielding.groupby(["playerID","yearID"], as_index=False).agg(
        G_fld=("G","sum"), PO=("PO","sum"), A=("A","sum"),
        E=("E","sum"), DP=("DP","sum")
    )
    fld_agg["FieldPct"] = (
        (fld_agg["PO"] + fld_agg["A"]) /
        (fld_agg["PO"] + fld_agg["A"] + fld_agg["E"]).replace(0, np.nan)
    )

    stats = bat_agg.merge(pit_agg, on=["playerID","yearID"],
                          how="outer", suffixes=("_bat","_pit"))
    stats = stats.merge(fld_agg, on=["playerID","yearID"], how="left")
    return stats


def build_batting_derived(df: pd.DataFrame) -> pd.DataFrame:
    """OBP, SLG, OPS from Lahman counting stats."""
    df = df.copy()

    # Handle suffixed column names from the bat/pit merge
    h   = df.get("H_bat",   df.get("H",   pd.Series(0, index=df.index)))
    ab  = df.get("AB",      pd.Series(0, index=df.index))
    bb  = df.get("BB_bat",  df.get("BB",  pd.Series(0, index=df.index)))
    hbp = df.get("HBP_bat", df.get("HBP", pd.Series(0, index=df.index)))
    sf  = df.get("SF_bat",  df.get("SF",  pd.Series(0, index=df.index)))
    b2  = df.get("2B",      pd.Series(0, index=df.index))
    b3  = df.get("3B",      pd.Series(0, index=df.index))
    hr  = df.get("HR_bat",  df.get("HR",  pd.Series(0, index=df.index)))

    for col in (h, ab, bb, hbp, sf, b2, b3, hr):
        col.fillna(0, inplace=True)

    singles = h - b2 - b3 - hr
    obp_denom = (ab + bb + hbp + sf).replace(0, np.nan)
    slg_denom = ab.replace(0, np.nan)

    df["OBP"] = (h + bb + hbp) / obp_denom
    df["SLG"] = (singles + 2*b2 + 3*b3 + 4*hr) / slg_denom
    df["OPS"] = df["OBP"].fillna(0) + df["SLG"].fillna(0)
    df[["OBP","SLG","OPS"]] = df[["OBP","SLG","OPS"]].fillna(0)
    return df


def build_awards_labels(awards: pd.DataFrame) -> pd.DataFrame:
    """
    Build four binary labels per player-season-league:
      is_top5_MVP    : finished in top 5 MVP vote points
      is_winner_MVP  : finished #1 in MVP vote points
      is_top5_CY     : finished in top 5 Cy Young vote points
      is_winner_CY   : finished #1 in Cy Young vote points
    """
    awards = awards.copy()
    awards = awards[awards["awardID"].isin(["Most Valuable Player","Cy Young Award"])].copy()
    awards["voteShare"] = awards["pointsWon"] / awards["pointsMax"].replace(0, np.nan)
    awards["voteShare"] = awards["voteShare"].fillna(0)

    def flag(df, col, n):
        df = df.copy()
        df["_rank"] = df.groupby(["yearID","lgID"])[col].rank(
            ascending=False, method="first"
        )
        top_n = (df["_rank"] <= n).astype(int)
        winner = (df.groupby(["yearID","lgID"])[col]
                    .transform(lambda x: x == x.max())).astype(int)
        return top_n, winner

    mvp = awards[awards["awardID"]=="Most Valuable Player"].copy()
    cy  = awards[awards["awardID"]=="Cy Young Award"].copy()

    mvp["is_top5_MVP"],   mvp["is_winner_MVP"]  = flag(mvp, "voteShare", 5)
    cy["is_top5_CY"],     cy["is_winner_CY"]    = flag(cy,  "voteShare", 5)

    mvp_out = mvp[["playerID","yearID","lgID",
                   "voteShare","pointsWon","pointsMax","votesFirst",
                   "is_top5_MVP","is_winner_MVP"]].copy()
    mvp_out.columns = ["playerID","yearID","lgID",
                       "voteShare_MVP","pointsWon_MVP","pointsMax_MVP","votesFirst_MVP",
                       "is_top5_MVP","is_winner_MVP"]

    cy_out = cy[["playerID","yearID","lgID",
                 "voteShare","pointsWon","pointsMax","votesFirst",
                 "is_top5_CY","is_winner_CY"]].copy()
    cy_out.columns = ["playerID","yearID","lgID",
                      "voteShare_CY","pointsWon_CY","pointsMax_CY","votesFirst_CY",
                      "is_top5_CY","is_winner_CY"]

    # Merge MVP and CY labels on (playerID, yearID, lgID)
    out = mvp_out.merge(cy_out, on=["playerID","yearID","lgID"], how="outer")
    for col in ["is_top5_MVP","is_winner_MVP","is_top5_CY","is_winner_CY"]:
        out[col] = out[col].fillna(0).astype(int)
    return out


def build_dataset() -> pd.DataFrame:
    """Full pipeline: load → filter → aggregate → merge → label → derive."""
    DATA_PROC.mkdir(parents=True, exist_ok=True)

    # 1 — Load & filter
    tables  = load_lahman(LAHMAN_ZIP, LAHMAN_PREFIX)
    tables  = filter_years(tables, TRAIN_START)
    batting, pitching, fielding = tables["Batting"], tables["Pitching"], tables["Fielding"]
    teams_raw, awards_raw = tables["Teams"], tables["AwardsSharePlayers"]

    # 2 — Team WinPct + flags
    team_wp = build_team_winpct(teams_raw)

    # 3 — Primary team per player-year
    primary = primary_team(batting, pitching)
    primary = primary.merge(
        team_wp[["yearID","teamID","lgID"]].drop_duplicates(),
        on=["yearID","teamID"], how="left"
    )

    # 4 — Aggregate stats
    print("🔧  Aggregating player stats ...")
    stats = aggregate_stats(batting, pitching, fielding)

    # 5 — Attach team + WinPct
    stats = stats.merge(primary[["playerID","yearID","teamID","lgID"]],
                        on=["playerID","yearID"], how="left")
    stats = stats.merge(
    team_wp.drop(columns=["lgID"], errors="ignore"),
    on=["yearID","teamID"], how="left"
)

    # 6 — Derive OBP / SLG / OPS
    # stats = build_batting_derived(stats) --removing Lahman derived stats

    # 7 — Build award labels
    print("🏆  Building award labels ...")
    labels = build_awards_labels(awards_raw)
    full = stats.merge(labels, on=["playerID","yearID","lgID"], how="left")

    # Fill label NaNs: players with no votes get 0 for all labels
    label_cols = ["is_top5_MVP","is_winner_MVP","is_top5_CY","is_winner_CY"]
    vote_cols  = ["voteShare_MVP","pointsWon_MVP","pointsMax_MVP","votesFirst_MVP",
                  "voteShare_CY","pointsWon_CY","pointsMax_CY","votesFirst_CY"]
    for c in label_cols: full[c] = full[c].fillna(0).astype(int)
    for c in vote_cols:
        if c in full.columns: full[c] = full[c].fillna(0)

    # 8 — Save
    full.to_csv(FULL_DATASET, index=False)
    print(f"\n✅  Saved {len(full):,} rows → {FULL_DATASET}")
    print(f"    Columns: {len(full.columns)}")
    print(f"    Year range: {full['yearID'].min()}–{full['yearID'].max()}")
    for lbl in label_cols:
        print(f"    {lbl}: {full[lbl].sum()} positive rows")

    return full


if __name__ == "__main__":
    build_dataset()
