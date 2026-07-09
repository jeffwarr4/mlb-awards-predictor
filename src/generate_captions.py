# src/generate_captions.py
# ---------------------------------------------------------------
# Generates ready-to-paste Instagram and Facebook captions with
# hashtags for each award/league graphic produced by render_graphics.py.
#
# Standalone:
#   python src/generate_captions.py                  # all 4
#   python src/generate_captions.py --award mvp --league AL
#
# Called from predict_awards.py after render_all_graphics():
#   from generate_captions import generate_all_captions
#   generate_all_captions(t10_mvp, t10_cy, outdir, year)
#
# Output:
#   predictions/{year}/graphics/caption_{award}_{league}.txt
# ---------------------------------------------------------------

import os, re, json, time, argparse
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict

import pandas as pd
from anthropic import Anthropic

CLAUDE_MODEL = "claude-haiku-4-5-20251001"

MVP_STAT_COLS = ["bat_WAR_fg", "bat_wRC_plus", "bat_AVG", "bat_OPS", "HR", "RBI"]
CY_STAT_COLS  = ["pit_WAR_fg", "pit_FIP", "pit_Kpct", "SO_pit", "IP"]

STAT_LABELS = {
    "bat_WAR_fg": "fWAR", "bat_wRC_plus": "wRC+", "bat_AVG": "AVG",
    "bat_OPS": "OPS",     "HR": "HR",              "RBI": "RBI",
    "pit_WAR_fg": "fWAR", "pit_FIP": "FIP",        "pit_Kpct": "K%",
    "SO_pit": "K's",      "IP": "IP",
}

# Team name mapping for readable hashtags
TEAM_HASHTAGS: Dict[str, str] = {
    "NYY": "#Yankees",   "NYM": "#Mets",      "BOS": "#RedSox",
    "BAL": "#Orioles",   "TBR": "#RaysUp",    "TOR": "#BlueJays",
    "DET": "#Tigers",    "CLE": "#Guardians", "CWS": "#WhiteSox",
    "MIN": "#Twins",     "KCR": "#Royals",    "HOU": "#Astros",
    "TEX": "#Rangers",   "OAK": "#Athletics", "SEA": "#Mariners",
    "LAA": "#Angels",    "LAD": "#Dodgers",   "SFG": "#SFGiants",
    "SDP": "#Padres",    "ARI": "#Dbacks",    "COL": "#Rockies",
    "ATL": "#Braves",    "MIA": "#Marlins",   "PHI": "#Phillies",
    "NYM": "#Mets",      "WSN": "#Nationals", "MIL": "#Brewers",
    "CHC": "#Cubs",      "CIN": "#Reds",      "PIT": "#Pirates",
    "STL": "#Cardinals",
}

SYSTEM_PROMPT = """You are a sports social media manager for STINGER, an MLB awards prediction tracker.

Write a weekly update post for one award/league graphic. Produce:
1. A punchy caption body — 2 sentences max, under 220 characters total. Data-driven and slightly hype. Reference actual stats from the input. No filler ("Let's dive in!", "Check this out!"). 1-2 emoji max. Do NOT include hashtags in the body.
2. 15 hashtags for Instagram. Choose ONLY from the approved base tags below plus player-name and team-hashtag tags from the user message. Do NOT invent tags that aren't in the approved list or derived from the player/team names provided.
3. 5 hashtags for Facebook — the 5 most relevant from your Instagram list (favour generic MLB + award tags over player/team specifics).

Approved base hashtags (pick the most relevant, always include #StingerMLB):
#MLB #BaseballTwitter #MLBTwitter #BaseballSZN #MLB2026
#MVPRace #ALMVP #NLMVP #CyYoung #ALCyYoung #NLCyYoung #CyYoungAward #MVPAward
#AmericanLeague #NationalLeague #BaseballStats #MLBStats #MLBHighlights
#StingerMLB #StingerSports

Output ONLY valid JSON in exactly this shape, no markdown fences, no preamble:
{"caption": "...", "hashtags_instagram": ["#tag1", ...], "hashtags_facebook": ["#tag1", ...]}"""


def _fmt(col: str, val) -> str:
    try:
        v = float(val)
    except (TypeError, ValueError):
        return str(val)
    if col in ("bat_AVG", "bat_OPS"):
        s = f"{v:.3f}"
        return s if v >= 1.0 else s.lstrip("0") or "0"
    if col in ("bat_WAR_fg", "pit_WAR_fg"):
        return f"{v:.1f}"
    if col == "pit_FIP":
        return f"{v:.2f}"
    if col in ("pit_Kpct",):
        pct = v * 100 if v <= 1.0 else v
        return f"{pct:.1f}%"
    if col in ("HR", "RBI", "SO_pit"):
        return str(int(round(v)))
    if col == "IP":
        return f"{v:.1f}"
    return f"{v:.1f}"


def _player_summary(player: pd.Series, stat_cols: List[str]) -> str:
    stats = ", ".join(
        f"{STAT_LABELS.get(c, c)}: {_fmt(c, player[c])}"
        for c in stat_cols if c in player and pd.notna(player[c])
    )
    return f"#{int(player['rank'])} {player['Name']} ({player['Team']}): {stats}"


def _build_prompt(df: pd.DataFrame, award: str, league: str,
                  stat_cols: List[str], chip_df: pd.DataFrame,
                  week_label: str) -> str:
    top3 = df[df["lgID"] == league].sort_values("rank").head(3)
    lines = [f"Award: {league} {'MVP' if award == 'mvp' else 'Cy Young'}",
             f"Week: {week_label}", "", "Top 3 contenders:"]
    for _, p in top3.iterrows():
        line = _player_summary(p, stat_cols)
        pkey = str(p.get("player_key", ""))
        if chip_df is not None and not chip_df.empty and pkey in chip_df.index:
            why = chip_df.loc[pkey, "Why"]
            if isinstance(why, str) and why:
                line += f"  | {why}"
        lines.append(line)
    return "\n".join(lines)


def _format_output(award: str, league: str, week_label: str,
                   data: dict) -> str:
    award_label = "MVP" if award == "mvp" else "Cy Young"
    ig_tags  = " ".join(data["hashtags_instagram"])
    fb_tags  = " ".join(data["hashtags_facebook"])
    return (
        f"=== {league} {award_label.upper()} TRACKER — {week_label} ===\n\n"
        f"CAPTION:\n{data['caption']}\n\n"
        f"INSTAGRAM HASHTAGS:\n{ig_tags}\n\n"
        f"FACEBOOK HASHTAGS:\n{fb_tags}\n"
    )


def generate_caption(
    client: Anthropic,
    df: pd.DataFrame,
    award: str,
    league: str,
    stat_cols: List[str],
    chip_df: pd.DataFrame,
    week_label: str,
    retries: int = 2,
) -> Optional[dict]:
    prompt = _build_prompt(df, award, league, stat_cols, chip_df, week_label)
    for attempt in range(retries + 1):
        try:
            resp = client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=400,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            block = next((b for b in resp.content if hasattr(b, "text")), None)
            if block is None:
                raise ValueError("No text block in response")
            text = re.sub(r"^```json|```$", "", block.text.strip(), flags=re.MULTILINE).strip()
            return json.loads(text)
        except Exception as e:
            if attempt == retries:
                print(f"  ! Caption generation failed for {league} {award}: {e}")
                return None
            time.sleep(1.5)


def generate_all_captions(
    t10_mvp: pd.DataFrame,
    t10_cy:  pd.DataFrame,
    outdir:  Path,
    year:    int = 2026,
) -> dict:
    """
    Generate captions for all 4 award/league combos and save as .txt files.
    Returns a dict of {key: path} for the saved files.
    Silently no-ops if ANTHROPIC_API_KEY is not set.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("  [WARN] ANTHROPIC_API_KEY not set — caption generation skipped")
        return {}

    client     = Anthropic()
    now        = datetime.now()
    week_label = f"Week of {now.strftime('%B')} {now.day}, {now.year}"
    repo_root  = Path(__file__).parent.parent
    chip_path  = repo_root / "predictions" / str(year) / "chip_data" / "chip_data_latest.csv"
    chip_df    = pd.read_csv(chip_path).set_index("player_key") \
                 if chip_path.exists() else pd.DataFrame()

    gfx_dir = outdir / "graphics"
    gfx_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    for award, df, stat_cols in [("mvp", t10_mvp, MVP_STAT_COLS),
                                   ("cy",  t10_cy,  CY_STAT_COLS)]:
        for league in ("AL", "NL"):
            print(f"  Generating caption: {league} {award.upper()}")
            data = generate_caption(client, df, award, league,
                                    stat_cols, chip_df, week_label)
            if data is None:
                continue
            txt  = _format_output(award, league, week_label, data)
            path = gfx_dir / f"caption_{award}_{league.lower()}.txt"
            path.write_text(txt, encoding="utf-8")
            print(f"  Saved → {path}")
            results[f"{league}_{award}_caption"] = str(path)

    return results


# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--award",  choices=["mvp","cy","both"], default="both")
    parser.add_argument("--league", choices=["AL","NL","both"],  default="both")
    parser.add_argument("--year",   type=int, default=2026)
    args = parser.parse_args()

    year     = args.year
    pred     = Path(__file__).parent.parent / "predictions" / str(year)
    awards   = ["mvp","cy"] if args.award  == "both" else [args.award]
    leagues  = ["AL","NL"]  if args.league == "both" else [args.league]

    t10_mvp = pd.concat([pd.read_csv(pred / f"top10_{lg.lower()}_mvp_latest.csv")
                          for lg in leagues if (pred / f"top10_{lg.lower()}_mvp_latest.csv").exists()],
                         ignore_index=True)
    t10_cy  = pd.concat([pd.read_csv(pred / f"top10_{lg.lower()}_cy_latest.csv")
                          for lg in leagues if (pred / f"top10_{lg.lower()}_cy_latest.csv").exists()],
                         ignore_index=True)

    generate_all_captions(t10_mvp, t10_cy, pred, year)


if __name__ == "__main__":
    main()
