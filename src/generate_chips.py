# src/generate_chips.py
# ---------------------------------------------------------------
# Generates the "Chip" (pipe-separated tags) and "Why" (one-line rationale)
# for the top-3 players in each award/league, using the Claude API.
#
# Drop this file next to predict_awards.py. In predict_awards.py, after
# t10_mvp / t10_cy are built, call:
#
#     from generate_chips import generate_chips
#     generate_chips(t10_mvp, t10_cy, MVP_STAT_COLS, CY_STAT_COLS, outdir)
#
# Output:
#   {outdir}/chip_data/chip_data_latest.csv   <- paste/import into ChipData tab
#   {outdir}/chip_data/chip_data_{ts}.csv     <- full archive (incl. award/league/rank)
#
# Setup:
#   pip install anthropic
#   Set ANTHROPIC_API_KEY as an environment variable (System Properties >
#   Environment Variables on Windows, or in run_and_push.ps1 before the
#   python call: $env:ANTHROPIC_API_KEY = "sk-ant-...")
# ---------------------------------------------------------------

import os
import re
import json
import time
from pathlib import Path
from datetime import datetime
from typing import List

import pandas as pd
from anthropic import Anthropic

# Haiku is the right model here: fast, cheap, no extended-thinking blocks,
# and highly reliable at returning strict JSON for simple structured prompts.
# Sonnet/Opus prepend ThinkingBlocks that complicate content parsing for no
# quality gain on 3-word tags for 12 players per week.
CLAUDE_MODEL = "claude-haiku-4-5-20251001"

# client is intentionally NOT instantiated at module level — importing this
# module without ANTHROPIC_API_KEY set in the environment must not raise.
# It is created lazily inside generate_chips() instead.


def make_player_key(name: str, team_abbr: str) -> str:
    """
    Stable slug matching the _headshot_key() convention in predict_awards.py:
      firstname_lastname_team, all lowercase, accents stripped (NFD → ASCII).

    e.g. 'José Ramírez', 'CLE'       → 'jose_ramirez_cle'
         'Jazz Chisholm Jr.', 'NYY'  → 'jazz_chisholm_jr_nyy'
         'Aaron Judge', 'NYY'        → 'aaron_judge_nyy'

    Note: when calling generate_chips() with DataFrames produced by top10()
    in predict_awards.py, the DataFrames already have a pre-computed
    player_key column (built via this same logic).  That column is used
    directly rather than recomputing here, so this function is exposed
    mainly for standalone use and testing.
    """
    import unicodedata
    combined = f"{name} {team_abbr}"
    nfd = unicodedata.normalize("NFD", combined).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", nfd.lower())).strip("_")


CHIP_SYSTEM_PROMPT = """You generate two short labels for MLB awards-prediction graphics: a "Chip" and a "Why".

Rules:
- "chip": exactly 3 short tags (1-3 words each), separated by " | ". Each tag names a distinct statistical strength (e.g. "Power", "Low K%", "Run producer", "Elite command", "Innings eater").
- "why": one sentence, under 100 characters, giving the stat-driven reason for the player's rank. No fluff, no adjectives that aren't backed by a stat in the input.
- Base every word ONLY on the stats given in the user message. Never invent, infer, or assume a stat that wasn't provided.
- Tags describe skill/performance, not identity — don't put the player's name, team, or rank number in the chip itself.
- Output ONLY valid JSON in exactly this shape, nothing else, no markdown fences, no preamble:
{"chip": "Tag1 | Tag2 | Tag3", "why": "One sentence explanation."}"""


def build_user_message(player_row: pd.Series, stat_cols: List[str], award: str) -> str:
    stats_str = ", ".join(
        f"{col}: {player_row[col]}"
        for col in stat_cols
        if col in player_row and pd.notna(player_row[col])
    )
    award_label = "MVP" if award == "mvp" else "Cy Young"

    return f"""Player: {player_row['Name']} ({player_row['Team']})
Rank: #{int(player_row['rank'])} in our model's {award_label} prediction
Stats: {stats_str}"""


def generate_chip_for_player(
    client: Anthropic,
    player_row: pd.Series,
    stat_cols: List[str],
    award: str,
    retries: int = 2,
) -> dict:
    user_message = build_user_message(player_row, stat_cols, award)
    for attempt in range(retries + 1):
        try:
            resp = client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=200,
                system=CHIP_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            )
            # Find the first TextBlock — models with extended thinking prepend
            # a ThinkingBlock before the actual response text.
            text_block = next((b for b in resp.content if hasattr(b, "text")), None)
            if text_block is None:
                raise ValueError("No text block in response")
            text = re.sub(r"^```json|```$", "", text_block.text.strip()).strip()
            data = json.loads(text)
            return {"chip": data["chip"], "why": data["why"]}
        except Exception as e:
            if attempt == retries:
                print(f"  ! Chip generation failed for {player_row.get('Name', '?')}: {e}")
                return {"chip": "", "why": ""}
            time.sleep(1.5)


def generate_chips(
    t10_mvp: pd.DataFrame,
    t10_cy: pd.DataFrame,
    mvp_stat_cols: List[str],
    cy_stat_cols: List[str],
    outdir: Path,
    top_n: int = 3,
) -> Path:
    """
    Generates Chip/Why for the top_n players per league per award and
    writes chip_data_latest.csv + a timestamped archive.

    Uses the player_key column already present in t10_mvp / t10_cy
    (populated by top10() in predict_awards.py) so keys are guaranteed
    to match the Headshots tab convention.
    """
    client = Anthropic()  # raises AuthenticationError here if key not set

    rows = []
    for award, df, stat_cols in [("mvp", t10_mvp, mvp_stat_cols), ("cy", t10_cy, cy_stat_cols)]:
        for lg in ["AL", "NL"]:
            sub = df[df["lgID"] == lg].sort_values("rank").head(top_n)
            for _, player in sub.iterrows():
                print(f"  Generating chip: {player['Name']} ({lg} {award.upper()})")
                result = generate_chip_for_player(client, player, stat_cols, award)
                # Use the pre-computed player_key from the DataFrame when available;
                # fall back to make_player_key() if the column is absent (standalone use).
                pkey = (
                    player["player_key"]
                    if "player_key" in player and pd.notna(player.get("player_key"))
                    else make_player_key(player["Name"], player["Team"])
                )
                rows.append(
                    {
                        "player_key": pkey,
                        "Chip": result["chip"],
                        "Why": result["why"],
                        "award": award,
                        "league": lg,
                        "rank": int(player["rank"]),
                    }
                )

    chip_df = pd.DataFrame(rows)

    chip_dir = Path(outdir) / "chip_data"
    chip_dir.mkdir(exist_ok=True, parents=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    latest_path = chip_dir / "chip_data_latest.csv"
    archive_path = chip_dir / f"chip_data_{ts}.csv"

    # Sheet-facing file: exactly the 3 columns for the ChipData tab
    chip_df[["player_key", "Chip", "Why"]].to_csv(latest_path, index=False)
    # Full version, kept for your own reference/debugging
    chip_df.to_csv(archive_path, index=False)

    print(f"\n✅  Chip data saved to {latest_path}")
    return latest_path
