# config.py
# ---------------------------------------------------------------
# Single source of truth for the mlb-awards-predictor project.
# Every script imports from here — no hardcoded paths anywhere else.
# ---------------------------------------------------------------

from pathlib import Path
from datetime import date

# ── Active year ───────────────────────────────────────────────────
CURRENT_YEAR = 2026      # season being predicted RIGHT NOW
TRAIN_START  = 1980      # first year of training data (Cy Young started 1956,
                         # but consistent advanced stats only from ~1980)
TRAIN_END    = 2025      # last year with complete voting results in Lahman
TEST_START   = 2022      # held-out test window start
TEST_END     = 2025      # held-out test window end

# ── Directory layout ──────────────────────────────────────────────
ROOT_DIR    = Path(__file__).parent          # project root
DATA_RAW    = ROOT_DIR / "data" / "raw"      # Lahman zip, FG cache (gitignored)
DATA_PROC   = ROOT_DIR / "data" / "processed"  # merged CSVs (gitignored)
DATA_VOTING = ROOT_DIR / "data" / "voting"   # historical award vote CSVs
MODELS_DIR  = ROOT_DIR / "models"            # saved .joblib files (gitignored)
PREDICTIONS = ROOT_DIR / "predictions"       # output CSVs (gitignored)
SRC_DIR     = ROOT_DIR / "src"
FG_EXPORT_DIR = DATA_RAW / "fg_exports"     # manually downloaded FG CSVs (gitignored)

# ── Lahman ────────────────────────────────────────────────────────
LAHMAN_ZIP    = DATA_RAW / "lahman_1871-2025_csv.zip"
LAHMAN_PREFIX = "lahman_1871-2025_csv/"

# ── Processed dataset file names ─────────────────────────────────
FULL_DATASET  = DATA_PROC / "player_season_full.csv"         # Lahman-only merge
FG_DATASET    = DATA_PROC / "player_season_features_fg.csv"  # + FanGraphs advanced stats

# ── FanGraphs pull range ──────────────────────────────────────────
FG_START = TRAIN_START
FG_END   = TRAIN_END          # update each preseason after Lahman 20XX zip is released

# ── Model targets ─────────────────────────────────────────────────
# Each entry: (task_name, label_column)
# top5   = finished in top 5 vote-getters for their league/year
# winner = finished #1 in vote points for their league/year
TASKS = {
    "MVP_top5":    "is_top5_MVP",
    "MVP_winner":  "is_winner_MVP",
    "CY_top5":     "is_top5_CY",
    "CY_winner":   "is_winner_CY",
}

# ── Random seed ───────────────────────────────────────────────────
RANDOM_STATE = 42

# ── FanGraphs projection type ─────────────────────────────────────
SEASON_START = date(CURRENT_YEAR, 3, 20)   # approximate Opening Day
SEASON_END   = date(CURRENT_YEAR, 9, 28)   # approximate last regular-season game

def fg_projection_type() -> str:
    """
    Return the Steamer endpoint variant to use for /api/projections calls.

    steamerr  — rest-of-season, updated weekly blending actuals + remaining.
                Used during the season so rankings reflect what's already happened.
    steamer   — full-season preseason projection, used in the offseason when
                no actuals exist yet (also the right component when building an
                additive actuals + projection blend later in the year).
    """
    today = date.today()
    if SEASON_START <= today <= SEASON_END:
        return "steamerr"
    return "steamer"

# ── FanGraphs credentials (WordPress Application Password) ───────────────────
# Store in config_local.py (gitignored) — never put real values here.
# Generate at: https://www.fangraphs.com/wp-admin/profile.php → Application Passwords
FG_USERNAME     = ""
FG_APP_PASSWORD = ""

try:
    from config_local import *  # noqa: F401,F403 — local overrides (gitignored)
except ImportError:
    pass

# Best model per task based on Top1HitRate + Recall@5
CHAMPION_MODEL = {
    "MVP_top5":   "model_logreg.joblib",      # tied, LR slightly more stable
    "MVP_winner": "model_logreg.joblib",       # Top1: 0.875 vs 0.500
    "CY_top5":    "model_randomforest.joblib", # Top1: 1.000 vs 0.750
    "CY_winner":  "model_randomforest.joblib", # Top1: 0.750 vs 0.250
}