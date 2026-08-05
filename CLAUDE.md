# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

ML pipeline that predicts MLB MVP and Cy Young award outcomes (AL/NL, top-5 finish and outright winner) from
historical Lahman + FanGraphs data, then scores the live current season weekly during the MLB season via
GitHub Actions. Pure Python, no frontend.

## Environment

- Python venv lives at `C:\DevVenvs2\mlb-awards-venv` (not inside the repo). Use its `python.exe` to run anything
  locally, e.g. `C:\DevVenvs2\mlb-awards-venv\Scripts\python.exe src\predict_awards.py`.
- Dependencies: `pip install -r requirements.txt` (pandas, numpy, scikit-learn, joblib, pybaseball, requests).
- FanGraphs credentials (for authenticated actuals) go in `config_local.py` (gitignored) as `FG_USERNAME` /
  `FG_APP_PASSWORD`, generated from a FanGraphs WordPress Application Password. Never put real values in `config.py`.
- **Windows console gotcha**: scripts print emoji and `↑`/`↓` movement arrows that crash with
  `UnicodeEncodeError` under the default Windows cp1252 console. Run with `PYTHONIOENCODING=utf-8` prefixed,
  or pipe through something that doesn't re-encode.

## Commands

The pipeline is four sequential stages, each reading the previous stage's output:

```bash
# 1. Build base training set from a Lahman Baseball Database zip (data/raw/lahman_*.zip)
python src/build_dataset.py
#   -> data/processed/player_season_full.csv

# 2. Merge in FanGraphs historical advanced stats (WAR, wRC+, FIP, etc.)
python src/merge_fangraphs.py
python src/merge_fangraphs.py --start 2025 --end 2025   # refresh a single year
#   -> data/processed/player_season_features_fg.csv

# 3. Train all 4 models (LogReg + RandomForest each) on that dataset
python src/train_model.py
python src/train_model.py --task MVP_top5   # single task: MVP_top5 | MVP_winner | CY_top5 | CY_winner
#   -> models/<task>/model_logreg.joblib, model_randomforest.joblib, feature_columns.joblib,
#      logreg_top_coeffs.csv, rf_top_importances.csv, recall_at5_*.csv
#   -> models/metrics_summary.csv (AUC/AP/F1/Recall@5/Top1HitRate across all tasks)

# 4a. Pull current-season FanGraphs actuals (one request per stat type)
python src/pull_fg_current.py
python src/pull_fg_current.py --year 2025 --force
python src/pull_fg_current.py --probe          # diagnose 403s: prints a client/header matrix
#   -> data/raw/fg_exports/fg_bat_{year}.csv, fg_pit_{year}.csv

# 4b. Score the live season and write prediction CSVs
python src/predict_awards.py
#   -> predictions/<year>/top10_{al,nl}_{mvp,cy}_latest.csv + timestamped archive + top5_flat_*.csv
```

There is no formal test suite — validation is via `models/metrics_summary.csv` (AUC, Recall@5, Top1HitRate per
task/model) and spot-checking `predictions/` output after a run.

## Architecture

**`config.py` is the single source of truth** — every script imports paths, year ranges (`TRAIN_START`/`TRAIN_END`/
`TEST_START`/`TEST_END`/`CURRENT_YEAR`), the `TASKS` dict, and `CHAMPION_MODEL` from it. Never hardcode a path
elsewhere. `config_local.py` (gitignored) layers in local secrets via a trailing `from config_local import *`.

**Two parallel data tracks feed the same feature names from different sources:**
- *Training* (`build_dataset.py` → `merge_fangraphs.py`): Lahman zip (1980–2025) aggregated to one row per
  player-season, joined to FanGraphs historical stats by bbref→IDfg mapping (with a Name+Year fallback pass).
- *Live scoring* (`predict_awards.py: build_features()`): pulls the current season from the MLB Stats API
  (always available, used as the floor) and overlays FanGraphs data with this priority order: manual CSV export
  in `data/raw/fg_exports/` > authenticated FanGraphs actuals (needs `config_local.py` creds) > Steamer
  rest-of-season projections > MLB Stats API alone (WAR defaults to 0 if nothing else is available).

**Award eligibility filters** (`predict_awards.py: main()`), intentionally asymmetric between awards:
- MVP: position players need `AB > 0` **and** to clear the real batting-title qualifying pace — 3.1 PA per team
  game played (`MVP_QUALIFYING_PA_PER_TEAM_GAME`), so an injured hitter falls out of contention as his team
  keeps playing without him. Two-way players with exceptional total WAR (`bat_WAR_fg >= 4.0`, the "Ohtani tier") bypass the
  PA gate entirely, since that path isn't about hitting volume.
- Cy Young: only requires `IPouts > 0`, deliberately lenient — several relief-pitcher CY winners (Eckersley '92,
  Gagné '03, M. Davis '89) never met the equivalent 1 IP/team-game qualifying pace, so CY must not gate on it.

**Champion model selection**: `train_model.py` trains both LogisticRegression and RandomForest per task;
`config.CHAMPION_MODEL` hardcodes which one `predict_awards.py` actually loads at inference time (LogReg for
MVP tasks, RandomForest for CY tasks) based on `Top1HitRate`/`Recall@5` in `metrics_summary.csv`. Check that
dict, not just the per-task `.joblib` files, when reasoning about what the live pipeline actually uses.

**Feature-collinearity trap** — when adding features to `train_model.py`'s training set, avoid near-duplicate or
linearly-dependent batting columns. The dataset has both Lahman-derived (`OPS`/`OBP`/`SLG`) and FanGraphs-derived
(`bat_OPS`/`bat_OBP`/`bat_SLG`) versions of the same stat, and `OPS = OBP + SLG` is an exact identity. Including
collinear features together lets the regularized LogReg split weight across them arbitrarily, sometimes flipping
a feature's coefficient sign — e.g. `bat_SLG` and `bat_OPS` showed up *negatively* weighted despite being clearly
positive indicators, until removed from `EXCLUDE_ALWAYS`. `bat_WAR_fg`/`bat_wRC_plus` already encode that signal
in weighted form, so prefer those over adding the raw rate-stat components back in.

**Data regeneration note**: `data/processed/player_season_full.csv` and `data/processed/player_season_features_fg.csv`
are gitignored and must be built locally from a Lahman zip you supply (`data/raw/lahman_*.zip`, also gitignored).
The Lahman-derived `OBP`/`SLG`/`OPS` computation in `build_dataset.py` and the equivalent columns in
`merge_fangraphs.py`'s `keep_core` list are currently commented out — a freshly regenerated dataset will *not*
contain those raw columns, only the `bat_*` FanGraphs versions, by design (see collinearity note above).

**Models are tracked in git despite `models/` being gitignored** (the specific `.joblib`/`.csv` files were
force-added previously). Use `git add -u` (or `git add -f` on individual paths) when committing retrained
models, not a bare `git add models/` — the gitignore rule will reject new untracked paths under it.

**Soft external dependency**: `predict_awards.py` optionally imports `sync_espn_headshots` from the sibling repo
`../stinger-assets/` to register newly-seen players for headshot syncing. It's wrapped in try/except, so it's a
no-op (not an error) if that sibling repo isn't on the Python path.

**Weekly automation** (`.github/workflows/weekly_predict.yml`): runs every Monday 13:00 UTC during the season,
calls `pull_fg_current.py --force` then `predict_awards.py`, and commits updated `predictions/` CSVs **and the
refreshed `data/raw/fg_exports/` CSVs** back to the repo as `github-actions[bot]` with `[skip ci]`. The FG pull
step is `continue-on-error` so a FanGraphs outage still yields predictions from the committed CSVs, but a final
step then fails the run so it can't pass unnoticed. `run_and_push.ps1` does the same refresh locally and is kept
as a manual backup only — it is not scheduled.

**FanGraphs 403s are self-inflicted — do not "fix" them by adding browser headers.** This repo spent a long time
believing Cloudflare blocked GitHub runner IPs and that the leaderboard endpoint had a 30-row cap requiring a
30-team crawl. Both were wrong:
- The 30-row cap was a paging bug. The API ignores `page="1_100"`; it takes `pageitems` + `pagenum`. One request
  returns the full leaderboard (~1350 batters / ~780 pitchers at `qual=0`).
- The 403s came from this repo sending a hardcoded `Chrome/124` `User-Agent`. A Python client claiming to be a
  browser is a *stronger* bot signal than one that doesn't. Plain `requests` with **default headers** returns 200 —
  no login, no cookies, no TLS impersonation, from any IP.

`pull_fg_current.py --probe` prints the full client/header matrix and shows this directly. `_FG_HEADERS` is kept
in that file **only** as a documented example of what not to send. `curl_cffi` is an optional fallback in
`make_session()`, not a requirement.

**Pitcher `K%` is a rate, not `K/9`.** `merge_fangraphs.py` trains `pit_Kpct` on FanGraphs `K%` (~0.21). The
manual dashboard export carries only `K/9` (~10.5), and `load_fg_exports()` picks whichever of `K%`/`K/9` appears
*first* in the CSV — so an export lacking `K%` silently fed a feature ~37x its trained scale into the Cy Young
models. `pull_fg_current.py` emits `K%`/`BB%` ahead of `K/9`/`BB/9` for this reason; keep that column order.

**Prediction output convention** in `predictions/<year>/`: `_latest.csv` per league/award (for Google Sheets
import), a timestamped archive copy (only the 4 most recent are kept, older ones auto-pruned), `top5_flat_*.csv`
(flattened layout for Canva graphics), and a hidden `.prev_top10_{mvp,cy}.csv` snapshot used solely to compute
the `↑`/`↓`/`NEW` rank-movement column on the next run — don't delete it between runs or movement tracking resets.
