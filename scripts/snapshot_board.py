"""Keep one copy of each board per week, for good, and mark a season's final board.

The same file lives in every awards-predictor repo (NFL, NBA, NHL, MLB); keep the copies identical.

Each run copies predictions/<season>/board_<award>_latest.csv to board_<award>_wkNN.csv, with NN
taken from the board's `week` or `through_week` column. MLB writes only top10_<award>_latest.csv
and has no week column, so its copies use the run date instead: top10_<award>_YYYY-MM-DD.csv.
A rerun in the same week (or day) overwrites that copy. These copies are never pruned;
ARCHIVES_TO_KEEP only prunes the timestamped run files.

--final writes <stem>_final.csv instead of the weekly copy. Run it once, after the season's last
regular-season board; the website's archive pages read the _final files.

Usage:
  python scripts/snapshot_board.py                 # weekly copy of the newest season's boards
  python scripts/snapshot_board.py --final         # mark the newest boards as the season's finals
  python scripts/snapshot_board.py --season 2026 --dry-run
"""
import argparse
import csv
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

PREDICTIONS = Path(__file__).resolve().parent.parent / "predictions"
LATEST = re.compile(r"^(board|top10)_(.+)_latest\.csv$")
WEEK_COLUMNS = ("week", "through_week")


def season_dir(season):
    if season:
        return PREDICTIONS / str(season)
    years = [p for p in PREDICTIONS.iterdir() if p.is_dir() and p.name.isdigit()]
    if not years:
        raise SystemExit(f"No season folders in {PREDICTIONS}")
    return max(years, key=lambda p: int(p.name))


def boards(folder):
    """board_*_latest.csv files, plus top10_*_latest.csv for awards that have no board (MLB)."""
    found = {}
    for path in sorted(folder.glob("*_latest.csv")):
        m = LATEST.match(path.name)
        if not m:
            continue
        kind, award = m.groups()
        if kind == "board" or award not in found:
            found[award] = path
    return list(found.values())


def label(path):
    with path.open(newline="", encoding="utf-8") as f:
        row = next(csv.DictReader(f), None)
    for col in WEEK_COLUMNS:
        value = (row or {}).get(col, "").strip()
        if value:
            return f"wk{int(float(value)):02d}"
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--season", help="season folder (default: the newest)")
    ap.add_argument("--final", action="store_true", help="write <stem>_final.csv instead of the weekly copy")
    ap.add_argument("--dry-run", action="store_true", help="print what would be written")
    args = ap.parse_args()

    folder = season_dir(args.season)
    paths = boards(folder)
    if not paths:
        raise SystemExit(f"No *_latest.csv boards in {folder}")
    for path in paths:
        stem = path.name[: -len("_latest.csv")]
        name = f"{stem}_final.csv" if args.final else f"{stem}_{label(path)}.csv"
        print(f"{path.relative_to(PREDICTIONS.parent)} -> {name}")
        if not args.dry_run:
            shutil.copyfile(path, folder / name)


if __name__ == "__main__":
    main()
