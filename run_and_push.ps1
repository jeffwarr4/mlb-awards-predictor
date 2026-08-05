<#
.SYNOPSIS
    Refresh FanGraphs current-season exports and push them to GitHub.

.DESCRIPTION
    Intended for Windows Task Scheduler. Runs from Jeff's home IP, which
    FanGraphs/Cloudflare does not block (GitHub Actions runner IPs are).
    The weekly GitHub Actions workflow uses the committed fg_exports CSVs
    as its fallback data source, so this script's whole job is to keep
    those two files fresh in the repo.

    Deliberately stages ONLY the two FanGraphs CSVs. The working tree
    routinely carries unrelated modified files (predictions, graphics,
    chip_data); a bare `git add -A` here would sweep those into an
    unrelated commit.

.PARAMETER RunPredict
    Also run predict_awards.py after refreshing, and commit predictions/.
    Off by default — the GitHub Actions workflow normally does the scoring.

.PARAMETER SkipPush
    Do everything except `git push`. Useful for a dry run.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File "C:\Users\jeffw\OneDrive\DevProj\mlb-awards-predictor\run_and_push.ps1"

.NOTES
    MANUAL BACKUP — not scheduled.

    The weekly GitHub Actions workflow now pulls FanGraphs itself and commits
    the refreshed fg_exports CSVs, so this script is not part of the normal
    weekly path. Keep it for one-off refreshes: if a run fails and you need
    fresh CSVs in the repo without waiting for the next scheduled run, this
    does it in one command from a machine with git credentials.

    If you ever do want it scheduled:
      Program:   powershell.exe
      Arguments: -ExecutionPolicy Bypass -NoProfile -File "C:\Users\jeffw\OneDrive\DevProj\mlb-awards-predictor\run_and_push.ps1"
      Start in:  C:\Users\jeffw\OneDrive\DevProj\mlb-awards-predictor
      Trigger:   Weekly, Monday ~08:00 local (ahead of the 13:00 UTC Actions run)
#>

[CmdletBinding()]
param(
    [switch]$RunPredict,
    [switch]$SkipPush
)

$ErrorActionPreference = 'Stop'

$RepoDir = 'C:\Users\jeffw\OneDrive\DevProj\mlb-awards-predictor'
$Python  = 'C:\DevVenvs2\mlb-awards-venv\Scripts\python.exe'
$LogDir  = Join-Path $RepoDir 'logs'
$LogFile = Join-Path $LogDir ("run_and_push_{0}.log" -f (Get-Date -Format 'yyyyMMdd_HHmmss'))

# Scripts print emoji and up/down arrows that blow up on the default
# Windows cp1252 console with UnicodeEncodeError.
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONPATH       = $RepoDir

function Write-Log {
    param([string]$Message, [string]$Level = 'INFO')
    $line = "[{0}] [{1}] {2}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $Level, $Message
    Write-Host $line
    Add-Content -Path $LogFile -Value $line -Encoding utf8
}

if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir -Force | Out-Null }

# Keep only the 20 most recent logs
Get-ChildItem $LogDir -Filter 'run_and_push_*.log' -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending | Select-Object -Skip 20 |
    Remove-Item -Force -ErrorAction SilentlyContinue

try {
    Set-Location $RepoDir
    Write-Log "=== FanGraphs refresh started ==="

    if (-not (Test-Path $Python)) { throw "Python interpreter not found: $Python" }

    # ── 1. Refresh the FanGraphs exports ──────────────────────────
    Write-Log "Running pull_fg_current.py --force"
    & $Python (Join-Path $RepoDir 'src\pull_fg_current.py') --force 2>&1 |
        ForEach-Object { Write-Log $_ 'FG' }

    if ($LASTEXITCODE -ne 0) {
        # pull_fg_current.py returns non-zero when a pull was rejected as
        # short/broken. It leaves the previous CSVs untouched in that case,
        # so bail out rather than committing anything.
        throw "pull_fg_current.py exited with code $LASTEXITCODE - existing CSVs left in place, nothing committed."
    }
    Write-Log "FanGraphs pull OK"

    # ── 2. Sanity-check the two files before committing ───────────
    $checker = @'
import sys, pandas as pd
from pathlib import Path
ok = True
for name, floor, cols in [("bat", 200, ["WAR","wRC+","MLBAMID"]),
                          ("pit", 150, ["WAR","FIP","K%","MLBAMID"])]:
    p = Path("data/raw/fg_exports") / f"fg_{name}_2026.csv"
    df = pd.read_csv(p)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        print(f"  {p.name}: MISSING COLUMNS {missing}"); ok = False
    elif len(df) < floor:
        print(f"  {p.name}: only {len(df)} rows (floor {floor})"); ok = False
    elif df["MLBAMID"].notna().sum() < len(df) * 0.5:
        print(f"  {p.name}: too many rows without MLBAMID"); ok = False
    else:
        print(f"  {p.name}: {len(df)} rows x {len(df.columns)} cols OK")
sys.exit(0 if ok else 1)
'@
    Write-Log "Validating exports"
    $checker | & $Python - 2>&1 | ForEach-Object { Write-Log $_ 'CHK' }
    if ($LASTEXITCODE -ne 0) { throw "Export validation failed - nothing committed." }

    # ── 3. Optionally re-score ────────────────────────────────────
    if ($RunPredict) {
        Write-Log "Running predict_awards.py"
        & $Python (Join-Path $RepoDir 'src\predict_awards.py') 2>&1 |
            ForEach-Object { Write-Log $_ 'PRD' }
        if ($LASTEXITCODE -ne 0) { throw "predict_awards.py exited with code $LASTEXITCODE" }
    }

    # ── 4. Commit just the FanGraphs exports ──────────────────────
    $paths = @(
        'data/raw/fg_exports/fg_bat_2026.csv',
        'data/raw/fg_exports/fg_pit_2026.csv'
    )
    if ($RunPredict) { $paths += 'predictions/' }

    git add -- $paths
    if ($LASTEXITCODE -ne 0) { throw "git add failed" }

    git diff --staged --quiet
    if ($LASTEXITCODE -eq 0) {
        Write-Log "No changes to commit - FanGraphs data is unchanged since last run."
    }
    else {
        $stamp = (Get-Date -Format 'yyyy-MM-dd HH:mm')
        git commit -m "chore: refresh FG exports with full 2026 season data ($stamp) [skip ci]"
        if ($LASTEXITCODE -ne 0) { throw "git commit failed" }
        Write-Log "Committed."

        if ($SkipPush) {
            Write-Log "SkipPush set - commit left local." 'WARN'
        }
        else {
            git push
            if ($LASTEXITCODE -ne 0) { throw "git push failed" }
            Write-Log "Pushed to origin."
        }
    }

    Write-Log "=== Completed successfully ==="
    exit 0
}
catch {
    Write-Log $_.Exception.Message 'ERROR'
    Write-Log "=== FAILED ===" 'ERROR'
    exit 1
}
