# src/render_graphics.py
# ---------------------------------------------------------------
# Generates award-tracker PNG graphics by compositing player
# headshots, team logos, chip labels, and stat blocks onto the
# pre-made background templates in the repo root.
#
# Usage (standalone):
#   python src/render_graphics.py                 # all 4 graphics
#   python src/render_graphics.py --award mvp     # MVP AL + NL
#   python src/render_graphics.py --league AL     # all awards, AL only
#
# Called from predict_awards.py:
#   from render_graphics import render_all_graphics
#   render_all_graphics(t10_mvp, t10_cy, outdir, year)
#
# Outputs: predictions/{year}/graphics/{award}_{league}_{ts}.png
# ---------------------------------------------------------------

import os, sys, argparse, textwrap
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime
from typing import List, Optional, Dict, Tuple

# ── Asset paths ───────────────────────────────────────────────────
REPO_ROOT      = Path(__file__).parent.parent
# Override with STINGER_ASSETS_DIR env var so CI (GitHub Actions) can point
# at a checked-out copy of the sibling repo instead of the local Windows path.
STINGER_ASSETS = Path(os.environ.get(
    "STINGER_ASSETS_DIR",
    r"C:\Users\jeffw\OneDrive\DevProj\stinger-assets",
))
HEADSHOTS_DIR  = STINGER_ASSETS / "mlb" / "Headshots"
LOGOS_DIR      = STINGER_ASSETS / "mlb" / "logos"

BACKGROUND = {
    "mvp": REPO_ROOT / "MVP Awards.png",
    "cy":  REPO_ROOT / "Cy Young Awards.png",
}

# ── Fonts ─────────────────────────────────────────────────────────
# CI font strategy: free OFL alternatives (Oswald / Barlow) downloaded to
# assets/fonts/ by the workflow; Windows proprietary fonts used locally.
_REPO_FONTS = REPO_ROOT / "assets" / "fonts"
_F_WIN      = Path("C:/Windows/Fonts")

def _find_font(*candidates: str) -> str:
    """Return the first candidate path that actually exists on this machine."""
    for p in candidates:
        if Path(p).exists():
            return str(p)
    return candidates[-1]  # not found → _font() try/except catches the error

FONT_PATH = {
    "agency_bold":  _find_font(
        str(_REPO_FONTS / "Oswald"                / "static" / "Oswald-Bold.ttf"),
        str(_F_WIN     / "AGENCYB.TTF"),
    ),
    "fg_med_cond":  _find_font(
        str(_REPO_FONTS / "Barlow_Condensed"      / "BarlowCondensed-Medium.ttf"),
        str(_F_WIN     / "FRAMDCN.TTF"),
    ),
    "fg_demi":      _find_font(
        str(_REPO_FONTS / "Barlow_Semi_Condensed" / "BarlowSemiCondensed-SemiBold.ttf"),
        str(_F_WIN     / "FRADM.TTF"),
    ),
    "fg_demi_cond": _find_font(
        str(_REPO_FONTS / "Barlow_Condensed"      / "BarlowCondensed-SemiBold.ttf"),
        str(_F_WIN     / "FRADMCN.TTF"),
    ),
}

# ── Brand colors ──────────────────────────────────────────────────
GOLD  = (255, 210, 0)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# ── Canvas ────────────────────────────────────────────────────────
IMG_W = 1080
IMG_H = 1350
CONTENT_TOP = 220          # below the STINGER header divider line

# ── Stat config ───────────────────────────────────────────────────
MVP_STAT_COLS = ["bat_WAR_fg","bat_wRC_plus","bat_AVG","bat_OPS","HR","RBI","PA","WinPct"]
CY_STAT_COLS  = ["pit_WAR_fg","pit_FIP","pit_Kpct","SO_pit","IP","WinPct"]

STAT_LABELS: Dict[str, str] = {
    "bat_WAR_fg":   "fWAR",   "bat_wRC_plus": "wRC+",
    "bat_AVG":      "AVG",    "bat_OPS":      "OPS",
    "HR":           "HR",     "RBI":          "RBI",
    "PA":           "PA",     "WinPct":       "Team W%",
    "pit_WAR_fg":   "fWAR",   "pit_FIP":      "FIP",
    "pit_Kpct":     "K%",     "SO_pit":       "K's",
    "IP":           "IP",
}
LOWER_IS_BETTER = {"pit_FIP", "pit_BBpct", "pit_ERA_minus", "pit_xFIP"}

# All MLB headshots are 600×436 (1.38:1 landscape). We maintain this ratio.
_HS_ASPECT = 600 / 436


# ─────────────────────────────────────────────────────────────────
# STAT SELECTOR — decoupled, reusable for NBA / NHL
# ─────────────────────────────────────────────────────────────────
def select_top_stats(
    player_row: pd.Series,
    pool_df: pd.DataFrame,
    stat_cols: List[str],
    n: int = 3,
    lower_is_better: Optional[set] = None,
) -> List[str]:
    """
    Return the top-n stat column names for this player, ranked by
    percentile position within pool_df.  Stats in lower_is_better
    (e.g. FIP) are inverted so a lower raw value = higher standout rank.
    Pass any pool_df + stat_cols to reuse across sports.
    """
    lower_is_better = lower_is_better or set()
    pcts: Dict[str, float] = {}
    for col in stat_cols:
        if col not in pool_df.columns:
            continue
        pval = pd.to_numeric(player_row.get(col), errors="coerce")
        if pd.isna(pval):
            continue
        vals = pd.to_numeric(pool_df[col], errors="coerce").dropna()
        if len(vals) < 2:
            continue
        pct = float((vals < pval).sum()) / len(vals) * 100.0
        pcts[col] = (100.0 - pct) if col in lower_is_better else pct
    return [c for c, _ in sorted(pcts.items(), key=lambda x: x[1], reverse=True)[:n]]


def format_stat(col: str, value) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if col in ("bat_AVG", "bat_OPS", "bat_OBP", "bat_SLG"):
        # OPS can legitimately exceed 1.000; avoid ".1066" — show "1.066" instead
        s = f"{v:.3f}"
        return s if v >= 1.0 else s.lstrip("0") or "0"
    if col == "WinPct":
        return f"{v:.3f}"
    if col in ("bat_WAR_fg", "pit_WAR_fg"):
        return f"{v:.1f}"
    if col == "pit_FIP":
        return f"{v:.2f}"
    if col in ("pit_Kpct", "pit_BBpct"):
        pct = v * 100 if v <= 1.0 else v
        return f"{pct:.1f}%"
    if col == "IP":
        return f"{v:.1f}"
    if col in ("HR", "RBI", "PA", "SO_pit"):
        return str(int(round(v)))
    return f"{v:.1f}"


# ─────────────────────────────────────────────────────────────────
# IMAGE HELPERS
# ─────────────────────────────────────────────────────────────────
def _font(key: str, size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(FONT_PATH[key], size)
    except Exception:
        return ImageFont.load_default()


def _paste(canvas: Image.Image, layer: Image.Image, xy: Tuple[int, int]):
    canvas.paste(layer, xy, layer) if layer.mode == "RGBA" else canvas.paste(layer, xy)


def load_headshot(player_key: str, target_w: int) -> Optional[Image.Image]:
    """
    Load headshot at natural 600×436 (1.38:1) aspect ratio, scaled so
    width == target_w.  Returns None on miss (caller draws placeholder).
    All ESPN MLB headshots are this fixed size so we hardcode the ratio.
    """
    path = HEADSHOTS_DIR / f"{player_key}.png"
    if not path.exists():
        return None
    try:
        img = Image.open(path).convert("RGBA")
        target_h = max(1, int(round(target_w / _HS_ASPECT)))
        return img.resize((target_w, target_h), Image.LANCZOS)
    except Exception:
        return None


def load_headshot_placeholder(target_w: int) -> Image.Image:
    """Grey silhouette at the same natural aspect ratio."""
    target_h = max(1, int(round(target_w / _HS_ASPECT)))
    ph = Image.new("RGBA", (target_w, target_h), (0, 0, 0, 0))
    ImageDraw.Draw(ph).rectangle((0, 0, target_w - 1, target_h - 1),
                                  fill=(70, 70, 70, 180))
    return ph


def load_logo(team_abbr: str, size: int) -> Image.Image:
    abbr = team_abbr.lower()
    for candidate in [abbr, abbr[:2], abbr[:3]]:
        p = LOGOS_DIR / f"{candidate}.png"
        if p.exists():
            img = Image.open(p).convert("RGBA")
            img.thumbnail((size, size), Image.LANCZOS)
            return img
    return Image.new("RGBA", (size, size), (0, 0, 0, 0))


def overlay_rect(canvas: Image.Image, box: Tuple, radius: int,
                 fill_rgba: Tuple, outline_rgba: Optional[Tuple] = None,
                 outline_w: int = 2):
    layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    ImageDraw.Draw(layer).rounded_rectangle(
        box, radius=radius, fill=fill_rgba,
        outline=outline_rgba, width=outline_w)
    canvas.alpha_composite(layer)


def draw_text_centered(draw, cx, y, text, font, fill,
                        stroke_fill=(0,0,0), stroke_w=2):
    bb = draw.textbbox((0, 0), text, font=font)
    tw, th = bb[2]-bb[0], bb[3]-bb[1]
    draw.text((cx - tw//2, y), text, font=font,
              fill=stroke_fill, stroke_width=stroke_w, stroke_fill=stroke_fill)
    draw.text((cx - tw//2, y), text, font=font, fill=fill)
    return th


def text_w(draw, text, font) -> int:
    bb = draw.textbbox((0, 0), text, font=font)
    return bb[2] - bb[0]


def fit_text(draw, text, font, max_w) -> str:
    if text_w(draw, text, font) <= max_w:
        return text
    while len(text) > 4:
        text = text[:-1]
        if text_w(draw, text + "…", font) <= max_w:
            return text + "…"
    return text


# ─────────────────────────────────────────────────────────────────
# LAYOUT CONSTANTS
# ─────────────────────────────────────────────────────────────────
#
#   1080 × 1350 canvas
#
#   y=220 ── league badge
#   y=255 ── top-3 cards begin
#   y=870 ── top-3 cards end
#   y=880 ── compact row 4
#   y=1070 ── compact row 5
#   y=1260 ── compact row 5 end
#
#   Each card column = 360 px wide
#   Top-3 card anatomy (no dark box — open over the background):
#     ● large team logo (260 px)  — drawn FIRST (behind)
#     ● headshot (310 px wide × 225 px tall, natural 1.38:1)  — on top of logo
#     ● gold rank badge (48 px Ø)  — bottom-left of image zone
#     ● solid black name banner  — spans card width
#     ● chip pills (optional)
#     ● 2 × 2 stat grid (4 standout stats)

CARD_TOP   = 255
CARD_BOT   = 872
CARD_COL_W = IMG_W // 3        # 360 px
CARD_CX    = [180, 540, 900]

# headshot scaled to this width; height follows 600:436 ratio → ~225 px
_HS_W  = 312
_HS_H  = int(round(_HS_W / _HS_ASPECT))   # 227 px

_LOGO_SZ   = 200      # smaller logo so the crown above the head reads clearly
_BADGE_R   = 24       # radius of rank badge circle
_NAME_H    = 50
_CHIP_H    = 32
_CELL_H    = 60       # tighter stat cells to reclaim vertical space
_CELL_COLS = 2

# compact rows — side-by-side (rank 4 left, rank 5 right)
_ROW_TOP   = 882
_ROW_H     = 210
_ROW_PAD   = 16       # outer left/right margin
_ROW_GAP   = 16       # gap between the two panels
_ROW_W     = (IMG_W - 2 * _ROW_PAD - _ROW_GAP) // 2   # 516 px each
_ROW_LEFTS = [_ROW_PAD, _ROW_PAD + _ROW_W + _ROW_GAP]
_ROW_HS_W  = 164                              # compact headshot width


# ─────────────────────────────────────────────────────────────────
# TOP-3 CARD
# ─────────────────────────────────────────────────────────────────
def draw_card(canvas: Image.Image, player: pd.Series, pool_df: pd.DataFrame,
              stat_cols: List[str], chip_row: Optional[pd.Series], rank: int):
    cx    = CARD_CX[rank - 1]
    left  = cx - CARD_COL_W // 2
    right = cx + CARD_COL_W // 2

    # ── 2. Player headshot at natural 1.38:1 aspect ──────────────────────────
    pkey     = str(player.get("player_key", ""))
    headshot = load_headshot(pkey, _HS_W) or load_headshot_placeholder(_HS_W)
    hs_x     = cx - headshot.width // 2
    # Shifted down so the logo (drawn behind) can crown the head area.
    # ESPN headshots are 600×436; the player's head sits ~80px into the image,
    # so logo_y = hs_y - 50 places the logo center at ~hs_y+80 (head level).
    # Pushed down so the logo (drawn behind) has room to crown the head from above.
    # ESPN headshots are 600×436; head sits ~55 px into the rescaled 312×227 image.
    # logo_y = CARD_TOP+8 → logo center ≈ y=365, head center ≈ hs_y+55 = 410.
    # That puts the logo ~45 px above the head: visible crown even at 200 px logo size.
    hs_y     = CARD_TOP + 100
    hs_bot   = hs_y + headshot.height

    # ── 1. Team logo — drawn first (behind) so it crowns the head ────────────
    logo   = load_logo(str(player.get("Team", "")), _LOGO_SZ)
    logo_x = cx - logo.width // 2
    logo_y = CARD_TOP + 8            # logo near card top → center well above head
    _paste(canvas, logo, (logo_x, logo_y))
    _paste(canvas, headshot, (hs_x, hs_y))

    # ── 3. Gold rank badge — bottom-left of image zone ────────────────────────
    badge_cx = left + _BADGE_R + 10
    badge_cy = hs_bot - _BADGE_R
    bl = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    bd = ImageDraw.Draw(bl)
    bd.ellipse((badge_cx-_BADGE_R, badge_cy-_BADGE_R,
                badge_cx+_BADGE_R, badge_cy+_BADGE_R),
               fill=(*GOLD, 255), outline=(*BLACK, 255), width=2)
    canvas.alpha_composite(bl)
    draw = ImageDraw.Draw(canvas)
    draw_text_centered(draw, badge_cx, badge_cy - _BADGE_R + 4,
                       f"#{rank}", _font("agency_bold", 26), BLACK, stroke_w=0)

    # ── 4. Solid black name banner — spans full card column ───────────────────
    name_y = hs_bot + 5
    overlay_rect(canvas, (left, name_y, right, name_y + _NAME_H),
                 radius=0, fill_rgba=(0, 0, 0, 235))
    draw = ImageDraw.Draw(canvas)
    name_font = _font("agency_bold", 28)
    name_str  = fit_text(draw, str(player.get("Name", "")).upper(),
                         name_font, CARD_COL_W - 20)
    draw_text_centered(draw, cx, name_y + (_NAME_H - 28) // 2,
                       name_str, name_font, WHITE, stroke_w=0)

    y = name_y + _NAME_H + 8

    # ── 5. Chip pills (only when chip data is available) ──────────────────────
    chip_str = chip_row.get("Chip") if chip_row is not None else None
    if chip_str and isinstance(chip_str, str) and "|" in chip_str:
        chips    = [c.strip() for c in chip_str.split("|")]
        max_pill_w = CARD_COL_W - 16
        # Auto-shrink font so all 3 pills always fit within the card column
        for chip_sz in range(17, 10, -1):
            chip_font = _font("fg_med_cond", chip_sz)
            pill_pad  = max(8, chip_sz - 4)
            pill_ws   = [text_w(draw, c, chip_font) + pill_pad * 2 for c in chips]
            total_pill = sum(pill_ws) + 6 * (len(chips) - 1)
            if total_pill <= max_pill_w:
                break
        px = cx - total_pill // 2
        cl = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        cd = ImageDraw.Draw(cl)
        for chip, pw in zip(chips, pill_ws):
            cd.rounded_rectangle((px, y, px + pw, y + _CHIP_H),
                                  radius=14, fill=(*GOLD, 235))
            bb = cd.textbbox((0, 0), chip, font=chip_font)
            cd.text((px + pill_pad, y + (_CHIP_H - (bb[3]-bb[1])) // 2 - bb[1]),
                    chip, font=chip_font, fill=BLACK)
            px += pw + 6
        canvas.alpha_composite(cl)
        draw = ImageDraw.Draw(canvas)
        y += _CHIP_H + 8

        # Why sentence — grey translucent box for readability over the open bg
        why = chip_row.get("Why") if chip_row is not None else None
        if why and isinstance(why, str):
            wf    = _font("fg_med_cond", 18)
            lines = textwrap.wrap(why, width=max(1, (CARD_COL_W - 16) // 7))[:2]
            if lines:
                # measure tallest line to size the box
                line_h = max((draw.textbbox((0,0), l, font=wf)[3]
                               - draw.textbbox((0,0), l, font=wf)[1]) for l in lines)
                box_h  = len(lines) * (line_h + 4) + 8
                overlay_rect(canvas, (left + 4, y, right - 4, y + box_h),
                             radius=6, fill_rgba=(20, 20, 20, 175))
                draw = ImageDraw.Draw(canvas)
                ty = y + 4
                for line in lines:
                    lh = draw.textbbox((0,0), line, font=wf)[3] \
                       - draw.textbbox((0,0), line, font=wf)[1]
                    draw_text_centered(draw, cx, ty, line, wf,
                                       (220, 220, 220), stroke_fill=BLACK, stroke_w=0)
                    ty += lh + 4
                y += box_h + 4

    # ── 6. 2 × 2 stat grid (top-4 standout stats by percentile) ──────────────
    top_stats = select_top_stats(player, pool_df, stat_cols, n=4,
                                  lower_is_better=LOWER_IS_BETTER)
    if top_stats:
        cell_w   = CARD_COL_W // _CELL_COLS
        lbl_font = _font("fg_med_cond", 14)
        val_font = _font("fg_demi",     26)
        for i, col in enumerate(top_stats):
            row_i, col_i = divmod(i, _CELL_COLS)
            cx_cell = left + col_i * cell_w + cell_w // 2
            cy_top  = y + row_i * _CELL_H
            overlay_rect(canvas,
                         (left + col_i*cell_w + 3, cy_top + 3,
                          left + (col_i+1)*cell_w - 3, cy_top + _CELL_H - 3),
                         radius=6, fill_rgba=(15, 15, 15, 200))
            draw = ImageDraw.Draw(canvas)
            lbl = STAT_LABELS.get(col, col)
            val = format_stat(col, player.get(col, 0))
            draw_text_centered(draw, cx_cell, cy_top + 7,
                               lbl, lbl_font, GOLD, stroke_fill=BLACK, stroke_w=1)
            draw_text_centered(draw, cx_cell, cy_top + 24,
                               val, val_font, WHITE, stroke_fill=BLACK, stroke_w=2)


# ─────────────────────────────────────────────────────────────────
# COMPACT ROW (ranks 4 & 5)
# ─────────────────────────────────────────────────────────────────
def draw_compact_row(canvas: Image.Image, player: pd.Series, pool_df: pd.DataFrame,
                     stat_cols: List[str], rank: int,
                     row_top: int, row_left: int, row_width: int):
    """Draws a compact player panel.  row_left/row_width let two panels sit side-by-side."""
    row_right = row_left + row_width
    row_bot   = row_top + _ROW_H
    row_cy    = (row_top + row_bot) // 2
    IPAD      = 10   # inner horizontal padding

    overlay_rect(canvas,
                 (row_left + 4, row_top + 4, row_right - 4, row_bot - 4),
                 radius=14, fill_rgba=(10, 10, 10, 192))

    # ── Headshot ──────────────────────────────────────────────────
    pkey     = str(player.get("player_key", ""))
    headshot = load_headshot(pkey, _ROW_HS_W) or load_headshot_placeholder(_ROW_HS_W)
    hs_x     = row_left + IPAD + 6
    hs_y     = row_cy - headshot.height // 2
    _paste(canvas, headshot, (hs_x, hs_y))

    # ── Team logo (top-right corner of headshot) ───────────────────
    logo    = load_logo(str(player.get("Team", "")), 40)
    logo_lx = hs_x + headshot.width - logo.width + 6
    logo_ty = hs_y - 6
    _paste(canvas, logo, (logo_lx, logo_ty))

    # ── Rank badge (bottom-left corner of headshot) ────────────────
    br  = 18
    bcx = hs_x + br - 4
    bcy = hs_y + headshot.height - br + 4
    bl  = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    ImageDraw.Draw(bl).ellipse(
        (bcx-br, bcy-br, bcx+br, bcy+br),
        fill=(*GOLD, 255), outline=(*BLACK, 255), width=2)
    canvas.alpha_composite(bl)
    draw = ImageDraw.Draw(canvas)
    draw_text_centered(draw, bcx, bcy-br+4,
                       f"#{rank}", _font("agency_bold", 22), BLACK, stroke_w=0)

    # ── Name + stats ───────────────────────────────────────────────
    text_x    = hs_x + headshot.width + 10
    max_text_w = row_right - text_x - IPAD - 4

    name_font = _font("agency_bold", 30)
    name_str  = fit_text(draw, str(player.get("Name", "")).upper(), name_font, max_text_w)
    draw.text((text_x, row_cy - 36), name_str, font=name_font, fill=WHITE,
              stroke_width=1, stroke_fill=BLACK)

    top_stats  = select_top_stats(player, pool_df, stat_cols, n=2,
                                   lower_is_better=LOWER_IS_BETTER)
    lbl_font   = _font("fg_med_cond", 15)
    val_font   = _font("fg_demi",     22)
    sx, sy     = text_x, row_cy + 5
    for i, col in enumerate(top_stats):
        lbl = STAT_LABELS.get(col, col) + ":"
        val = format_stat(col, player.get(col, 0))
        draw.text((sx, sy), lbl, font=lbl_font, fill=GOLD)
        sx += text_w(draw, lbl, lbl_font) + 4
        draw.text((sx, sy - 2), val, font=val_font, fill=WHITE,
                  stroke_width=1, stroke_fill=BLACK)
        sx += text_w(draw, val, val_font) + 12
        if i < len(top_stats) - 1:
            draw.text((sx, sy), "·", font=val_font, fill=GOLD)
            sx += text_w(draw, "·", val_font) + 12


# ─────────────────────────────────────────────────────────────────
# LEAGUE BADGE
# ─────────────────────────────────────────────────────────────────
def draw_league_badge(canvas: Image.Image, league: str):
    font  = _font("agency_bold", 36)
    bw, bh = 120, 44
    bx    = IMG_W // 2 - bw // 2
    by    = CONTENT_TOP + 2
    overlay_rect(canvas, (bx, by, bx+bw, by+bh),
                 radius=10, fill_rgba=(*GOLD, 245),
                 outline_rgba=(*BLACK, 255), outline_w=2)
    draw  = ImageDraw.Draw(canvas)
    bb    = draw.textbbox((0, 0), league, font=font)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    # Subtract bb offsets so the glyph is truly centred in the pill
    draw.text((bx + (bw - tw) // 2 - bb[0],
               by + (bh - th) // 2 - bb[1]),
              league, font=font, fill=BLACK)


# ─────────────────────────────────────────────────────────────────
# MAIN RENDER
# ─────────────────────────────────────────────────────────────────
def render_award_graphic(
    df_top10: pd.DataFrame,
    chip_df:  pd.DataFrame,
    award:    str,
    league:   str,
    outdir:   Path,
    year:     int = 2026,
) -> Optional[Path]:
    stat_cols = MVP_STAT_COLS if award == "mvp" else CY_STAT_COLS
    bg_path   = BACKGROUND[award]
    if not bg_path.exists():
        print(f"  Background not found: {bg_path}")
        return None

    canvas = Image.open(bg_path).convert("RGBA")

    df = (df_top10[df_top10["lgID"] == league]
          .sort_values("rank").reset_index(drop=True))
    if df.empty:
        print(f"  No {league} {award.upper()} data — skipping")
        return None

    chip_map: Dict[str, Dict] = {}
    if chip_df is not None and not chip_df.empty and "player_key" in chip_df.columns:
        chip_map = chip_df.set_index("player_key").to_dict(orient="index")

    pool_df = df

    draw_league_badge(canvas, league)

    for rank in (1, 2, 3):
        sub = df[df["rank"] == rank]
        if sub.empty:
            continue
        player   = sub.iloc[0]
        pkey     = str(player.get("player_key", ""))
        chip_row = pd.Series(chip_map[pkey]) if pkey in chip_map else None
        draw_card(canvas, player, pool_df, stat_cols, chip_row, rank)

    for idx, rank in enumerate((4, 5)):
        sub = df[df["rank"] == rank]
        if sub.empty:
            continue
        draw_compact_row(canvas, sub.iloc[0], pool_df, stat_cols,
                         rank, _ROW_TOP, _ROW_LEFTS[idx], _ROW_W)

    final = Image.new("RGB", (IMG_W, IMG_H), BLACK)
    final.paste(canvas, mask=canvas.split()[3])

    outdir.mkdir(parents=True, exist_ok=True)
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = outdir / f"{award}_{league.lower()}_{ts}.png"
    final.save(out_path, "PNG")
    print(f"  Saved → {out_path}")
    return out_path


# ─────────────────────────────────────────────────────────────────
# PIPELINE INTEGRATION
# ─────────────────────────────────────────────────────────────────
def render_all_graphics(
    t10_mvp: pd.DataFrame,
    t10_cy:  pd.DataFrame,
    outdir:  Path,
    year:    int = 2026,
) -> dict:
    chip_path = REPO_ROOT / "predictions" / str(year) / "chip_data" / "chip_data_latest.csv"
    chip_df   = pd.read_csv(chip_path) if chip_path.exists() else pd.DataFrame()
    gfx_dir   = outdir / "graphics"
    results   = {}
    for award, df in [("mvp", t10_mvp), ("cy", t10_cy)]:
        for lg in ("AL", "NL"):
            path = render_award_graphic(df, chip_df, award, lg, gfx_dir, year)
            if path:
                results[f"{lg}_{award}_graphic"] = str(path)
    return results


# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--award",  choices=["mvp","cy","both"], default="both")
    parser.add_argument("--league", choices=["AL","NL","both"],  default="both")
    parser.add_argument("--year",   type=int, default=2026)
    args    = parser.parse_args()
    year    = args.year
    pred    = REPO_ROOT / "predictions" / str(year)
    out     = pred / "graphics"
    chip_df = pd.read_csv(pred/"chip_data"/"chip_data_latest.csv") \
              if (pred/"chip_data"/"chip_data_latest.csv").exists() \
              else pd.DataFrame()

    awards  = ["mvp","cy"] if args.award  == "both" else [args.award]
    leagues = ["AL","NL"]  if args.league == "both" else [args.league]
    for award in awards:
        for league in leagues:
            csv = pred / f"top10_{league.lower()}_{award}_latest.csv"
            if not csv.exists():
                print(f"  Missing: {csv}"); continue
            render_award_graphic(pd.read_csv(csv), chip_df, award, league, out, year)


if __name__ == "__main__":
    main()
