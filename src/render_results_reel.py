"""
Awards results reel: the model's final regular-season boards against the actual BBWAA vote, as a silent
1080×1920 MP4 (25 s + 2 s hold) Jeff voices himself. One scene per award (winner, our rank for him, his
first-place votes, CALLED IT / MISSED), then a scorecard (top-5 overlap per award, the biggest miss) and the CTA.

    C:\\DevVenvs2\\mlb-awards-venv\\Scripts\\python.exe src\\render_results_reel.py --season 2026
    ... --frames-only     one PNG per scene (+ *_guides.png with the platform UI zones shaded), no video

Inputs
  predictions/{season}/top10_{award}_final.csv   the model's final boards (scripts/snapshot_board.py --final)
  data/raw/award_votes_{season}.csv              the vote: season, award, place, player_name, team, points,
                                                 first_place_votes, source_url (one row per vote-getter)
  predictions/{season}/reel/results_story.json   hook, and the hand-written script: beats (one per scene),
                                                 instagram_caption, hashtags. Optional; without it the hook is
                                                 "Our model went N for 4" and there is no script.
Outputs (predictions/{season}/reel/)
  results_reel.mp4 (gitignored), results_script.txt, results_meta.json

Ported from nfl-awards-predictor/src/publish/render_reel.py (same look, fonts, safe zones: everything important
inside x 120-960, y 250-1520). The model ranks hitters for MVP, so a pitcher's MVP votes can't be "missed".
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.render_graphics import (  # noqa: E402
    BACKGROUND, BLACK, GOLD, HEADSHOTS_DIR, WHITE, _font, fit_text, load_headshot, load_headshot_placeholder,
    load_logo, text_w,
)

REPO = Path(__file__).parent.parent
W, H, FPS = 1080, 1920, 30
X0, X1, Y0, Y1 = 120, 960, 250, 1520
CX, MAX_W = W // 2, 840
GREEN, RED, GREY = (60, 200, 90), (230, 70, 60), (170, 170, 170)
HOLD_S, XFADE = 2, 0.25

# Announcement order (Cy Young first, MVPs last) so the reel builds to the MVPs
AWARDS = [("al_cy", "AL CY YOUNG", "cy"), ("nl_cy", "NL CY YOUNG", "cy"),
          ("al_mvp", "AL MVP", "mvp"), ("nl_mvp", "NL MVP", "mvp")]
SCENES = [("hook", 0, 3), ("al_cy", 3, 6.5), ("nl_cy", 6.5, 10), ("al_mvp", 10, 13.5), ("nl_mvp", 13.5, 17),
          ("scorecard", 17, 22), ("cta", 22, 25)]
DURATION = SCENES[-1][2] + HOLD_S
WORDS = (55, 75)
FORBIDDEN = re.compile(r"%|\bpercent|\bprobab|\bodds\b|\bchances?\b|\blikel(y|ihood)\b|\bbet(s|ting)?\b|\bwager",
                       re.IGNORECASE)

F = lru_cache(maxsize=None)(_font)


# ─────────────────────────────────────────────────────────────────
# Data: final boards joined to the vote
# ─────────────────────────────────────────────────────────────────
def norm(name: str) -> str:
    n = unicodedata.normalize("NFKD", str(name))
    return re.sub(r"[^a-z]", "", "".join(c for c in n if not unicodedata.combining(c)).lower())


def key_for(name: str, team: str) -> str:
    n = unicodedata.normalize("NFKD", f"{name} {team}")
    n = "".join(c for c in n if not unicodedata.combining(c)).lower().replace("'", "").replace(".", "").replace("-", " ")
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", n)).strip("_")


def award_results(season: int) -> List[dict]:
    votes = pd.read_csv(REPO / "data" / "raw" / f"award_votes_{season}.csv")
    out = []
    for award, label, kind in AWARDS:
        board = pd.read_csv(REPO / "predictions" / str(season) / f"top10_{award}_final.csv")
        board = board.assign(n=board["Name"].map(norm)).sort_values("rank")
        model_rank = dict(zip(board["n"], board["rank"].astype(int)))
        v = votes[votes["award"] == award].assign(n=lambda d: d["player_name"].map(norm)).sort_values("place")
        place = dict(zip(v["n"], v["place"].astype(int)))
        w = v.iloc[0]
        on_board = board[board["n"] == w["n"]]
        top5_actual = v[v["place"] <= 5]
        misses = [{"name": r["Name"], "team": r["Team"], "key": r["player_key"], "model": int(r["rank"]),
                   "place": place.get(r["n"]), "award": label}
                  for _, r in board[board["rank"] <= 5].iterrows()]
        out.append({
            "award": award, "label": label, "kind": kind,
            "winner": w["player_name"], "team": w["team"],
            "key": on_board["player_key"].iloc[0] if len(on_board) else key_for(w["player_name"], w["team"]),
            "first_place": int(w["first_place_votes"]), "ballots": int(v["first_place_votes"].sum()),
            "model_rank": model_rank.get(w["n"]),
            "top5_hits": int(sum(1 for n in top5_actual["n"] if model_rank.get(n, 99) <= 5)),
            "top5_total": len(top5_actual),
            "voters": len(v), "misses": misses, "source": w["source_url"],
        })
    return out


def biggest_miss(results: List[dict]) -> Optional[dict]:
    """The model top-5 player the voters ranked furthest below his model rank (no votes = after the last vote-getter)."""
    best, gap = None, 0
    for r in results:
        for m in r["misses"]:
            actual = m["place"] or r["voters"] + 1
            if actual - m["model"] > gap:
                best, gap = {**m, "actual": m["place"]}, actual - m["model"]
    return best


def ordinal(n: int) -> str:
    return f"{n}{'th' if 11 <= n % 100 <= 13 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


# ─────────────────────────────────────────────────────────────────
# Drawing helpers (as in the NFL weekly reel)
# ─────────────────────────────────────────────────────────────────
def ease(x):
    x = min(1.0, max(0.0, x))
    return 1 - (1 - x) ** 3


def prog(t, start, dur):
    return ease((t - start) / dur)


def put(canvas, img, xy, a=1.0):
    if a <= 0:
        return
    if a < 1:
        img = img.copy()
        img.putalpha(img.getchannel("A").point(lambda v: int(v * a)))
    layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    layer.paste(img, (int(xy[0]), int(xy[1])))
    canvas.alpha_composite(layer)


@lru_cache(maxsize=None)
def text_img(text, font_key, size, fill=WHITE, stroke=0):
    font = F(font_key, size)
    l, t, r, b = ImageDraw.Draw(Image.new("RGBA", (1, 1))).textbbox((0, 0), text, font=font, stroke_width=stroke)
    img = Image.new("RGBA", (max(1, r - l), max(1, b - t)), (0, 0, 0, 0))
    ImageDraw.Draw(img).text((-l, -t), text, font=font, fill=fill, stroke_width=stroke, stroke_fill=BLACK)
    return img


def fit_size(text, font_key, start, max_w, min_size=20):
    d = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    while start > min_size and text_w(d, text, F(font_key, start)) > max_w:
        start -= 2
    return start


def put_c(canvas, cy, text, font_key, size, fill=WHITE, stroke=0, a=1.0, scale=1.0):
    img = text_img(text, font_key, size, fill, stroke)
    if scale != 1:
        img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))), Image.LANCZOS)
    put(canvas, img, (CX - img.width // 2, cy - img.height // 2), a)


def panel(size, radius, fill, outline=None, width=0):
    img = Image.new("RGBA", size, (0, 0, 0, 0))
    ImageDraw.Draw(img).rounded_rectangle((0, 0, size[0] - 1, size[1] - 1), radius=radius, fill=fill,
                                          outline=outline, width=width)
    return img


def pill(text, size=46, fill=GOLD, fg=BLACK):
    lab = text_img(text, "agency_bold", size, fg)
    p = panel((lab.width + 60, lab.height + 34), 16, (*fill, 245), (*BLACK, 255), 3)
    p.alpha_composite(lab, (30, 17))
    return p


def check_mark(size, color):
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    ImageDraw.Draw(img).line([(size * .12, size * .55), (size * .4, size * .82), (size * .9, size * .2)],
                             fill=(*color, 255), width=max(4, size // 7), joint="curve")
    return img


@lru_cache(maxsize=None)
def headshot(key, width):
    return load_headshot(key, width) or load_headshot_placeholder(width)


@lru_cache(maxsize=None)
def headshot_circle(key, d):
    try:
        img = Image.open(HEADSHOTS_DIR / f"{key}.png").convert("RGBA")
    except Exception:
        img = load_headshot_placeholder(600)
    w, h = img.size
    side = int(h * 0.92)
    img = img.crop(((w - side) // 2, 0, (w - side) // 2 + side, side)).resize((d, d), Image.LANCZOS)
    m = Image.new("L", (d, d), 0)
    ImageDraw.Draw(m).ellipse((0, 0, d - 1, d - 1), fill=255)
    ring = Image.new("RGBA", (d + 8, d + 8), (0, 0, 0, 0))
    ImageDraw.Draw(ring).ellipse((0, 0, d + 7, d + 7), fill=(20, 20, 20, 255), outline=(*GOLD, 255), width=4)
    face = Image.new("RGBA", (d, d), (0, 0, 0, 0))
    face.paste(img, (0, 0), Image.composite(img.getchannel("A"), m, m))
    ring.alpha_composite(face, (4, 4))
    return ring


@lru_cache(maxsize=None)
def background(kind):
    """The award graphic's ballpark (below its header band) scaled to 1080×1920 and darkened."""
    src = Image.open(BACKGROUND[kind]).convert("RGB").resize((1080, 1350), Image.LANCZOS)
    field = src.crop((0, 245, 1080, 1350))
    field = field.resize((round(field.width * H / field.height), H), Image.LANCZOS)
    left = (field.width - W) // 2
    arr = np.asarray(field.crop((left, 0, left + W, H))).astype(np.float32) * 0.5
    y = np.arange(H, dtype=np.float32)[:, None]
    shade = 1 - 0.45 * np.clip((Y0 - y) / Y0, 0, 1) - 0.45 * np.clip((y - Y1) / (H - Y1), 0, 1)
    return Image.fromarray((arr * shade[..., None]).clip(0, 255).astype(np.uint8)).convert("RGBA")


@lru_cache(maxsize=None)
def banner(kind, width):
    """The header (bees, STINGER, tracker title) as a badge: rounded corners and a gold border. The striped
    outfield behind it can't be keyed out cleanly the way the NFL reel keys its turf."""
    hdr = Image.open(BACKGROUND[kind]).convert("RGBA").resize((1080, 1350)).crop((0, 0, 1080, 215))
    hdr = hdr.resize((width, round(hdr.height * width / 1080)), Image.LANCZOS)
    mask = Image.new("L", hdr.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, hdr.width - 1, hdr.height - 1), radius=28, fill=255)
    hdr.putalpha(mask)
    ImageDraw.Draw(hdr).rounded_rectangle((0, 0, hdr.width - 1, hdr.height - 1), radius=28, outline=(*GOLD, 255), width=5)
    return hdr


# ─────────────────────────────────────────────────────────────────
# Scenes
# ─────────────────────────────────────────────────────────────────
class ResultsReel:
    def __init__(self, season, results, miss, hook):
        self.season, self.results, self.miss, self.hook = season, {r["award"]: r for r in results}, miss, hook
        self.all = results
        self.scenes = [(n, s, e) for n, s, e in SCENES]

    def draw(self, name, c, t):
        if name in self.results:
            return self.scene_award(self.results[name], c, t)
        return getattr(self, f"scene_{name}")(c, t)

    def kind(self, name):
        return self.results[name]["kind"] if name in self.results else "mvp"

    def scene_hook(self, c, t):
        bn = banner("mvp", 880)
        put(c, bn, (CX - bn.width // 2, Y0 + 10), prog(t, 0, .4))
        p = pill(f"{self.season} MLB AWARDS")
        put(c, p, (CX - p.width // 2, 520), prog(t, .2, .4))
        words, lines, cur = self.hook.upper().split(), [], ""
        size = 140
        d = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
        for wd in words:
            if cur and text_w(d, f"{cur} {wd}", F("agency_bold", size)) > MAX_W:
                lines.append(cur); cur = wd
            else:
                cur = f"{cur} {wd}".strip()
        lines.append(cur)
        a, k = prog(t, .5, .45), .88 + .12 * prog(t, .5, .45)
        lh = int(size * 1.1)
        y = 960 - len(lines) * lh // 2
        for i, line in enumerate(lines):
            put_c(c, y + i * lh + lh // 2, line, "agency_bold", fit_size(line, "agency_bold", size, MAX_W), WHITE, 5, a, k)
        bw = int(420 * prog(t, 1.0, .6))
        if bw:
            put(c, panel((bw, 12), 6, (*GOLD, 255)), (CX - bw // 2, y + len(lines) * lh + 30))

    def scene_award(self, r, c, t):
        p = pill(r["label"], 56)
        put(c, p, (CX - p.width // 2, Y0 + 10), prog(t, 0, .3))
        a, dy = prog(t, .1, .45), int(50 * (1 - prog(t, .1, .45)))
        lg = load_logo(r["team"], 240)
        put(c, lg, (X1 - lg.width + 10, Y0 + 130 + dy), a * .95)
        hs = headshot(r["key"], 640)
        put(c, hs, (CX - hs.width // 2, Y0 + 150 + dy), a)
        y = Y0 + 150 + hs.height
        name = r["winner"].upper()
        put(c, panel((MAX_W, 100), 14, (0, 0, 0, 225)), (X0, y), a)
        put_c(c, y + 50, name, "agency_bold", fit_size(name, "agency_bold", 76, MAX_W - 40), WHITE, 0, a)

        hit = r["model_rank"] == 1
        rows = [("OUR MODEL", f"No. {r['model_rank']}" if r["model_rank"] else "NOT TOP 10"),
                ("1ST-PLACE VOTES", f"{r['first_place']} OF {r['ballots']}")]
        for i, (lab, val) in enumerate(rows):
            ai = prog(t, .7 + .45 * i, .35)
            card = panel((MAX_W, 120), 18, (10, 10, 10, 215))
            ImageDraw.Draw(card).rounded_rectangle((0, 0, 14, 119), radius=7, fill=(*GOLD, 255))
            li, vi = text_img(lab, "fg_demi", 44, GOLD), text_img(val, "agency_bold", 84, WHITE)
            card.alpha_composite(li, (44, (120 - li.height) // 2))
            card.alpha_composite(vi, (MAX_W - vi.width - 40, (120 - vi.height) // 2))
            put(c, card, (X0 + int(-80 * (1 - ai)), y + 130 + i * 136), ai)

        a3 = prog(t, 1.8, .3)
        stamp_txt = text_img("CALLED IT" if hit else "MISSED", "agency_bold", 72, WHITE)
        mark = check_mark(64, WHITE) if hit else None
        sw = stamp_txt.width + (mark.width + 16 if mark else 0) + 60
        stamp = panel((sw, stamp_txt.height + 40), 20, (*(GREEN if hit else RED), 240))
        x = 30
        if mark:
            stamp.alpha_composite(mark, (x, (stamp.height - mark.height) // 2)); x += mark.width + 16
        stamp.alpha_composite(stamp_txt, (x, 20))
        k = 1.25 - .25 * a3
        stamp = stamp.resize((int(stamp.width * k), int(stamp.height * k)), Image.LANCZOS)
        put(c, stamp, (CX - stamp.width // 2, min(Y1 - stamp.height, y + 420) - (stamp.height - int(stamp.height / k)) // 2), a3)

    def scene_scorecard(self, c, t):
        put_c(c, Y0 + 40, "THE REST OF THE BALLOT", "agency_bold", 72, GOLD, 4, prog(t, 0, .3))
        put_c(c, Y0 + 110, "voters' top 5 that were in our top 5", "fg_demi", 38, GREY, 0, prog(t, .1, .3))
        y = Y0 + 170
        for i, r in enumerate(self.all):
            ai = prog(t, .2 + .15 * i, .35)
            row = panel((MAX_W, 110), 16, (10, 10, 10, 210))
            lab, val = text_img(r["label"], "agency_bold", 54, WHITE), text_img(f"{r['top5_hits']} OF {r['top5_total']}", "agency_bold", 64, GOLD)
            row.alpha_composite(lab, (36, (110 - lab.height) // 2))
            row.alpha_composite(val, (MAX_W - val.width - 36, (110 - val.height) // 2))
            put(c, row, (X0 + int(60 * (1 - ai)), y + i * 124), ai)
        m = self.miss
        if not m:
            return
        y = y + 4 * 124 + 60
        a = prog(t, 1.4, .4)
        put_c(c, y + 20, "BIGGEST MISS", "agency_bold", 60, RED, 3, a)
        card = panel((MAX_W, 190), 18, (10, 10, 10, 225), (*RED, 255), 4)
        ring = headshot_circle(m["key"], 150)
        card.alpha_composite(ring, (24, (190 - ring.height) // 2))
        tx = 24 + ring.width + 24
        nm = m["name"].upper()
        card.alpha_composite(text_img(nm, "agency_bold", fit_size(nm, "agency_bold", 56, MAX_W - tx - 24), WHITE), (tx, 26))
        actual = f"voters: {ordinal(m['actual'])}" if m["actual"] else "voters: no votes"
        card.alpha_composite(text_img(f"{m['award']} · our No. {m['model']} · {actual}", "fg_demi",
                                      fit_size(f"{m['award']} · our No. {m['model']} · {actual}", "fg_demi", 40, MAX_W - tx - 24), GOLD), (tx, 112))
        put(c, card, (X0, y + 70), a)

    def scene_cta(self, c, t):
        a = prog(t, 0, .5)
        bn = banner("mvp", 880)
        put(c, bn, (CX - bn.width // 2, Y0 + 60), a)
        put_c(c, 780, "FULL BOARD AT", "fg_demi", 58, WHITE, 0, a)
        put_c(c, 880, "stingercollectibles.com", "agency_bold", fit_size("stingercollectibles.com", "agency_bold", 112, MAX_W), GOLD, 5, prog(t, .2, .5))
        put_c(c, 1030, "Every final board: Past Results", "fg_demi", 44, WHITE, 0, prog(t, .5, .5))
        put_c(c, 1100, f"Votes: BBWAA, {self.season}", "fg_demi", 40, GREY, 0, prog(t, .5, .5))

    def frame(self, t):
        idx = next((i for i, (_, s, e) in enumerate(self.scenes) if t < e), len(self.scenes) - 1)
        name, s, e = self.scenes[idx]
        c = background(self.kind(name)).copy()
        self.draw(name, c, t - s)
        if idx and t - s < XFADE:
            pn, ps, pe = self.scenes[idx - 1]
            prev = background(self.kind(pn)).copy()
            self.draw(pn, prev, pe - ps)
            c = Image.blend(prev, c, (t - s) / XFADE)
        return c


# ─────────────────────────────────────────────────────────────────
# Outputs
# ─────────────────────────────────────────────────────────────────
def encode(reel, out: Path):
    import imageio_ffmpeg
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(out)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    held = None
    try:
        for i in range(round(DURATION * FPS)):
            t = i / FPS
            if t >= SCENES[-1][1] + 1.0:
                held = held or reel.frame(t).convert("RGB").tobytes()
                proc.stdin.write(held)
            else:
                proc.stdin.write(reel.frame(t).convert("RGB").tobytes())
    finally:
        proc.stdin.close()
        if proc.wait():
            raise RuntimeError("ffmpeg failed")
    return out


def guides(img):
    g = img.copy()
    layer = Image.new("RGBA", g.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    for box in ((0, 0, W, Y0), (0, Y1, W, H), (W - 120, Y0, W, Y1), (0, Y0, 60, Y1)):
        d.rectangle(box, fill=(255, 0, 0, 90))
    d.rectangle((X0, Y0, X1, Y1), outline=(0, 255, 255, 200), width=3)
    g.alpha_composite(layer)
    return g


def check_script(script: dict) -> List[str]:
    out = []
    names = [s[0] for s in SCENES]
    if [b["scene"] for b in script["beats"]] != names:
        out.append(f"beats must be exactly {names}")
    n = sum(len(b["text"].split()) for b in script["beats"])
    if not WORDS[0] <= n <= WORDS[1]:
        out.append(f"{n} words; aim for {WORDS[0]}-{WORDS[1]}")
    hits = {m.group(0) for b in script["beats"] for m in FORBIDDEN.finditer(b["text"])}
    hits |= {m.group(0) for m in FORBIDDEN.finditer(script.get("instagram_caption", ""))}
    if hits:
        out.append(f"probability/betting language: {', '.join(sorted(hits))}")
    if not script["beats"][-1]["text"].rstrip(". ").endswith("stingercollectibles.com"):
        out.append("the last beat should end with stingercollectibles.com")
    return out


def clock(s: float) -> str:
    return f"0:{s:04.1f}".replace(".0", "")


def make(season: int, frames_only: bool = False) -> Path:
    results = award_results(season)
    miss = biggest_miss(results)
    rdir = REPO / "predictions" / str(season) / "reel"
    rdir.mkdir(parents=True, exist_ok=True)
    story_path = rdir / "results_story.json"
    story = json.loads(story_path.read_text(encoding="utf-8")) if story_path.exists() else {}
    hits = sum(r["model_rank"] == 1 for r in results)
    hook = story.get("hook") or f"Our model went {hits} for {len(results)}"

    for r in results:
        print(f"{r['label']:12s} winner {r['winner']} ({r['first_place']}/{r['ballots']} first)  our rank: "
              f"{r['model_rank'] or 'not top 10'}  top-5 overlap {r['top5_hits']}/{r['top5_total']}")
    if miss:
        print(f"Biggest miss: {miss['name']} ({miss['award']}) our No. {miss['model']}, voters "
              f"{ordinal(miss['actual']) if miss['actual'] else 'no votes'}")

    lines = [f"MLB AWARDS RESULTS REEL — {season}", f"Hook: {hook}",
             f"Video: {DURATION} s ({SCENES[-1][2]} s + {HOLD_S} s hold). Votes: BBWAA.", ""]
    script = story.get("script")
    warnings = check_script(script) if script else []
    if script:
        lines.append(f"SCRIPT ({sum(len(b['text'].split()) for b in script['beats'])} words)")
        win = {n: (s, e) for n, s, e in SCENES}
        for i, b in enumerate(script["beats"], 1):
            s, e = win[b["scene"]]
            lines += ["", f"SCENE {i} · {b['scene'].upper()} · {clock(s)}–{clock(e)}", b["text"]]
        lines += ["", "INSTAGRAM CAPTION", script.get("instagram_caption", ""), "", "HASHTAGS",
                  " ".join(script.get("hashtags", []))]
        if warnings:
            lines += ["", "CHECK BEFORE RECORDING", *[f"- {w}" for w in warnings]]
    else:
        lines.append("(No script: add predictions/{season}/reel/results_story.json)")
    for w in warnings:
        print(f"[WARN] {w}")
    (rdir / "results_script.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    meta = {"season": season, "hook": hook, "results": [{k: v for k, v in r.items() if k != "misses"} for r in results],
            "biggest_miss": miss, "warnings": warnings, "video": "results_reel.mp4",
            "release_tag": "reel-latest", "release_asset": f"mlb_results_reel_{season}.mp4"}
    (rdir / "results_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=int) + "\n", encoding="utf-8")

    reel = ResultsReel(season, results, miss, hook)
    if frames_only:
        fdir = rdir / "frames_results"
        fdir.mkdir(exist_ok=True)
        for i, (n, s, e) in enumerate(SCENES, 1):
            img = reel.frame(e - .05)
            img.convert("RGB").save(fdir / f"{i}_{n}.png")
            guides(img).convert("RGB").save(fdir / f"{i}_{n}_guides.png")
        print(f"Frames -> {fdir}")
        return fdir
    out = encode(reel, rdir / "results_reel.mp4")
    print(f"Reel -> {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return out


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--frames-only", action="store_true")
    a = ap.parse_args()
    make(a.season, a.frames_only)
