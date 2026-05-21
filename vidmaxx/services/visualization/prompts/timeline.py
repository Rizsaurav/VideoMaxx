TEMPLATE = """\
from manim import *
import numpy as np

class MainScene(Scene):
    def construct(self):
        self.camera.background_color = "#0a0a0a"

        LINE_Y = 0.2
        LINE_L = -6.0
        LINE_R =  6.0
        TXT    = "#EBEBEB"
        RED    = "#E63946"
        DIM    = "#555555"

        n       = len(events)
        min_yr  = min(year_vals)
        max_yr  = max(year_vals)
        yr_span = max(max_yr - min_yr, 1)

        xs = [LINE_L + (y - min_yr) / yr_span * (LINE_R - LINE_L) for y in year_vals]
        for i in range(1, len(xs)):
            if xs[i] - xs[i - 1] < 1.0:
                xs[i] = xs[i - 1] + 1.0
        shift = max(0, xs[-1] - LINE_R) / 2 if xs else 0
        xs = [x - shift for x in xs]

        timeline = Line([LINE_L, LINE_Y, 0], [LINE_R, LINE_Y, 0], color=TXT, stroke_width=2)
        self.play(Create(timeline), run_time=max(0.3, DURATION_SEC * 0.10))

        t_each = (DURATION_SEC * 0.58) / max(n, 1)

        for i, (ev, x) in enumerate(zip(events, xs)):
            is_key = ev.get("key", False)
            dot_r  = 0.12 if is_key else 0.07
            col    = RED if is_key else TXT

            dot = Circle(radius=dot_r, color=col, fill_color=col, fill_opacity=1, stroke_width=0)
            dot.move_to([x, LINE_Y, 0])

            tick = Line([x, LINE_Y - 0.06, 0], [x, LINE_Y + 0.06, 0], color=TXT, stroke_width=1.5)

            year_txt = Text(ev["year"], font_size=22, color=DIM)
            year_txt.move_to([x, LINE_Y - 0.38, 0])

            desc_txt = Text(ev["desc"], font_size=24, color=col, line_spacing=0.8)
            if i % 2 == 0:
                desc_txt.move_to([x, LINE_Y + 0.85, 0])
            else:
                desc_txt.move_to([x, LINE_Y - 0.95, 0])

            grp = VGroup(dot, tick, year_txt, desc_txt)
            grp.set_opacity(0)
            self.add(grp)
            self.play(grp.animate.set_opacity(1), run_time=max(0.3, t_each), rate_func=ease_out_cubic)

        i1, i2 = GAP_X_IDX
        gap_x   = (xs[i1] + xs[i2]) / 2
        brace_y = LINE_Y - 1.55
        brace = BraceBetweenPoints(
            np.array([xs[i1], brace_y, 0]),
            np.array([xs[i2], brace_y, 0]),
            direction=DOWN,
            color=RED,
        )
        gap_lbl = Text(GAP_LABEL_TEXT, font_size=26, color=RED)
        gap_lbl.move_to([gap_x, brace_y - 0.45, 0])
        self.play(FadeIn(brace), FadeIn(gap_lbl), run_time=0.5)

        self.wait(max(0.1, DURATION_SEC * 0.12))
"""

EXTRACT = """\
Sentence: "{sentence_text}"
Duration: {duration_sec}s

Return a JSON object with exactly these keys. No explanation, no markdown.

{{
  "events": [
    {{"year": "1935", "desc": "Social Security\\\\ncreated", "key": false}},
    {{"year": "2003", "desc": "capital gains\\\\ncut to 15%", "key": true}}
  ],
  "year_vals":      [<integers matching events in order, e.g. [1935, 2003]>],
  "GAP_LABEL_TEXT": "label for the largest time gap, e.g. \\"53 years without a raise\\"",
  "GAP_X_IDX":      [<i1>, <i2>],
  "DURATION_SEC":   {duration_sec}
}}

Rules:
- year must be a string: "1935" not 1935
- year_vals must be integers: [1935, 2003] not strings
- events and year_vals must have the same length
- key: true for exactly one event — the one the sentence argues about
- GAP_X_IDX: [i1, i2] where i1 and i2 are adjacent indices bounding the largest gap
- use \\\\n in desc to break across 2 lines (3-6 words total)
"""

DEBUG = """\
This timeline config failed. Fix the JSON and return corrected JSON only.

Sentence: "{sentence_text}"
Attempt: {attempt}/{max_attempts}

CURRENT JSON:
{broken_config}

ERROR:
{error_output}

Common fixes:
- year must be a string: "1935" not 1935
- year_vals must be integers: [1935, 2003] not strings
- GAP_X_IDX must be a 2-element array of valid indices: [0, 1]
- events and year_vals must have the same length
- exactly one event must have key=true
"""
