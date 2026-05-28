TEMPLATE = """\
from manim import *

class MainScene(Scene):
    def construct(self):
        self.camera.background_color = "#0a0a0a"

        MAX_HEIGHT = 4.0
        BAR_W      = 1.4
        GAP        = 0.9
        BASELINE_Y = -2.8

        n       = len(bars_data)
        total_w = n * BAR_W + (n - 1) * GAP
        start_x = -total_w / 2 + BAR_W / 2

        bars, val_labels, name_labels, trackers = [], [], [], []

        for i, d in enumerate(bars_data):
            x        = start_x + i * (BAR_W + GAP)
            t        = ValueTracker(0)
            target_h = (d["pct"] / MAX_PCT) * MAX_HEIGHT
            trackers.append(t)

            bar = always_redraw(lambda t=t, x=x, c=d["color"], th=target_h: (
                Rectangle(width=BAR_W, height=max(0.001, t.get_value() * th))
                .set_fill(c, opacity=0.92)
                .set_stroke(width=0)
                .move_to([x, BASELINE_Y + t.get_value() * th / 2, 0])
            ))

            val = always_redraw(lambda t=t, x=x, d=d, th=target_h: (
                Text(
                    f"{round(t.get_value() * d['pct'])}{UNIT_SUFFIX}",
                    font_size=28, color="#EBEBEB",
                ).move_to([x, BASELINE_Y + t.get_value() * th + 0.32, 0])
            ))

            name = Text(
                d["label"], font_size=22, color="#666666",
            ).move_to([x, BASELINE_Y - 0.45, 0])

            bars.append(bar)
            val_labels.append(val)
            name_labels.append(name)

        caption = Text(
            CAPTION, font_size=22, color="#444444",
        ).move_to([0, BASELINE_Y - 1.05, 0])

        # — fair line (what equal treatment looks like) ————————
        fair_pct  = bars_data[FAIR_LINE_IDX]["pct"]
        fair_y    = BASELINE_Y + (fair_pct / MAX_PCT) * MAX_HEIGHT
        fair_line = DashedLine(
            start=LEFT * 3.2, end=RIGHT * 3.2,
            dash_length=0.12, stroke_width=1.5, color="#E63946",
        ).move_to([0, fair_y, 0]).set_opacity(0)
        fair_label = Text(
            FAIR_LINE_LABEL, font_size=20, color="#E63946",
        ).next_to(fair_line, RIGHT, buff=0.16).set_opacity(0)

        [self.add(b, v, nm) for b, v, nm in zip(bars, val_labels, name_labels)]
        self.add(caption, fair_line, fair_label)

        grow_time = DURATION_SEC * 0.60
        self.play(
            AnimationGroup(
                *[t.animate.set_value(1) for t in trackers],
                lag_ratio=0.2,
            ),
            run_time=grow_time,
            rate_func=smooth,
        )

        # — reveal fair line ———————————————————————————————————
        self.play(
            fair_line.animate.set_opacity(0.7),
            fair_label.animate.set_opacity(1),
            run_time=0.5,
        )

        self.wait(max(0.1, DURATION_SEC - grow_time - 0.5))
"""

EXTRACT = """\
Sentence: "{sentence_text}"
Duration: {duration_sec}s

Return a JSON object with exactly these keys. No explanation, no markdown.

{{
  "bars_data": [
    {{"label": "use \\\\n for line breaks", "pct": <integer>, "color": "#457B9D or #E63946"}}
  ],
  "UNIT_SUFFIX":    "% for percentages, x for ratios, empty string for plain numbers",
  "CAPTION":        "short caption naming what the bars show, e.g. \\"effective tax rate\\"",
  "FAIR_LINE_IDX":  <integer index into bars_data of the fair/should-be reference bar>,
  "FAIR_LINE_LABEL": "label for the reference line, e.g. \\"investor rate\\"",
  "DURATION_SEC":   {duration_sec}
}}

Rules:
- bars_data must have 2-4 items
- pct must be an integer (not float, not string)
- use "#457B9D" for neutral bars, "#E63946" for the alarming bar
- FAIR_LINE_IDX must be a valid index (0 to len(bars_data)-1)
- MAX_PCT is computed automatically — do not include it
"""

DEBUG = """\
This comparison animation config failed. Fix the JSON and return corrected JSON only.

Sentence: "{sentence_text}"
Attempt: {attempt}/{max_attempts}

CURRENT JSON:
{broken_config}

ERROR:
{error_output}

Common fixes:
- pct must be an integer: 22 not 22.0 or "22"
- FAIR_LINE_IDX must be a valid index (0 to len(bars_data)-1)
- color must be a hex string: "#457B9D" or "#E63946"
- bars_data must have 2-4 items
"""
