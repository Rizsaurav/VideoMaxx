TEMPLATE = """\
from manim import *

class MainScene(Scene):
    def construct(self):
        self.camera.background_color = "#0a0a0a"

        # 3-column layout: [labels | bars | amounts]
        LABEL_R  = -2.5   # right edge of label column
        BAR_L    = -2.3   # left edge of all bars
        MAX_BAR  = 7.2    # max bar width (= gross bar width)
        AMT_L    = 5.1    # left edge of amount column
        BAR_H    = 0.62
        ROW_STEP = 1.05

        def make_row(y, label_txt, amount, color):
            w = max(MAX_BAR * (amount / GROSS), 0.08)
            bar = Rectangle(width=w, height=BAR_H)
            bar.set_fill(color, opacity=0.9).set_stroke(width=0)
            bar.move_to([BAR_L + w / 2, y, 0])
            lbl = Text(label_txt, font_size=22, color="#888888")
            lbl.next_to([LABEL_R, y, 0], LEFT, buff=0)
            amt = Text(f"−${amount:,}", font_size=22, color=color)
            amt.next_to([AMT_L, y, 0], RIGHT, buff=0)
            return bar, lbl, amt

        gross_y   = 2.2
        gross_bar = Rectangle(width=MAX_BAR, height=BAR_H)
        gross_bar.set_fill("#EBEBEB", opacity=0.15).set_stroke("#EBEBEB", width=1)
        gross_bar.move_to([BAR_L + MAX_BAR / 2, gross_y, 0])
        gross_lbl = Text(GROSS_LABEL, font_size=24, color="#EBEBEB")
        gross_lbl.next_to([LABEL_R, gross_y, 0], LEFT, buff=0)
        gross_amt = Text(f"${GROSS:,}", font_size=24, color="#EBEBEB")
        gross_amt.next_to([AMT_L, gross_y, 0], RIGHT, buff=0)

        self.add(gross_bar, gross_lbl, gross_amt)
        self.wait(0.6)

        n_ded     = len(deductions)
        row_ys    = [gross_y - ROW_STEP * (i + 1) for i in range(n_ded)]
        step_time = (DURATION_SEC * 0.65) / max(n_ded, 1)

        for d, y in zip(deductions, row_ys):
            bar, lbl, amt = make_row(y, d["label"], d["amount"], d["color"])
            bar.set_opacity(0)
            lbl.set_opacity(0)
            amt.set_opacity(0)
            self.add(bar, lbl, amt)
            self.play(
                bar.animate.set_opacity(0.9),
                lbl.animate.set_opacity(1),
                amt.animate.set_opacity(1),
                run_time=step_time * 0.7,
                rate_func=smooth,
            )
            self.wait(step_time * 0.3)

        total_deducted = sum(d["amount"] for d in deductions)
        remaining_amt  = GROSS - total_deducted
        remain_w       = max(MAX_BAR * (remaining_amt / GROSS), 0.08)
        remain_y       = row_ys[-1] - ROW_STEP if row_ys else -2.5

        remain_bar = Rectangle(width=remain_w, height=BAR_H)
        remain_bar.set_fill("#E63946", opacity=0.95).set_stroke(width=0)
        remain_bar.move_to([BAR_L + remain_w / 2, remain_y, 0])
        remain_lbl = Text("what remains", font_size=24, color="#E63946")
        remain_lbl.next_to([LABEL_R, remain_y, 0], LEFT, buff=0)
        remain_val = Text(f"${remaining_amt:,}", font_size=24, color="#E63946")
        remain_val.next_to([AMT_L, remain_y, 0], RIGHT, buff=0)

        self.play(
            FadeIn(remain_bar, shift=UP * 0.1),
            FadeIn(remain_lbl),
            FadeIn(remain_val),
            run_time=0.7,
        )

        used = 0.6 + (step_time * n_ded) + 0.7
        self.wait(max(0.1, DURATION_SEC - used))
"""

EXTRACT = """\
Sentence: "{sentence_text}"
Duration: {duration_sec}s

Return a JSON object with exactly these keys. No explanation, no markdown.

{{
  "GROSS":       <integer — starting amount, e.g. 60000>,
  "GROSS_LABEL": "what the gross amount represents, e.g. \\"gross salary\\"",
  "deductions":  [
    {{"label": "lowercase 2-4 words", "amount": <integer>, "color": "#E63946"}}
  ],
  "DURATION_SEC": {duration_sec}
}}

Rules:
- GROSS must be an integer
- deduction amounts must be integers, sorted largest first
- sum of amounts must be less than GROSS
- 3-5 deductions max — combine small ones
- colors in order: "#E63946", "#c1121f", "#9d0208", "#6a040f"
- if exact amounts aren't stated, estimate proportionally from percentages
"""

DEBUG = """\
This shrink animation config failed. Fix the JSON and return corrected JSON only.

Sentence: "{sentence_text}"
Attempt: {attempt}/{max_attempts}

CURRENT JSON:
{broken_config}

ERROR:
{error_output}

Common fixes:
- GROSS must be an integer: 60000 not 60000.0
- amounts must be integers, not floats or strings
- sum of deduction amounts must be less than GROSS
- deductions must have "label", "amount", "color" keys
"""
