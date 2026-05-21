TEMPLATE = """\
from manim import *

class MainScene(Scene):
    def construct(self):
        self.camera.background_color = "#0a0a0a"

        tracker    = ValueTracker(0)
        color_flag = ValueTracker(0)   # 0 = #EBEBEB, 1 = #E63946

        number = always_redraw(
            lambda: Text(
                f"{int(tracker.get_value()):,}",
                font_size=96,
                color="#E63946" if color_flag.get_value() > 0.5 else "#EBEBEB",
            ).move_to(ORIGIN + UP * 0.3)
        )

        dollar = always_redraw(
            lambda: Text("$", font_size=56, color="#EBEBEB")
            .next_to(number, LEFT, buff=0.08)
        )

        subtitle = Text(
            LABEL, font_size=26, color="#555555"
        ).move_to([0, -0.48, 0])

        # — threshold line (hidden initially) —————————————————
        cap_line = Line(
            start=LEFT * 5.5, end=RIGHT * 5.5,
            stroke_width=1.5, color="#E63946",
        ).move_to(ORIGIN + UP * 1.9)
        cap_line.set_opacity(0)

        cap_label = Text(
            THRESHOLD_LABEL if THRESHOLD_LABEL else "·", font_size=22, color="#E63946",
        ).next_to(cap_line, UP, buff=0.16)
        cap_label.set_opacity(0)

        # — add to scene ——————————————————————————————————————
        if PREFIX == "$":
            self.add(subtitle, dollar, number, cap_line, cap_label)
        else:
            self.add(subtitle, number, cap_line, cap_label)

        # — phase 1: count up to threshold (38% of duration) ——
        t1 = DURATION_SEC * 0.38
        self.play(
            tracker.animate.set_value(THRESHOLD if THRESHOLD else VALUE),
            run_time=t1,
            rate_func=rush_into,
        )

        # — reveal threshold line (only when threshold exists) —
        if THRESHOLD:
            self.play(
                cap_line.animate.set_opacity(0.7),
                cap_label.animate.set_opacity(1),
                run_time=0.4,
            )
            self.wait(0.3)

        # — phase 2: continue to target (40% of duration) ———
        t2 = DURATION_SEC * 0.40
        self.play(
            tracker.animate.set_value(VALUE),
            run_time=t2,
            rate_func=linear,
        )

        self.play(color_flag.animate.set_value(1), run_time=0.25)

        # — hold ———————————————————————————————————————————
        used = t1 + t2 + (0.7 if THRESHOLD else 0) + 0.25
        self.wait(max(0.1, DURATION_SEC - used))
"""

EXTRACT = """\
Sentence: "{sentence_text}"
Duration: {duration_sec}s

Return a JSON object with exactly these keys. No explanation, no markdown.

{{
  "LABEL":           "short phrase naming what is being counted, e.g. \\"your annual income\\"",
  "VALUE":           <float — the target number, e.g. 400000.0 or 38.0>,
  "PREFIX":          "\\"$\\" for dollar amounts, \\"\\" for everything else",
  "THRESHOLD":       <float if sentence mentions a cap/fair value to cross, else null>,
  "THRESHOLD_LABEL": "label for threshold line e.g. \\"tax cap: $184,500\\" — empty string if THRESHOLD is null",
  "DURATION_SEC":    {duration_sec}
}}

Rules:
- VALUE must be a number, not a string
- THRESHOLD must be a number less than VALUE, or null (not the string "null")
- PREFIX is "$" only for dollar amounts; for percentages use "" and include % in LABEL
- LABEL names what the number represents, never repeats the number itself
"""

DEBUG = """\
This counting animation config failed. Fix the JSON and return corrected JSON only.

Sentence: "{sentence_text}"
Attempt: {attempt}/{max_attempts}

CURRENT JSON:
{broken_config}

ERROR:
{error_output}

Common fixes:
- VALUE and THRESHOLD must be numbers: 400000.0 not "400,000"
- THRESHOLD must be null or a number less than VALUE
- PREFIX must be a string: "$" or ""
- LABEL must be a string
"""
