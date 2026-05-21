TEMPLATE = """\
from manim import *

class MainScene(Scene):
    def construct(self):
        self.camera.background_color = "#0a0a0a"

        NODE_W = 2.1
        NODE_H = 0.92
        GAP    = 0.68
        TXT    = "#EBEBEB"
        RED    = "#E63946"
        DIM    = "#444444"

        n       = len(nodes_data)
        total_w = n * NODE_W + (n - 1) * GAP
        start_x = -total_w / 2 + NODE_W / 2

        node_groups = []
        for i, nd in enumerate(nodes_data):
            x   = start_x + i * (NODE_W + GAP)
            col = RED if nd.get("highlight") else DIM

            box = RoundedRectangle(
                corner_radius=0.07,
                width=NODE_W, height=NODE_H,
                color=col,
                fill_color="#1a1a1a",
                fill_opacity=1,
                stroke_width=1.5,
            ).move_to([x, 0.3, 0])

            title_txt = Text(nd["title"], font_size=26, color=TXT)
            title_txt.move_to(box.get_center() + UP * 0.16)

            sub_txt = Text(nd["sub"], font_size=20, color=col)
            sub_txt.move_to(box.get_center() + DOWN * 0.16)

            grp = VGroup(box, title_txt, sub_txt)
            grp.set_opacity(0)
            node_groups.append((grp, x))
            self.add(grp)

        t_each = (DURATION_SEC * 0.70) / max(2 * n - 1, 1)

        for i, (grp, x) in enumerate(node_groups):
            self.play(grp.animate.set_opacity(1), run_time=max(0.3, t_each))

            if i < len(node_groups) - 1:
                x_next = node_groups[i + 1][1]
                arr = Arrow(
                    start=[x + NODE_W / 2, 0.3, 0],
                    end=[x_next - NODE_W / 2, 0.3, 0],
                    color=RED, buff=0,
                    stroke_width=2.5, tip_length=0.18,
                    max_tip_length_to_length_ratio=0.5,
                )
                arr.set_opacity(0)
                self.add(arr)
                self.play(arr.animate.set_opacity(1), run_time=max(0.2, t_each * 0.5))

        if CIRCULAR:
            first_x = node_groups[0][1]
            last_x  = node_groups[-1][1]
            ret = CurvedArrow(
                start_point=[last_x, 0.3 - NODE_H / 2 - 0.05, 0],
                end_point=[first_x, 0.3 - NODE_H / 2 - 0.05, 0],
                angle=-TAU / 4,
                color=RED,
                stroke_width=2,
                tip_length=0.18,
            )
            ret_lbl = Text(RETURN_LABEL, font_size=20, color=RED)
            ret_lbl.next_to(ret, DOWN, buff=0.13)
            self.play(Create(ret), FadeIn(ret_lbl), run_time=0.5)

        self.wait(max(0.1, DURATION_SEC * 0.15))
"""

EXTRACT = """\
Sentence: "{sentence_text}"
Duration: {duration_sec}s

Return a JSON object with exactly these keys. No explanation, no markdown.

{{
  "nodes_data": [
    {{"title": "1-2 word entity name", "sub": "2-3 word action", "highlight": <true if entity harmed, else false>}}
  ],
  "CIRCULAR":     <true if flow loops back to origin, else false>,
  "RETURN_LABEL": "label on return arrow e.g. \\"← nothing returned\\"",
  "DURATION_SEC": {duration_sec}
}}

Rules:
- nodes_data must have 2-5 items in flow order
- title: "you", "landlord", "REIT", "Wall St"
- sub: "pay rent", "keeps margin", "extracts value"
- highlight: true for the entity subject to harm/unfair treatment
- CIRCULAR: true only when money/power returns to origin
"""

DEBUG = """\
This flow diagram config failed. Fix the JSON and return corrected JSON only.

Sentence: "{sentence_text}"
Attempt: {attempt}/{max_attempts}

CURRENT JSON:
{broken_config}

ERROR:
{error_output}

Common fixes:
- highlight must be a boolean: true or false (not "true" string)
- CIRCULAR must be a boolean: true or false
- nodes_data must have 2-5 items with "title", "sub", "highlight" keys
"""
