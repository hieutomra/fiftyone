import os
import fiftyone.operators as foo
import fiftyone.operators.types as types
from fiftyone.operators import ExecutionContext


class DataQualityPanel(foo.Panel):

    ISSUE_TYPES = [
        "brightness",
        "blurriness",
        "aspect ratio",
        "entropy",
        "near duplicates",
        "exact duplicates",
    ]

    @property
    def config(self):
        return foo.PanelConfig(
            name="data_quality_panel",
            label="Data Quality",
        )

    ###
    # LIFECYCLE METHODS
    ###

    def on_load(self, ctx: ExecutionContext):
        """Set initial state"""
        ctx.panel.state.screen = "home"
        ctx.panel.state.issue_type = None

    ###
    # EVENT HANDLERS
    ###

    def navigate_to_screen(
        self, panel, current_screen, ctx: ExecutionContext, issue_type=None
    ):
        """Changes which screen to show"""
        if issue_type:
            ctx.panel.state.issue_type = issue_type
        else:
            ctx.panel.state.issue_type = None

        if current_screen == "home":
            self.home_screen(panel, ctx)
        elif current_screen == "pre_load_compute":
            ctx.panel.state.screen = "pre_load_compute"
            self.pre_load_compute_screen(panel, issue_type, ctx)
        elif current_screen == "analysis":
            ctx.panel.state.screen = "analysis"
        else:
            ctx.panel.state.screen = "home"

    def on_compute_click(self, panel, issue_type, ctx: ExecutionContext):
        self.pre_load_compute_screen(panel, issue_type, ctx, computing=True)
        self.scan_dataset(issue_type, ctx)

    ###
    # COMPUTATION
    ###

    def scan_dataset(self, scan_type, ctx: ExecutionContext):
        pass

    ###
    # SCREENS
    ###
    def home_screen(self, panel, ctx: ExecutionContext):

        subtitle = (  # TODO change text color to muted
            "###### Find data quality issues in your dataset and act on them."
        )
        panel.md(subtitle, name="home_subtitle")

        for i, issue_type in enumerate(self.ISSUE_TYPES):
            card = panel.h_stack(
                f"collapsed_{issue_type}",
                align_x="center",
                align_y="center",
                gap=5,
                container=types.PaperContainer(),
            )
            sub_card_left = card.h_stack(
                f"collapsed_sub_left_{issue_type}",
                align_x="left",
                gap=0,
                container=types.PaperContainer(py=2, px=2, elevation=1),
            )

            sub_card_left.list(
                f"collapsed_text_{issue_type}",
                types.Object(),
                view=types.MarkdownView(),
                label=f"##### {issue_type.title()}",
            )

            sub_card_right = card.h_stack(
                f"collapsed_sub_right_{issue_type}",
                align_x="right",
                gap=3,
                container=types.PaperContainer(py=2, px=2, elevation=1),
            )
            badge_schema = {
                "text": "Not Started"
            }  # TODO: hook this up to review stage stored in ExecutionStore
            badge = types.PillBadgeView(**badge_schema)
            sub_card_right.obj(f"collapsed_badge_{issue_type}", view=badge)

            icon_schema = {"icon": "ArrowForwardIcon"}
            icon = types.IconButtonView(**icon_schema)
            sub_card_right.obj(
                f"collapsed_icon_{issue_type}",
                view=icon,
                # on_click=self.navigate_to_screen( TODO: bug, on_click triggers immediately on render, why?
                #     panel, "pre_load_compute", ctx, issue_type=issue_type
                # ),
            )

    def pre_load_compute_screen(
        self, panel, issue_type, ctx: ExecutionContext, computing=False
    ):
        card_header = panel.h_stack(
            "navbar",
            align_x="left",
            align_y="center",
            gap=2,
            container=types.PaperContainer(),
        )

        icon_schema = {"icon": "ArrowForwardIcon"}
        icon = types.IconButtonView(**icon_schema)
        card_header.obj(
            f"back_button",
            view=icon,
            on_click=self.navigate_to_screen(panel, "home_screen", ctx),
        )

        text = "All data quality issue types"
        card_header.md(text, name="back_button_text")

        card_main = panel.v_stack(
            "pre_load_compute",
            align_y="center",
            align_x="center",
            container=types.PaperContainer(),
        )
        sub_card_main = card_main.h_stack(
            "sub_card_main",
            align_x="center",
            align_y="center",
            container=types.PaperContainer(),
        )
        sub_card_main_left = sub_card_main.h_stack(
            f"expanded_sub_left_{issue_type}",
            align_x="left",
            gap=0,
            container=types.PaperContainer(py=2, px=2, elevation=1),
        )

        sub_card_main_left.list(
            f"collapsed_text_{issue_type}",
            types.Object(),
            view=types.MarkdownView(),
            label=f"##### {issue_type.title()}",
        )

        sub_card_main_right = sub_card_main.h_stack(
            f"expanded_sub_right_{issue_type}",
            align_x="right",
            gap=3,
            container=types.PaperContainer(py=2, px=2, elevation=1),
        )
        badge_schema = {
            "text": "Not Started"
        }  # TODO: hook this up to review stage stored in ExecutionStore
        badge = types.PillBadgeView(**badge_schema)
        sub_card_main_right.obj(f"expanded_badge_{issue_type}", view=badge)

        if computing:
            loader_schema = {
                "variant": "spinner",
                "color": "base",
                "size": "medium",
            }
            loader = types.LoadingView(**loader_schema)
            card_main.obj("loader", view=loader)
        else:
            card_main.md(  # TODO: figure out how to render an image from source
                f'![Pixelated Heart]({"assets/pixelated-heart.svg"} "Pixelated Heart")',
                name="pixelated_heart",
            )

        text = "Find, curate, and act on blurry images within your dataset easily with FiftyOne."
        card_main.md(text, name="back_button_text")

        if computing:
            card_main.btn(
                f"compute_button",
                label=f"Scanning Dataset for {issue_type.title()}...",
                variant="contained",
                # disabled=True,
            )
        else:
            card_main.btn(
                f"compute_button",
                label=f"Scan Dataset for {issue_type.title()}",
                variant="contained",
                on_click=self.on_compute_click(
                    panel, issue_type, ctx
                ),  # TODO: bug, on_click triggers immediately on render, why?
            )

    def render(self, ctx: ExecutionContext):
        panel = types.Object()

        self.navigate_to_screen(panel, "home", ctx)

        return types.Property(
            panel,
            view=types.GridView(align_x="left", gap=3, px=3, pt=3),
        )


def register(p):
    p.register(DataQualityPanel)
