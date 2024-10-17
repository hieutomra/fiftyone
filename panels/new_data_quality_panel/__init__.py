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

    DEFAULT_ISSUE_CONFIG = {
        "brightness": {
            "detect_method": "std_dev",
            "std": 3.0,
            # "handling": "save_view" # if this is set then panel will auto
            # handle the issues without prompting the user
        },
        "blurriness": {"detect_method": "threshold", "min": None, "max": 150},
        "aspect ratio": {
            "detect_method": "threshold",
            "min": None,
            "max": 0.3,
        },
        "entropy": {"detect_method": "percentage", "min": None, "max": 0.15},
        "near duplicates": {"sim_threshold": 0.05},
        "exact duplicates": {"sim_threshold": 0.05},
    }

    DEFAULT_ISSUE_COUNTS = {
        "brightness": 0,
        "blurriness": 0,
        "aspect ratio": 0,
        "entropy": 0,
        "near duplicates": 0,
        "exact duplicates": 0,
    }

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
        ctx.trigger(
            "split_panel",
            {"name": "data_quality_panel", "layout": "horizontal"},
        )
        ctx.panel.state.screen = "pre_load_compute"
        ctx.panel.state.issue_type = "brightness"
        ctx.panel.state.computing = False
        ctx.panel.state.issue_config = self.DEFAULT_ISSUE_CONFIG
        ctx.panel.state.issue_counts = self.DEFAULT_ISSUE_COUNTS

    ###
    # EVENT HANDLERS
    ###

    def navigate_to_screen(self, ctx: ExecutionContext):
        """Changes which screen to show"""
        ctx.panel.state.issue_type = ctx.params.get("issue_type", None)
        ctx.panel.state.screen = ctx.params.get("next_screen", "home")

    def on_compute_click(self, ctx: ExecutionContext):
        issue_type = ctx.params.get("issue_type", None)

        ctx.panel.state.computing = True
        self.scan_dataset(issue_type, ctx)
        ctx.panel.state.computing = False

    def on_change_set_threshold(self, ctx: ExecutionContext):
        """Change the config based on the threshold values"""
        config = ctx.panel.state.issue_config
        issue_type = ctx.panel.state.issue_type

        if ctx.params["value"] == "Save Threshold":
            config[issue_type] = {}
            config[issue_type]["detect_method"] = "threshold"
            config[issue_type]["min"] = ctx.panel.state.hist_lower_thresh
            config[issue_type]["max"] = ctx.panel.state.hist_upper_thresh
            ctx.panel.state.issue_config = config
        elif ctx.params["value"] == "Reset Threshold":
            config[issue_type] = self.DEFAULT_ISSUE_CONFIG[issue_type]
            ctx.panel.state.hist_lower_thresh = self.DEFAULT_ISSUE_CONFIG[
                issue_type
            ]["min"]
            ctx.panel.state.hist_upper_thresh = self.DEFAULT_ISSUE_CONFIG[
                issue_type
            ]["max"]
            ctx.panel.state.issue_config = config

            # Reset view to default
            # TODO - REST VIEW LOGIC HERE

    ###
    # COMPUTATION
    ###

    def scan_dataset(self, scan_type, ctx: ExecutionContext):
        ctx.panel.state.screen = "analysis"

        # TODO - ADD SCAN DATASET LOGIC HERE

    ###
    # SCREENS
    ###
    def home_screen(self, panel, ctx: ExecutionContext):
        subtitle = (  # TODO change text color to muted
            "###### Find data quality issues in your dataset and act on them."
        )
        panel.md(subtitle, name="home_subtitle")

        for issue_type in self.ISSUE_TYPES:
            card = panel.h_stack(
                f"collapsed_{issue_type}",
                align_x="center",
                align_y="center",
                gap=5,
                container=types.PaperContainer(py=2, px=2),
            )
            sub_card_left = card.h_stack(
                f"collapsed_sub_left_{issue_type}",
                align_x="left",
                align_y="center",
                gap=0,
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
                align_y="center",
                gap=3,
            )
            badge_schema = {
                "text": "Not Started"
            }  # TODO: hook this up to review stage stored in ExecutionStore
            badge = types.PillBadgeView(**badge_schema)
            sub_card_right.obj(f"collapsed_badge_{issue_type}", view=badge)

            icon_schema = {"icon": "arrow_forward", "variant": "filled"}
            icon = types.IconButtonView(**icon_schema)
            sub_card_right.view(
                f"collapsed_icon_{issue_type}",
                view=icon,
                on_click=self.navigate_to_screen,
                params={
                    "issue_type": issue_type,
                    "next_screen": "pre_load_compute",
                },
            )

    def _shared_screen(
        self, panel, issue_type, shared_section, ctx: ExecutionContext
    ):
        if shared_section == "navbar":
            card_header = panel.h_stack(
                "navbar",
                align_x="left",
                align_y="center",
                gap=2,
            )
            icon_schema = {"icon": "arrow_back", "variant": "filled"}
            icon = types.IconButtonView(**icon_schema)
            card_header.obj(
                f"back_button",
                view=icon,
                on_click=self.navigate_to_screen,
                params={"next_screen": "home"},
            )
            text = "All data quality issue types"
            card_header.md(text, name="back_button_text")
        elif shared_section == "expanded_card":

            card_main = panel.v_stack(
                "pre_load_compute",
                align_y="center",
                align_x="center",
            )
            sub_card_main = card_main.h_stack(
                "sub_card_main", align_x="center", align_y="center", gap=20
            )
            sub_card_main_left = sub_card_main.h_stack(
                f"expanded_sub_left_{issue_type}",
                align_x="left",
                gap=2,
            )
            sub_card_main_left.list(
                f"collapsed_text_{issue_type}",
                types.Object(),
                view=types.MarkdownView(),
                label=f"##### {issue_type.title()}",
            )

            if ctx.panel.state.screen == "analysis":
                # TODO : add circle icon
                sub_card_main_left.list(
                    f"collapsed_issue_count_{issue_type}",
                    types.Object(),
                    view=types.MarkdownView(),
                    label=f"###### {ctx.panel.state.issue_counts[issue_type]} Potential Issues",
                )

            sub_card_main_right = sub_card_main.h_stack(
                f"expanded_sub_right_{issue_type}",
                align_x="right",
                gap=1,
            )

            if ctx.panel.state.screen == "analysis":
                badge_schema = {
                    "text": [["In Review", "primary"], ["Reviewed", "success"]]
                }  # TODO: hook this up to review stage stored in ExecutionStore
                badge = types.PillBadgeView(**badge_schema)
                sub_card_main_right.obj(
                    f"expanded_badge_{issue_type}", view=badge
                )

                dropdown = types.DropdownView(icon="SettingsIcon")
                dropdown.add_choice("Save Threshold", label="Save Threshold")
                dropdown.add_choice("Reset Threshold", label="Reset Threshold")
                sub_card_main_right.view(
                    "threshold_setting_menu",
                    view=dropdown,
                    on_change=self.on_change_set_threshold,
                )
            else:
                badge_schema = {"text": "Not Started"}
                badge = types.PillBadgeView(**badge_schema)
                sub_card_main_right.obj(
                    f"expanded_badge_{issue_type}", view=badge
                )

    def pre_load_compute_screen(
        self, panel, issue_type, ctx: ExecutionContext, computing=False
    ):
        self._shared_screen(
            panel, issue_type, shared_section="navbar", ctx=ctx
        )

        card_main = panel.v_stack(
            "pre_load_compute",
            align_y="center",
            align_x="center",
            container=types.PaperContainer(py=2, px=2),
        )

        self._shared_screen(
            card_main, issue_type, shared_section="expanded_card", ctx=ctx
        )

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
                disabled=True,
            )
        else:
            card_main.btn(
                f"compute_button",
                label=f"Scan Dataset for {issue_type.title()}",
                variant="contained",
                on_click=self.on_compute_click,
                params={"issue_type": issue_type},
            )

    def analysis_screen(self, panel, issue_type, ctx: ExecutionContext):
        self._shared_screen(
            panel, issue_type, shared_section="navbar", ctx=ctx
        )
        card_main = panel.v_stack(
            "analysis",
            align_y="center",
            align_x="center",
            container=types.PaperContainer(py=2, px=2),
        )
        self._shared_screen(
            card_main, issue_type, shared_section="expanded_card", ctx=ctx
        )

        # TODO - ADD CHARTING & ANALYSIS SCREEN HERE

    def render(self, ctx: ExecutionContext):
        panel = types.Object()

        if ctx.panel.state.screen == "home":
            self.home_screen(panel, ctx)
        elif ctx.panel.state.screen == "pre_load_compute":
            self.pre_load_compute_screen(
                panel,
                ctx.panel.state.issue_type,
                ctx,
                ctx.panel.state.computing,
            )
        elif ctx.panel.state.screen == "analysis":
            self.analysis_screen(panel, ctx.panel.state.issue_type, ctx)
        else:
            self.home_screen(panel, ctx)

        return types.Property(
            panel,
            view=types.GridView(align_x="left", gap=3, px=3, pt=3),
        )


def register(p):
    p.register(DataQualityPanel)
