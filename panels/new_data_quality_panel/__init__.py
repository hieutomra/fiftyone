import fiftyone.operators as foo
import fiftyone.operators.types as types
from fiftyone.operators import ExecutionContext


class DataQualityPanel(foo.Panel):

    ISSUE_TYPES = [
        "brightness",
        "blurriness",
        "aspect_ratio",
        "entropy",
        "near_duplicates",
        "exact_duplicates",
    ]

    @property
    def config(self):
        return foo.PanelConfig(
            name="data_quality_panel",
            label="Data Quality Issues Panel",
        )

    def on_load(self, ctx: ExecutionContext):
        """Set initial state"""
        ctx.panel.state.screen = "home"
        ctx.panel.state.issue_type = None

    def navigate_to_screen(self, panel, current_screen, ctx: ExecutionContext):
        """Changes which screen to show"""
        if current_screen == "home":
            self.home_screen(panel, ctx)
        elif current_screen == "pre_load_compute":
            ctx.panel.state.screen = "pre_load_compute"
        elif current_screen == "analysis":
            ctx.panel.state.screen = "analysis"
        else:
            ctx.panel.state.screen = "home"

    def home_screen(self, panel, ctx):

        subtitle = "Find data quality issues in your dataset and act on them."
        panel.md(subtitle, name="home_subtitle")

    def render(self, ctx: ExecutionContext):
        panel = types.Object()

        self.navigate_to_screen(panel, ctx.panel.state.screen, ctx)

        return types.Property(
            panel,
            view=types.GridView(
                align_y="center",
                align_x="center",
                orientation="vertical",
                height=100,
            ),
        )


def register(p):
    p.register(DataQualityPanel)
