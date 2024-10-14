import fiftyone.operators as foo
import fiftyone.operators.types as types


class DataQualityPanel(foo.Panel):
    @property
    def config(self):
        return foo.PanelConfig(
            name="data_quality_panel",
            label="Data Quality Issues Panel",
        )

    def on_load(self, ctx):
        pass

    def render(self, ctx):
        panel = types.Object()

        panel.str("Hello World!")

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
