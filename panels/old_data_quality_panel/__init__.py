import json
import logging
from collections import Counter
from typing import Callable

import fiftyone as fo
import fiftyone.brain as fob
import fiftyone.core.patches as fop
import fiftyone.operators as foo
import fiftyone.operators.types as types
import fiftyone.zoo.models as fozm
import numpy as np
from fiftyone import ViewField as F
from PIL import Image

from .ctx_inp import _execution_mode, _handle_patch_inputs
from .duplication_operators import (
    ComputeHash,
    gen_approx_duplicate_groups_view,
)
from .image_quality_operators import (
    ComputeAllIssues,
    ComputeAspectRatio,
    ComputeBlurriness,
    ComputeBrightness,
    ComputeEntropy,
    ComputeExposure,
)
from .fix_issue_operators import DeleteSamples, SaveView, TagSamples
from .utils import (
    _convert_opencv_to_pillow,
    _convert_pillow_to_opencv,
    _crop_pillow_image,
    _get_opencv_grayscale_image,
    _get_pillow_patch,
    get_filepath,
)

DEFAULT_ISSUE_TYPE = "summary"
ISSUE_TYPES = [
    "brightness",
    "blurriness",
    "aspect_ratio",
    "entropy",
    "near_duplicates",
    "exact_duplicates",
]
ISSUE_DESCRIPTIONS = {
    "brightness": "Find very dark and very bright images in your dataset",
    "blurriness": "Find blurry images due to motion, defocus, or low quality",
    "aspect_ratio": "Find images with odd aspect ratios",
    "entropy": "Find images with low or high image complexity",
    "near_duplicates": "Find near duplicates in your dataset",
    "exact_duplicates": "Find exact duplicates in your dataset",
}


class OldDataQualityPanel(foo.Panel):
    """Old Data Quality Panel"""

    @property
    def config(self):
        return foo.PanelConfig(
            name="old_data_quality_panel", label="Old Old Data Quality Panel"
        )

    def on_change_view(self, ctx: foo.executor.ExecutionContext):
        """Pass on change view to trigger re-render"""
        # TODO: When the slider under the primitives section on the left hand
        # side-bar is changed, we should update the thresholds
        pass

    def _button(
        self,
        stack: foo.types.Object,
        name: str,
        operator: str,
        label: str,
        operator_method,
        variant="contained",
        color=None,
    ):
        """Adds a button with the given operator and label to the stack"""
        if name == "edit_config":
            icon = "settings"
        else:
            icon = ""
        btns = stack.obj(name)
        btns.type.btn(
            operator,
            label=label,
            on_click=operator_method,
            variant=variant,
            color=color,
            icon=icon,
        )

    def on_load(self, ctx: foo.executor.ExecutionContext):
        """Run on startup"""
        ctx.trigger(
            "split_panel",
            {"name": "old_data_quality_panel", "layout": "horizontal"},
        )
        ctx.panel.state.histogram = None
        ctx.panel.state.layout = None
        ctx.panel.state.hist_lower_thresh = None
        ctx.panel.state.hist_upper_thresh = None
        ctx.panel.state.brain_key = None
        ctx.panel.state.sim_threshold_value = 0.03
        ctx.panel.state.exact_dup_filehashs = []
        # Editable config for the auto detection of quality issues
        ctx.panel.state.issue_config = {
            "brightness": {
                "detect_method": "std_dev",
                "std": 3.0,
                # "handling": "save_view" # if this is set then panel will auto
                # handle the issues without prompting the user
            },
            "blurriness": {
                "detect_method": "threshold",
                "min": None,
                "max": 150,
            },
            "aspect_ratio": {
                "detect_method": "threshold",
                "min": None,
                "max": 0.3,
            },
            "entropy": {
                "detect_method": "percentage",
                "min": None,
                "max": 0.15,
            },
            "near_duplicates": {"sim_threshold": 0.05},
        }
        ctx.panel.state.issue_type = DEFAULT_ISSUE_TYPE
        ctx.panel.set_state("welcome.summary_table", [])

    def on_unload(self, ctx: foo.executor.ExecutionContext):
        """Clear the view after closing panel"""
        ctx.ops.clear_view()

    def set_hist_defaults(self, ctx: foo.executor.ExecutionContext):
        """Sets the default histogram values according to the auto detection method"""
        field = ctx.panel.state.issue_type

        if field not in ctx.dataset.get_field_schema():
            return

        # TODO: Cache these values in a custom run as well?
        (min_v, max_v) = ctx.dataset.bounds(field)
        value_range = max_v - min_v
        config = ctx.panel.state.issue_config[ctx.panel.state.issue_type]
        # TODO: A user may want to have multiple range thresholds for an issue
        # but right now only a single one is supported.
        if config["detect_method"] == "threshold":
            if ctx.panel.state.hist_lower_thresh is None:
                lower_thresh = config["min"] if config["min"] else min_v
                upper_thresh = config["max"] if config["max"] else max_v
                ctx.panel.state.hist_lower_thresh = lower_thresh
                ctx.panel.state.hist_upper_thresh = upper_thresh
                default_slider_vals = [lower_thresh, upper_thresh]
        elif config["detect_method"] == "percentage":
            if ctx.panel.state.hist_lower_thresh is None:
                lower_thresh = (
                    (min_v + (config["min"] * value_range))
                    if config["min"]
                    else min_v
                )
                upper_thresh = (
                    (min_v + (config["max"] * value_range))
                    if config["max"]
                    else max_v
                )
                ctx.panel.state.hist_lower_thresh = lower_thresh
                ctx.panel.state.hist_upper_thresh = upper_thresh
                default_slider_vals = [lower_thresh, upper_thresh]
        elif config["detect_method"] == "std_dev":
            if ctx.panel.state.hist_lower_thresh is None:
                std = ctx.dataset.std(field)
                mean = ctx.dataset.mean(field)
                ctx.panel.state.hist_lower_thresh = mean - (
                    config["std"] * std
                )
                ctx.panel.state.hist_upper_thresh = mean + (
                    config["std"] * std
                )
                default_slider_vals = [
                    ctx.panel.state.hist_lower_thresh,
                    ctx.panel.state.hist_upper_thresh,
                ]
        ctx.panel.state.default_slider_vals = default_slider_vals

    def tab_change(self, ctx: foo.executor.ExecutionContext):
        """Changes which Old Data Quality issue to view"""
        # BUG: Why does this error pop up at the bottom even though nothing is wrong?
        # object of type 'NoneType' has no len() (operation: @voxel51/data_quality/old_data_quality_panel#tab_change)
        ctx.panel.state.issue_type = ctx.params["value"] or DEFAULT_ISSUE_TYPE
        ctx.panel.state.hist_lower_thresh = None
        ctx.panel.state.hist_upper_thresh = None
        ctx.panel.state.show_config = False

        if ctx.panel.state.issue_type not in [
            "near_duplicates",
            "exact_duplicates",
        ]:
            # Set the histogram and slider defaults based on auto detection method
            self.set_hist_defaults(ctx)

        # Change view based on new issue type
        view_change = self.change_view(ctx, ctx.panel.state.issue_type)

    def slider_change(self, ctx: foo.executor.ExecutionContext):
        """Changes the x-axis bounds on the histograms"""
        ctx.panel.state.hist_lower_thresh = ctx.params["value"][0]
        ctx.panel.state.hist_upper_thresh = ctx.params["value"][1]

        # Change view based on new thresholds
        self.change_view(ctx, ctx.panel.state.issue_type)

    def hist_select(self, ctx: foo.executor.ExecutionContext):
        """Selects a range of values from the histogram"""
        selected_x_value_indices = [x["idx"] for x in ctx.params["data"]]
        selected_x_values = [
            ctx.panel.state.histogram[0]["x"][i]
            for i in selected_x_value_indices
        ]
        ctx.panel.state.hist_lower_thresh = min(selected_x_values)
        ctx.panel.state.hist_upper_thresh = max(selected_x_values)

        # Change view based on new thresholds
        self.change_view(ctx, ctx.panel.state.issue_type)

    def get_histogram(
        self,
        stack: foo.types.Object,
        ctx: foo.executor.ExecutionContext,
        field: str,
    ):
        """Adds a histogram and selection sliders to the panel"""
        method = "old_data_quality_panel"
        run_key = f"{field}"

        if run_key in ctx.dataset.list_runs(method=method):
            data = ctx.dataset.load_run_results(run_key=run_key)
            counts = data.counts
            edges = data.edges
        else:
            config = ctx.dataset.init_run(method=method, foo="bar")
            ctx.dataset.register_run(run_key, config)
            counts, edges, _ = ctx.dataset.histogram_values(field, bins=50)
            results = ctx.dataset.init_run_results(
                method=method, run_key=run_key, counts=counts, edges=edges
            )
            ctx.dataset.save_run_results(run_key, results)

        (min_v, max_v) = ctx.dataset.bounds(field)

        histogram_data = [
            {
                "x": edges,
                "y": counts,
                "type": "bar",
                "marker": {"color": "orange"},
            }
        ]

        # Auto-detect issues
        default_slider_vals = ctx.panel.state.default_slider_vals

        config = {"scrollZoom": False}
        layout = {
            "width": 400,
            "xaxis": {
                "title": f"{field}",
                "tickmode": "auto",
                "nticks": 10,
                "ticks": "inside",
            },
            "yaxis": {
                "title": "Count",
            },
            "title": "Blurriness Histogram",
            "dragmode": "select",
            "selectdirection": "h",
        }

        ctx.panel.state.histogram = histogram_data
        ctx.panel.state.layout = layout

        # Bar Chart - Histogram
        stack.plot(
            "histogram",
            data=ctx.panel.state.histogram,
            layout=ctx.panel.state.layout,
            config=config,
            on_selected=self.hist_select,
        )

        # Double Slider
        # BUG: Slider calls on_change more times than ideal still
        hist_lower_thresh = ctx.panel.state.hist_lower_thresh
        hist_upper_thresh = ctx.panel.state.hist_upper_thresh
        if hist_lower_thresh is not None:
            default_slider_vals = [
                hist_lower_thresh,
                hist_upper_thresh,
            ]  ######
        stack.list(
            f"double_slider_{field}",
            types.Number(),
            on_change=self.slider_change,
            view=types.SliderView(
                label="Choose range",  # BUG: Not including a label causes this component to not show up
                componentsProps={
                    "slider": {
                        "defaultValue": default_slider_vals,
                        "min": min_v,
                        "max": max_v,
                        "step": 0.01,
                        "key": f"{str(hist_lower_thresh)}_{str(hist_upper_thresh)}",
                    },
                    "container": {
                        "sx": {"width": "400px"}
                    },  # TODO: This is not ideal
                },
            ),
            # width=50 BUG: This does not work?
        )

    def change_view(
        self, ctx: foo.executor.ExecutionContext, quality_issue: str
    ):
        """Filters the view based on the histogram bounds"""
        if (
            ctx.panel.state.hist_lower_thresh
            or ctx.panel.state.hist_upper_thresh
        ):
            # Filter the view to be between hist_lower_thresh and hist_upper_thresh
            view = ctx.dataset.filter_field(
                quality_issue, F() < ctx.panel.state.hist_upper_thresh
            ).filter_field(
                quality_issue, F() > ctx.panel.state.hist_lower_thresh
            )
            num_issue_samples = len(view)

            # BUG: For some reason using set_state causes an infinite render loop
            # Using set_data works but shouldn't. Need to investigate
            table_data = ctx.panel.get_state("welcome.summary_table") or []
            found = False
            for i, d in enumerate(table_data):
                if d["Category"] == quality_issue:
                    table_data[i] = {
                        "Category": quality_issue,
                        "Count": num_issue_samples,
                    }
                    found = True
            if not found:
                table_data.append(
                    {"Category": quality_issue, "Count": num_issue_samples}
                )
            ctx.panel.set_data("welcome.summary_table", table_data)
            if len(view) == 0:
                return False
            else:
                ctx.ops.set_view(view)
                return True
        else:
            # Set view to entire dataset
            if quality_issue == "exact_duplicates":
                if "filehash" in ctx.dataset.get_field_schema():
                    exact_dup_view = ctx.dataset.match(
                        F("filehash").is_in(
                            ctx.panel.state.exact_dup_filehashs
                        )
                    ).sort_by("filehash")
                    ctx.ops.set_view(exact_dup_view)
            elif quality_issue == "near_duplicates":
                ctx.ops.set_view(ctx.dataset.view())

    def check_new_samples(
        self, ctx: foo.executor.ExecutionContext, stack, method: Callable
    ):
        """Checks if samples don't have a quality issue computed"""
        issue_type = ctx.panel.state.issue_type
        if issue_type in ctx.dataset.get_field_schema():
            not_labeled_samples = len(
                ctx.dataset.select_by(issue_type, [None])
            )
            if not_labeled_samples != 0:
                hstack = stack.v_stack(
                    "new_sample_buttons", gap=0, align_y="center"
                )
                # Trigger alert
                hstack.str(
                    "new_sample",
                    label=f"You have new samples without {issue_type} computed",
                    view=types.AlertView(variant="filled", severity="warning"),
                )
                self._button(hstack, issue_type, issue_type, "Execute", method)

    def show_exact_dups(self, ctx: foo.executor.ExecutionContext):
        """Sets the view to show all exact duplicates"""
        exact_dup_view = ctx.dataset.match(
            F("filehash").is_in(ctx.panel.state.exact_dup_filehashs)
        ).sort_by("filehash")
        ctx.ops.set_view(exact_dup_view)

    def brain_change(self, ctx: foo.executor.ExecutionContext):
        """Selects the brain key for similarity computation"""
        ctx.panel.state.brain_key = ctx.params["value"]

    def sim_threshold_change(self, ctx: foo.executor.ExecutionContext):
        """Changes the threshold for similarity computation"""
        ctx.panel.state.sim_threshold_value = ctx.params["value"]

        # Change config
        config = ctx.panel.state.issue_config
        config["near_duplicates"]["sim_threshold"] = ctx.params["value"]
        ctx.panel.state.issue_config = config

    def deduplicate(self, ctx: foo.executor.ExecutionContext):
        """Deduplicates the dataset by removing all but 1 sample per similar group"""
        ctx.ops.set_view(name="approx_dup_groups_view")

        remove_samples_ids = []
        for group_id in ctx.view.distinct("approx_dup_group_id"):
            group_view = ctx.view.match(F("approx_dup_group_id") == group_id)
            # Remove all but the first sample
            remove_samples_ids.extend(group_view.values("id")[1:])

        ctx.dataset.delete_samples(remove_samples_ids)
        ctx.dataset.delete_saved_view("approx_dup_groups_view")
        ctx.ops.set_view(ctx.dataset.view())

    def show_near_dups(self, stack, ctx: foo.executor.ExecutionContext):
        """Adds inputs for near duplicate computation to the stack"""
        # Show dropdown of available brain keys / embeddings
        brain_runs = ctx.dataset.list_brain_runs()
        brain_choices = types.Dropdown()
        for run in brain_runs:
            if ctx.dataset.get_brain_info(run).config.type == "similarity":
                brain_choices.add_choice(run, label=run)

        if len(brain_runs) == 0:
            default_str = (
                "No similarity brain keys found. Compute similarity first."
            )
        else:
            default_str = brain_runs[0]
            ctx.panel.state.brain_key = default_str
        # TODO: Want to make the default choice when no brain runs are listed
        # look gray or have an opacity to indicate it's not selectable
        stack.enum(
            "brain_key",
            brain_choices.values(),
            label="Select Brain Key",
            description="Select the brain key to use for similarity computation",
            default=default_str,
            view=brain_choices,
            on_change=self.brain_change,
        )

        # Slider for controllable threshold
        # TODO: Can we place the label to the left of the input?
        cfg = ctx.panel.state.issue_config["near_duplicates"]
        stack.float(
            "sim_threshold_value",
            default=cfg["sim_threshold"],
            label="Distance Threshold",
            on_change=self.sim_threshold_change,
            description="Select the distance threshold for determining approximate duplicates",
        )

        # Find near duplicates
        if ctx.panel.state.brain_key is not None:
            self._button(
                stack,
                "near_dups",
                "near_dups",
                "Show near duplicates",
                self.compute_near_dups,
            )
        # Button to delete or deduplicate near dups
        if "approx_dup_groups_view" in ctx.dataset.list_saved_views():
            self._button(
                stack,
                "deduplicate",
                "deduplicate",
                "Deduplicate",
                self.deduplicate,
            )

    def edit_config(self, ctx: foo.executor.ExecutionContext):
        """Set state to show editable config"""
        ctx.panel.set_state("show_config", True)

    def update_config(self, ctx: foo.executor.ExecutionContext):
        """Update the issue config based on the JSON input"""
        ctx.panel.state.issue_config = json.loads(ctx.params["value"])

    def threshold_update_config(self, ctx: foo.executor.ExecutionContext):
        """Update the issue config based on the threshold values"""
        issue_type = ctx.panel.state.issue_type
        config = ctx.panel.state.issue_config
        config[issue_type] = {}
        config[issue_type]["detect_method"] = "threshold"
        config[issue_type]["min"] = ctx.panel.state.hist_lower_thresh
        config[issue_type]["max"] = ctx.panel.state.hist_upper_thresh
        ctx.panel.state.issue_config = config

    def close_config(self, ctx: foo.executor.ExecutionContext):
        """Set state to hide editable config"""
        ctx.panel.state.show_config = False

    def check_config_json(self, config):
        """Check if the JSON config is formatted correctly"""
        try:
            # Make sure JSON is formatted correctly
            cnf = json.dumps(config)
            cnf = config

            # Make sure there is a key for each issue type
            for key in cnf.keys():
                if key not in [
                    "brightness",
                    "blurriness",
                    "aspect_ratio",
                    "entropy",
                    "near_duplicates",
                    "exact_duplicates",
                ]:
                    return False
            for key in ["brightness", "blurriness", "aspect_ratio", "entropy"]:
                if key not in cnf:
                    return False

                # Make sure each issue type has a detect method
                if "detect_method" not in cnf[key]:
                    return False

                # Make sure each issue type has a valid detect method
                if cnf[key]["detect_method"] not in [
                    "std_dev",
                    "threshold",
                    "percentage",
                ]:
                    return False

                # Make sure each issue type has required fields
                if cnf[key]["detect_method"] == "std_dev":
                    if "std" not in cnf[key]:
                        return False
                elif cnf[key]["detect_method"] == "threshold":
                    if "min" not in cnf[key] or "max" not in cnf[key]:
                        return False
                elif cnf[key]["detect_method"] == "percentage":
                    if "min" not in cnf[key] or "max" not in cnf[key]:
                        return False
            return True
        except json.JSONDecodeError:
            return False

    def show_issue_tab(self, stack, ctx, issue_type):
        """Shows the content of the issues tabs"""
        if issue_type == "brightness":
            issue_method = self.compute_brightness
        elif issue_type == "blurriness":
            issue_method = self.compute_blur
        elif issue_type == "aspect_ratio":
            issue_method = self.compute_aspect_ratio
        elif issue_type == "entropy":
            issue_method = self.compute_entropy

        if issue_type in ctx.dataset.get_field_schema():
            ctx.ops.set_active_fields(fields=[issue_type])
        else:
            vstack = stack.v_stack(
                f"{issue_type}_welcome", align_y="center", pad_t=40
            )
            vstack.md(
                f"""##### {issue_type.capitalize()} Issues""",
                name=f"{issue_type}_title",
            )
            self._button(
                vstack,
                f"compute_{issue_type}",
                f"compute_{issue_type}",
                "Execute",
                issue_method,
            )
        # Check and notify if new samples w/o issue computed are found
        self.check_new_samples(ctx, stack, issue_method)

        if issue_type in ctx.dataset.get_field_schema():
            hstack = stack.h_stack(
                f"{issue_type}_tab",
                componentsProps={
                    "grid": {"sx": {"flexWrap": "unset"}},
                    "item": {"sx": {"overflow": "unset"}},
                },
            )
            vstack1 = hstack.v_stack(f"{issue_type}_histogram", align_y="left")
            vstack2 = hstack.v_stack(f"{issue_type}_buttons", align_y="center")

            vstack2.str(
                "issue_type",
                # label=f"{issue_type.capitalize()} Issues",
                description=ISSUE_DESCRIPTIONS[issue_type],
                caption="caption",
                view=types.AlertView(variant="filled", severity="info"),
            )

            self.get_histogram(vstack1, ctx, issue_type)

            # Change config based on thresholds
            self._button(
                vstack1,
                "update_config",
                "update_config",
                "Save Threshold",
                self.threshold_update_config,
                variant="outlined",
                color="51",
            )

            # Display # of issues
            # BUG: This is not working, and is most likely related to the issue
            # with the setting of welcome.summary_table
            # for i,d in enumerate(ctx.panel.get_state('welcome.summary_table')):
            #     if d["Category"] == issue_type:
            #         num_issue_samples = d["Count"]
            # stack.md(
            #     f"Found {num_issue_samples} samples with brightness issues",
            #     name="num_issues_{issue_type}",
            # )
            # Save, delete, or tag filtered samples
            self.filter_buttons(vstack2, ctx)

    def show_all_issues(self, ctx: foo.executor.ExecutionContext):
        """Sets the view to show all issues"""
        summary_table = ctx.panel.get_state("welcome.summary_table") or []
        issue_views = []
        for issue_type in ISSUE_TYPES:
            if issue_type not in ctx.dataset.get_field_schema():
                continue
            (min_v, max_v) = ctx.dataset.bounds(issue_type)
            value_range = max_v - min_v
            # Get saved thresholds from config
            issue_config = ctx.panel.state.issue_config[issue_type]
            if issue_config["detect_method"] == "threshold":
                lower_thresh = (
                    issue_config["min"] if issue_config["min"] else min_v
                )
                upper_thresh = (
                    issue_config["max"] if issue_config["max"] else max_v
                )
            elif issue_config["detect_method"] == "percentage":
                lower_thresh = (
                    (min_v + (issue_config["min"] * value_range))
                    if issue_config["min"]
                    else min_v
                )
                upper_thresh = (
                    (min_v + (issue_config["max"] * value_range))
                    if issue_config["max"]
                    else max_v
                )
            elif issue_config["detect_method"] == "std_dev":
                if ctx.panel.state.hist_lower_thresh is None:
                    std = ctx.dataset.std(issue_type)
                    mean = ctx.dataset.mean(issue_type)
                    lower_thresh = mean - (issue_config["std"] * std)
                    upper_thresh = mean + (issue_config["std"] * std)
            issue_view = ctx.dataset.filter_field(
                issue_type, F() < upper_thresh
            ).filter_field(issue_type, F() > lower_thresh)
            issue_views.append(issue_view)
        # Combine all issue views
        if len(issue_views) > 0:
            combined_view = issue_views[0]
            for view in issue_views[1:]:
                combined_view = combined_view.concat(view)
            ctx.ops.set_view(combined_view)

    def show_summary_tab(self, stack, ctx):
        """Shows the content of the summary tab"""
        # Create table of existing issues detected
        table = types.TableView()
        table.add_column("Category", label="category")
        table.add_column("Count", label="count")

        stack.list(
            "summary_table",
            types.Object(),
            view=table,
            label="Summary of Issues",
        )

        # Check if any issue types are present in the schema
        #     if yes then show table of fields and found issues
        #     if no then show execute all button
        self._button(
            stack,
            "execute_all",
            "execute_all",
            "Execute all",
            self.compute_all,
        )
        # Show all issues
        self._button(
            stack,
            "show_all_issues",
            "show_all_issues",
            "Show all issues",
            self.show_all_issues,
        )

    def show_exact_duplicates_tab(self, stack, ctx):
        stack.md(
            """
            #### Exact Duplicates
                    """
        )
        self._button(
            stack,
            "compute_hash",
            "compute_hash",
            "Compute Hash",
            self.compute_hash,
            variant="outlined",
            color="51",
        )
        # Show exact duplicates
        if "filehash" in ctx.dataset.get_field_schema():
            exact_dups = ctx.panel.state.exact_dup_filehashs
            num_dups = len(exact_dups)
            if len(exact_dups) > 0:
                stack.md(
                    f"**{num_dups} exact duplicates found**",
                    name="no_exact_dups",
                )
            else:
                stack.md("**No exact duplicates found**", name="no_exact_dups")

    def show_near_duplicates_tab(self, stack, ctx):
        """Shows the content of the duplicates tab"""
        stack.md(
            """
            This method will find exact and near duplicates in your
            dataset. Exact duplicates will be computed using a hash
            function which should give a unique hash per image. Near
            duplicates will be measured according to your specified
            embedding model.
        """,
            name="similarity_description",
        )
        stack.md(
            """
            #### Near Duplicates
                    """
        )
        # TODO - Use this version to automate the near duplicate comp
        # stack.btn(
        #     "compute_similarity",
        #     label="Compute Similarity",
        #     #on_click=fob.compute_similarity,
        #     on_click="@voxel51/brain/compute_similarity",
        #     params={
        #         #"samples": ctx.dataset,
        #         "embeddings": "clip_embeddings",
        #         "brain_key": "old_data_quality_panel_similarity",
        #         "model": "clip-vit-base32-torch",
        #         "batch_size": 8,
        #         "backend": "sklearn",
        #         "metric": "cosine",
        #         "progress": True
        #     },
        #     prompt=False)
        self._button(
            stack,
            "compute_similarity",
            "compute_similarity",
            "Compute Similarity",
            self.compute_sim,
            variant="outlined",
            color="51",
        )

        # Show near duplicates
        self.show_near_dups(stack, ctx)

    def show_tab(self, stack, ctx, issue_type):
        if issue_type == "summary":
            self.show_summary_tab(stack, ctx)
        elif issue_type == "near_duplicates":
            self.show_near_duplicates_tab(stack, ctx)
        elif issue_type == "exact_duplicates":
            self.show_exact_duplicates_tab(stack, ctx)
        else:
            self.show_issue_tab(stack, ctx, issue_type)

    def render(self, ctx: foo.executor.ExecutionContext):
        panel = types.Object()
        stack = panel.v_stack(
            "welcome",
            gap=2,
            width=80,
            pad_b=2,
            pad_t=2,
            componentsProps={
                "grid": {"sx": {"flexWrap": "unset"}},
                "item": {"sx": {"overflow": "unset"}},
            },
        )
        if ctx.panel.state.show_config:
            json_str = json.dumps(ctx.panel.state.issue_config, indent=4)
            valid_json = self.check_config_json(ctx.panel.state.issue_config)
            stack.str(
                "src",
                default=json_str,
                on_change=self.update_config,
                view=types.CodeView(language="python"),
            )
            if not valid_json:
                stack.md("Invalid JSON configuration", name="invalid_json")
            else:
                self._button(
                    stack,
                    "save_config",
                    "save_config",
                    "Save Config",
                    self.close_config,
                    variant="outlined",
                    color="51",
                )
        else:
            hstack = stack.h_stack("buttons", align_x="space-between")
            self._button(
                hstack,
                "edit_config",
                "edit_config",
                "Edit Config",
                self.edit_config,
                variant="outlined",
                color="51",
            )

            tabs = types.TabsView()
            tabs.add_choice("summary", label="Summary")
            tabs.add_choice("exact_duplicates", label="Exact Duplicates")
            tabs.add_choice("near_duplicates", label="Near Duplicates")
            tabs.add_choice("brightness", label="Brightness")
            tabs.add_choice("blurriness", label="Blurriness")
            tabs.add_choice("aspect_ratio", label="Aspect Ratio")
            tabs.add_choice("entropy", label="Entropy")

            stack.str("tabs", view=tabs, on_change=self.tab_change)

            issue_type = ctx.panel.state.issue_type
            if ctx.panel.state.welcome:
                self.show_tab(stack, ctx, issue_type)
        return types.Property(
            panel, view=types.GridView(width=100, align_x="center")
        )

    ################## Post Processing Quality Issues ###################
    def filter_buttons(
        self, stack: foo.types.Object, ctx: foo.executor.ExecutionContext
    ):
        """Adds buttons to save, tag, or delete samples with quality issues"""

        issue_type = ctx.panel.state.issue_type

        button_stack = stack.v_stack("filter_buttons", gap=2, align_x="left")
        button_stack.md(f"Reference it later", name="save_view_btn")
        btns = button_stack.obj("save_view")
        btns.type.btn(
            "save_view",
            label="Save View",
            on_click=self.button_save_view,
            variant="outlined",
            componentProps={"sx": {"width": "30px"}},
        )
        # Tag samples
        button_stack.md(f"Annotate", name="tag_samples_btn")
        btns3 = button_stack.obj("tag_samples")
        btns3.type.btn(
            "tag_samples",
            label="Tag Samples",
            on_click=self.button_tag_samples,
            variant="outlined",
        )
        # Delete samples
        button_stack.md(f"Clean up", name="delete_samples_btn")
        btns2 = button_stack.obj("delete_samples")
        btns2.type.btn(
            "delete_samples",
            label="Delete Samples",
            on_click=self.button_delete_samples,
            variant="outlined",
        )

    def button_tag_samples(self, ctx: foo.executor.ExecutionContext):
        """Button operator for tagging samples"""
        ctx.ops.track_event("old_data_quality_panel_tag_samples")
        ctx.prompt("@voxel51/data_quality/tag_samples")
        # BUG: Why does it say "No operator provided for panel event"?
        # TODO: Set selection of tags to the new tags

    def button_save_view(self, ctx: foo.executor.ExecutionContext):
        """Button operator for saving a view"""
        ctx.ops.track_event("old_data_quality_panel_save_view")
        ctx.prompt("@voxel51/data_quality/save_view", on_success=self.reload)

    def reset_view(self, ctx: foo.executor.ExecutionContext):
        ctx.ops.set_view(ctx.dataset.view())

    def button_delete_samples(self, ctx: foo.executor.ExecutionContext):
        """Button operator for deleting samples"""
        ctx.ops.track_event("old_data_quality_panel_delete_samples")
        ctx.prompt(
            "@voxel51/data_quality/delete_samples", on_success=self.reset_view
        )

    ################## Image Deduplication Computation ###################
    def load_exact_dups(self, ctx: foo.executor.ExecutionContext):
        filehash_counts = Counter(sample.filehash for sample in ctx.dataset)
        dup_filehashes = [k for k, v in filehash_counts.items() if v > 1]
        ctx.panel.state.exact_dup_filehashs = dup_filehashes

        exact_dup_view = ctx.dataset.match(
            F("filehash").is_in(ctx.panel.state.exact_dup_filehashs)
        ).sort_by("filehash")
        ctx.ops.set_view(exact_dup_view)

    def compute_hash(self, ctx: foo.executor.ExecutionContext):
        ctx.ops.track_event("old_data_quality_panel_compute_hash")
        ctx.prompt(
            "@voxel51/data_quality/compute_hash",
            on_success=self.load_exact_dups,
        )

    def reload(self, ctx: foo.executor.ExecutionContext):
        ctx.ops.reload_dataset()

    def compute_sim(self, ctx: foo.executor.ExecutionContext):
        ctx.ops.track_event("old_data_quality_panel_compute_similarity")
        ctx.prompt(
            "@voxel51/brain/compute_similarity",
            params={
                "embeddings": "clip_embeddings",
                "brain_key": "old_data_quality_panel_similarity",
                "model": "clip-vit-base32-torch",
                "batch_size": 8,
                "backend": "sklearn",
                "progress": True,
            },
            on_success=self.reload,
        )

    def compute_near_dups(self, ctx: foo.executor.ExecutionContext):
        # Make sure to recompute if there is a different threshold selected
        threshold = ctx.panel.state.sim_threshold_value
        view_name = f"near_dup_groups_view_{threshold}"
        if view_name not in ctx.dataset.list_saved_views():
            index = ctx.dataset.load_brain_results(ctx.panel.state.brain_key)
            index.find_duplicates(thresh=threshold)

            near_dup_view, group_view = gen_approx_duplicate_groups_view(
                ctx, index
            )

            ctx.dataset.save_view(
                f"near_dup_view_{threshold}", near_dup_view, overwrite=True
            )
            ctx.dataset.save_view(view_name, group_view, overwrite=True)
            ctx.ops.reload_dataset()
        ctx.ops.set_view(name=view_name)

    ################### Image Quality Issue Computation ###################
    def compute_brightness(self, ctx: foo.executor.ExecutionContext):
        ctx.ops.track_event("compute_brightness")
        ctx.prompt("@voxel51/data_quality/compute_brightness")

    def compute_aspect_ratio(self, ctx: foo.executor.ExecutionContext):
        ctx.ops.track_event("data_quality_compute_aspect_ratio")
        ctx.prompt("@voxel51/data_quality/compute_aspect_ratio")

    def compute_entropy(self, ctx: foo.executor.ExecutionContext):
        ctx.ops.track_event("data_quality_compute_entropy")
        ctx.prompt("@voxel51/data_quality/compute_entropy")

    def compute_blur(self, ctx: foo.executor.ExecutionContext):
        ctx.ops.track_event("data_quality_compute_blurriness")
        ctx.prompt("@voxel51/data_quality/compute_blurriness")

    def compute_all(self, ctx: foo.executor.ExecutionContext):
        ctx.ops.track_event("data_quality_compute_all_issues")
        ctx.prompt("@voxel51/data_quality/compute_all_issues")


def register(p):
    p.register(OldDataQualityPanel)
    p.register(ComputeBrightness)
    p.register(ComputeBlurriness)
    p.register(ComputeEntropy)
    p.register(ComputeAspectRatio)
    p.register(ComputeExposure)
    p.register(DeleteSamples)
    p.register(TagSamples)
    p.register(SaveView)
    p.register(ComputeHash)
    p.register(ComputeAllIssues)
