# Old Data Quality Panel

A panel for detecting and triaging problematic images in your dataset.

## Installation

Install latest:

```shell
fiftyone plugins download https://github.com/voxel51/old_data_quality_panel
```

Or, development install:

```shell
git clone https://github.com/voxel51/old_data_quality_panel

cd old_data_quality_panel
ln -s "$(pwd)" "$(fiftyone config plugins_dir)/old_data_quality_panel"
```

## Usage

```py
import fiftyone as fo
import fiftyone.zoo as foz

dataset = foz.load_zoo_dataset("quickstart")
session = fo.launch_app(dataset)
```

Then click the `+` next to the `Samples` tab and open the `Old Data Quality`
panel.
