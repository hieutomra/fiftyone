"""
FiftyOne quickstart.

| Copyright 2017-2024, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import fiftyone as fo
import fiftyone.core.context as focx
import fiftyone.core.session as fos
import fiftyone.zoo.datasets as fozd


def quickstart(video=False, port=None, address=None, remote=False):
    """Runs the FiftyOne quickstart.

    This method loads an interesting dataset from the Dataset Zoo, launches the
    App, and prints some suggestions for exploring the dataset.

    Args:
        video (False): whether to launch a video dataset
        port (None): the port number to serve the App. If None,
            ``fiftyone.config.default_app_port`` is used
        address (None): the address to serve the App. If None,
            ``fiftyone.config.default_app_address`` is used
        remote (False): whether this is a remote session, and opening the App
            should not be attempted

    Returns:
        a tuple containing

        -   dataset: the :class:`fiftyone.core.dataset.Dataset` that was loaded
        -   session: the :class:`fiftyone.core.session.Session` instance for
            the App that was launched
    """
    if video:
        return _video_quickstart(port, address, remote)

    return _quickstart(port, address, remote)


def _quickstart(port, address, remote):
    print(_QUICKSTART_GUIDE)
    dataset = fozd.load_zoo_dataset("quickstart")
    return _launch_app(dataset, port, address, remote)


def _video_quickstart(port, address, remote):
    print(_VIDEO_QUICKSTART_GUIDE)
    dataset = fozd.load_zoo_dataset("quickstart-video")
    return _launch_app(dataset, port, address, remote)


def _launch_app(dataset, port, address, remote):
    session = fos.launch_app(
        dataset=dataset,
        port=port,
        address=address,
        remote=remote,
    )

    return dataset, session


_QUICKSTART_GUIDE = """
Welcome to FiftyOne!

This quickstart downloaded a dataset from the Dataset Zoo and created a
session, which is a connection to an instance of the App.

The dataset contains ground truth labels in a `ground_truth` field and
predictions from an off-the-shelf detector in a `predictions` field. It also
has a `uniqueness` field that indexes the dataset by visual uniqueness.

Here are some things you can do to explore the dataset:


(a) Click on an image to explore its labels in more detail


(b) Sort the dataset by uniqueness:

    - Click `add stage`
    - Select the `SortBy` stage
    - Select the `uniqueness` field

    Try setting `reverse` to `True` to show the *most unique* images first.
    Try setting `reverse` to `False` to show the *least unique* images first.


(c) Filter predictions by confidence

    The predictions field is noisy, but you can use FiftyOne to filter them!

    In the filters menu on the left, click on the `v` caret to the
    right of the `predictions` field to open a label filter. Drag the
    confidence slider to only include predictions with confidence at least 0.8!

    You can also filter the detections from Python. Assuming you ran the
    quickstart like this::

        import fiftyone as fo

        dataset, session = fo.quickstart()

    Then you can filter the predictions by creating a view:

        from fiftyone import ViewField as F

        # Create a view that only contains predictions whose confidence is at
        # least 0.8
        high_conf_view = dataset.filter_labels("predictions", F("confidence") > 0.8)

        # Open the view in the App!
        session.view = high_conf_view

Resources:

-   Using the App: https://docs.voxel51.com/user_guide/app.html
-   Dataset Zoo:   https://docs.voxel51.com/user_guide/dataset_zoo/index.html

"""


_VIDEO_QUICKSTART_GUIDE = """
Welcome to FiftyOne!

This quickstart downloaded a dataset from the Dataset Zoo and created a
session, which is a connection to an instance of the App.

The dataset contains small video segments with dense object detections
generated by human annotators.

Here are some things you can do to explore the dataset:


(a) Hover over the videos in the grid view to play their contents

(b) Use the filters menu to toggle and filter detections

(c) Click on a video to open the expanded view, and use the video player
    to scrub through the frames


Resources:

-   Using the App: https://docs.voxel51.com/user_guide/app.html
-   Dataset Zoo:   https://docs.voxel51.com/user_guide/dataset_zoo/index.html

"""
