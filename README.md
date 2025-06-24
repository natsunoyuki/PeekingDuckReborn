<br />

<img src="https://raw.githubusercontent.com/natsunoyuki/PeekingDuckReborn/main/docs/source/assets/peekingduck_reborn_new.svg">

---
### Original Documentation
<h4 align="center">
  <a href="https://peekingduck.readthedocs.io/en/stable/getting_started/index.html">Getting started</a>
  <span> · </span>
  <a href="https://peekingduck.readthedocs.io/en/stable/tutorials/index.html">Tutorials</a>
  <span> · </span>
  <a href="https://peekingduck.readthedocs.io/en/stable/master.html#api-documentation">API docs</a>
  <span> · </span>
  <a href="https://peekingduck.readthedocs.io/en/stable/faq.html">FAQ</a>
  <span> · </span>
  <a href="https://github.com/natsunoyuki/PeekingDuckReborn/issues">Report a bug</a>
  <span> · </span>
  <a href="#communities">Communities</a>
</h4>
The links above lead to the original PeekingDuck documentation. The documentation for PeekingDuckReborn is currently a work-in-progress.

---

**PeekingDuckReborn** is the modernized version of [PeekingDuck](https://github.com/aisingapore/PeekingDuck), an open-source, modular framework in Python, built for computer vision (CV) inference, originally developed by [AI Singapore](https://github.com/aisingapore/PeekingDuck). The name "PeekingDuck" is a play on: "Peeking" in a nod to CV; and "Duck" in [duck typing](https://en.wikipedia.org/wiki/Duck_typing). "Reborn" is more straightforward - taking something old and giving it a new breath of life.

## To Do
- [X] Full compatibility with Python3.12+.
- [X] Update dependencies.
- [X] Replace `pkg_resources` with `importlib.metadata`.
- [X] Added BoT-SORT bounding box tracker (without reID) dabble node.
- [X] Added RT-DETR object detector model node.
- [ ] Modernize test suite.
- [ ] Remove/deprecate PeekingDuck Mosse tracker.
- [ ] Remove/deprecate PeekingDuck pose estimation models. 
- [ ] Implement re-ID for BoT-SORT tracker dabble node. 
- [ ] Implement advanced pose estimation models in the models node.
- [ ] Implement multi-GPU support.
- [ ] Implement ONNX deployment for all models in the models node.
- [ ] Implement norfair-sort object tracker.
- [ ] Updated documentation on the internet to replace [the original](https://peekingduck.readthedocs.io/en/stable/index.html#what-is-peekingduck).
- [ ] Updated model weights repository on the internet to replace [the original](https://storage.googleapis.com/peekingduck/models)

## Features
### Build realtime computer vision pipelines
* PeekingDuck enables you to build powerful computer vision pipelines with minimal lines of code.

### Leverage on SOTA models
* PeekingDuck comes with various [object detection](https://peekingduck.readthedocs.io/en/stable/resources/01a_object_detection.html), [pose estimation](https://peekingduck.readthedocs.io/en/stable/resources/01b_pose_estimation.html), [object tracking](https://peekingduck.readthedocs.io/en/stable/resources/01c_object_tracking.html), and [crowd counting](https://peekingduck.readthedocs.io/en/stable/resources/01d_crowd_counting.html) models. Mix and match different nodes to construct solutions for various [use cases](https://peekingduck.readthedocs.io/en/stable/use_cases/index.html).

### Create custom nodes
* You can create [custom nodes](https://peekingduck.readthedocs.io/en/stable/tutorials/02_duck_confit.html#custom-nodes) to meet your own project's requirements. PeekingDuck can also be [imported as a library](https://peekingduck.readthedocs.io/en/stable/tutorials/05_calling_peekingduck_in_python.html) to fit into your existing workflows.


## Installation
### Pip Install from GitHub
Install from [GitHub](https://github.com/natsunoyuki/PeekingDuckReborn) using `pip`

```bash
pip install git+https://github.com/natsunoyuki/PeekingDuckReborn
```

### Local Install (Developer Mode)
Clone the repository from [GitHub](https://github.com/natsunoyuki/PeekingDuckReborn) and install locally in developer mode to implement your customizations.

```bash
git clone https://github.com/natsunoyuki/PeekingDuckReborn
cd PeekingDuckReborn
pip install -e .
```

### Install Options
The following install options are available.
```bash
pip install ".[<install-option>]"
```
1. `test`: Test functionality with `pytest`.

### Verifying the Installation
```bash
peekingduck verify-install
```

You should see a video of a person waving his hand with
[bounding boxes overlaid](https://raw.githubusercontent.com/natsunoyuki/PeekingDuckReborn/main/docs/source/assets/getting_started/verify_install.gif).

The video will close automatically when it is run to the end, select the video window and press `q` to exit earlier.


## Usage
Create a project folder and initialize a PeekingDuck project.
```bash
mkdir <project_dir>
cd <project_dir>
peekingduck init
```

Run the demo pipeline.
```bash
peekingduck run
```

If you have a webcam, you should see a man waving on the output screen with
[skeletal frame overlaid](https://raw.githubusercontent.com/natsunoyuki/PeekingDuckReborn/main/docs/source/assets/getting_started/default_pipeline.gif).

Terminate the program by clicking on the output screen and pressing `q`.

Use `peekingduck --help` to display help information for PeekingDuck's command-line interface.

### Specifying YAML Configuration Files

Run the pipeline with a specified configuration `.yml` file.
```bash
peekingduck run --config_path <path-to-config-yml-file>
```

## Currently Available Nodes
The currently available nodes are listed here. `(pkd)` indicates a node implemented in the original version of PeekingDuck.
### `input` nodes
* `visual (pkd)`
### `augment` nodes
* `brightness (pkd)`
* `contrast (pkd)`
* `undistort (pkd)`
### `model` nodes
* `csrnet (pkd)`
* `efficientdet (pkd)`
* `fairmot (pkd)`
* `hrnet (pkd)`
* `jde (pkd)`
* `mask_rcnn (pkd)`
* `movenet (pkd)`
* `mtcnn (pkd)`
* `posenet (pkd)`
* `rt-detr`
* `yolact_edge (pkd)`
* `yolo_face (pkd)`
* `yolo_license_plate (pkd)`
* `yolo (pkd)`
* `yolox (pkd)`
### `dabble` nodes
* `bbox_count (pkd)`
* `bbox_to_3d_loc (pkd)`
* `bbox_to_btm_midpoint (pkd)`
* `bot_sort`
* `camera_calibration (pkd)`
* `check_large_groups (pkd)`
* `check_nearby_objs (pkd)`
* `fps (pkd)`
* `group_nearby_objs (pkd)`
* `keypoints_to_3d_loc (pkd)`
* `statistics (pkd)`
* `tracking (pkd)`
* `zone_count (pkd)`
### `draw` nodes
* `bbox (pkd)`
* `blur_bbox (pkd)`
* `btm_midpoint (pkd)`
* `group_bbox_and_tag (pkd)`
* `heat_map (pkd)`
* `instance_mask (pkd)`
* `legend (pkd)`
* `mosaic_bbox (pkd)`
* `poses (pkd)`
* `tag (pkd)`
* `zones (pkd)`
### `output` nodes
* `csv_writer (pkd)`
* `media_writer (pkd)`
* `screen (pkd)`


## Acknowledgements
PeekingDuckReborn is an independent derivative of [PeekingDuck](https://github.com/aisingapore/PeekingDuck), which was supported by the National Research Foundation, Singapore under its AI Singapore Programme. 

Any opinions, findings and conclusions or recommendations expressed in this material are those of the authors of PeekingDuckReborn and do not reflect the views of National Research Foundation, Singapore, or of AI Singapore, or of the original authors of [PeekingDuck](https://github.com/aisingapore/PeekingDuck).

PeekingDuckReborn is neither supported nor funded by, nor affiliated with the National Research Foundation, Singapore, or AI Singapore, or the original authors of [PeekingDuck](https://github.com/aisingapore/PeekingDuck).
