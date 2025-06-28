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
- [X] Implement advanced pose estimation models (VITPose) in the models node.
- [ ] Modernize tests (WIP).
- [ ] Remove/deprecate PeekingDuck Mosse tracker.
- [ ] Remove/deprecate PeekingDuck pose estimation models. 
- [ ] Implement re-ID for BoT-SORT tracker dabble node. 
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

### Windows with CUDA
Before installing PeekingDuckReborn, install torch with CUDA first following the <a href="https://pytorch.org/get-started/locally/">official PyTorch instructions</a>. If not, torch will only be able to access the CPU. Tensorflow 2.11 and newer do not support CUDA on Windows.
```bash
# Install torch with CUDA first.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
# Then install PeekingDuckReborn.
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

### Local Model Weights Directory
The original PeekingDuck model weights will be downloaded from https://storage.googleapis.com/peekingduck/models to a local subdirectory, which is set to `peekingduck_weights/` by default in PeekingDuckReborn. The name of `peekingduck_weights/` will depend if local environment variables are specified using a `.env` file.

For normal installations, the original PeekingDuck model weights will be downloaded to `PeekingDuckReborn/venv/Lib/site-packages/peekingduck_weights/`, and when installed in developer mode, they will be downloaded to `PeekingDuckReborn/peekingduck_weights/` by default if no local environment variables are set. Torchvision and HuggingFace model weights will be downloaded to the local cache directory. 

#### Specifying Local Environment Variables with a `.env` File.
Create a `.env` file under `PeekingDuckReborn/` with the following contents to specify another subdirectory name instead of `peekingduck_weights/`.
```
PEEKINGDUCK_WEIGHTS_SUBDIR=<subdirectory name>
```

#### `peekingduck_weights/` Structure
In general, `peekingduck_weights/` will have the following structure:
```
peekingduck_weights/
└───model_name_1/
        ├───pytorch/
        |       ├─── model_type_1/
        |       └─── model_type_2/
        └───tensorflow/
                ├─── model_type_1/
                └─── model_type_2/
...
```
where `model_name` corresponds to the node name, e.g. `yolox`, while `model_type` corresponds to `model_type` in the corresponding node, e.g. `yolox-s`. While most of the models have pytorch weights, some models have tensorflow weights. We follow the original convention set in PeekingDuck and separate the weights according to whether they are written in pytorch or tensorflow.

HuggingFace and Torchvision weights can also be manually placed under `peekingduck_weights/`. For example, for the `rt_detr` model, we can manually download the <a href="https://huggingface.co/PekingU/rtdetr_r18vd/tree/main">`rtdetr_r18vd/` weights from HuggingFace</a> and store them under `PeekingDuckReborn/peekingduck_weights/`.

```
peekingduck_weights/
└───rt_detr/
        └───pytorch/
                └───rtdetr_r18vd/
                        ├───config.json
                        ├───model.safetensors
                        └───preprocessor_config.json
...
```


## Currently Available Nodes
The currently available nodes are listed here. Nodes include both original PeekingDuck nodes, and PeekingDuckReborn nodes. `(pkd)` indicates a node inherited from the original version of PeekingDuck. For the usage options, please refer to the YAML configuration files in `peekingduck/configs/`. More detailed documentation is a work-in-progress.

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
* `hrnet (pkd)` - Buggy original PKD implementation. Should not be used.
* `jde (pkd)`
* `mask_rcnn (pkd)`
* `movenet (pkd)` - Buggy original PKD implementation. Should not be used.
* `mtcnn (pkd)`
* `posenet (pkd)` - Buggy original PKD implementation. Should not be used.
* `rt-detr`
* `vit_pose`
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


## Important Things to Note
### Images Are Numpy Arrays with BGR Channels
In the pipeline, images are `numpy` arrays with shape `[H, W, C]`, where `C` corresponds to Blue-Green-Red channels following OpenCV convention. Please keep this in mind when implementing new nodes or tests.

### Buggy Original Pose Estimation Models
The original PeekingDuck pose estimation models `hrnet`, `movenet` and `posenet` are extremely buggy and should not be used.


## Acknowledgements
PeekingDuckReborn is an independent derivative of [PeekingDuck](https://github.com/aisingapore/PeekingDuck), which was supported by the National Research Foundation, Singapore under its AI Singapore Programme. 

Any opinions, findings and conclusions or recommendations expressed in this material are those of the authors of PeekingDuckReborn and do not reflect the views of National Research Foundation, Singapore, or of AI Singapore, or of the original authors of [PeekingDuck](https://github.com/aisingapore/PeekingDuck).

PeekingDuckReborn is neither supported nor funded by, nor affiliated with the National Research Foundation, Singapore, or AI Singapore, or the original authors of [PeekingDuck](https://github.com/aisingapore/PeekingDuck).
