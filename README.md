<br />

<img src="https://raw.githubusercontent.com/natsunoyuki/PeekingDuckReborn/main/docs/source/assets/peekingduck_reborn_new.svg">

---
### PeekingDuckReborn Wiki Documentation
<h4 align="center">
  <a href="https://github.com/Natsunoyuki-AI-Laboratory/PeekingDuckReborn/wiki">Wiki</a>
  <span> · </span>
  <a href="https://github.com/Natsunoyuki-AI-Laboratory/PeekingDuckReborn/wiki/Installation">Installation</a>
  <span> · </span>
  <a href="https://github.com/Natsunoyuki-AI-Laboratory/PeekingDuckReborn/wiki/PeekingDuckReborn-Nodes">API</a>
  <span> · </span>
  <a href="https://github.com/Natsunoyuki-AI-Laboratory/PeekingDuckReborn/wiki/Inference-with-PeekingDuckReborn">Usage</a>
</h4>


---

**PeekingDuckReborn** is the modernized version of [PeekingDuck](https://github.com/aisingapore/PeekingDuck), an open-source, modular framework in Python, built for computer vision (CV) inference, originally developed by [AI Singapore](https://github.com/aisingapore/PeekingDuck). The name "PeekingDuck" is a play on: "Peeking" in a nod to CV; and "Duck" in [duck typing](https://en.wikipedia.org/wiki/Duck_typing). "Reborn" is more straightforward - taking something old and giving it a new breath of life.

## To Do
- [X] Full compatibility with Python3.12+.
- [X] Update dependencies.
- [X] Fix issue regarding the local weights subdirectory and parent directory with `dotenv`.
- [X] Replace `pkg_resources` with `importlib.metadata`.
- [X] Implement BoT-SORT bounding box tracker (without reID) dabble node.
- [X] Implement RT-DETR object detector model node.
- [X] Implement VITPose pose keypoint detection model node.
- [X] Create [PeekingDuckReborn wiki](https://github.com/Natsunoyuki-AI-Laboratory/PeekingDuckReborn/wiki) documentation to replace [the original](https://peekingduck.readthedocs.io/en/stable/index.html#what-is-peekingduck).
- [ ] Modernize tests (WIP).
- [ ] Fix issues involving TensorFlow on Windows GPU.
- [ ] Remove/deprecate PeekingDuck Mosse tracker.
- [ ] Remove/deprecate PeekingDuck pose estimation models (HRNet, MoveNet, PoseNet). 
- [ ] Implement re-ID for BoT-SORT tracker dabble node. 
- [ ] Update model weights repository on the internet to replace [the original](https://storage.googleapis.com/peekingduck/models)

## Features
### Build realtime computer vision pipelines
* Use PeekingDuckReborn to develop custom computer vision pipelines with minimal lines of code.

### Leverage on SOTA models
* PeekingDuckReborn comes with powerful models such as the <a href="https://github.com/Natsunoyuki-AI-Laboratory/PeekingDuckReborn/wiki/Model-Nodes#rt_detr">RT-DETR object detection model</a>, <a href="https://github.com/Natsunoyuki-AI-Laboratory/PeekingDuckReborn/wiki/Dabble-Nodes#bot_sort">BoT-SORT tracker</a>, and the <a href="https://github.com/Natsunoyuki-AI-Laboratory/PeekingDuckReborn/wiki/Model-Nodes#vit_pose">VITPose human pose estimation model</a>. Mix and match different nodes to develop solutions to solve custom use cases.


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

## Local Weights Subdirectory
The original PeekingDuck model weights will be downloaded from https://storage.googleapis.com/peekingduck/models to a local subdirectory, which is set to `peekingduck_weights/` by default in PeekingDuckReborn. The name and location of `peekingduck_weights/` can be specified through local environment variables specified using a `.env` file.

For normal installations, the original PeekingDuck model weights will be downloaded to `PeekingDuckReborn/venv/Lib/site-packages/peekingduck_weights/`, and when installed in developer mode, they will be downloaded to `PeekingDuckReborn/peekingduck_weights/` by default if no local environment variables are set. Torchvision and HuggingFace model weights will be downloaded to the local cache directory. 

#### Specifying Local Environment Variables with a `.env` File.
Create a `.env` file under `PeekingDuckReborn/` to specify the path to `PeekingDuckReborn`, or another subdirectory name instead of `peekingduck_weights/`.
```
PEEKINGDUCK_DIR=<path to PeekingDuckReborn>
PEEKINGDUCK_WEIGHTS_SUBDIR=<weights subdirectory name>
```
For developer mode, `PEEKINGDUCK_DIR` only needs to be specified if you want `peekingduck_weights/` to be located somewhere else from `PeekingDuckReborn/`. For normal installations, `PEEKINGDUCK_DIR` can be used to specify a more convenient location for `peekingduck_weights/`.

#### `peekingduck_weights/` Structure
In general, `peekingduck_weights/` will have the following structure:
```
peekingduck_weights/
└───model_name_1/
    ├───pytorch/
    |   ├─── model_type_1/
    |   └─── model_type_2/
    └───tensorflow/
        ├─── model_type_1/
        └─── model_type_2/
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
```


## Currently Available Nodes
Please refer to the <a href="https://github.com/Natsunoyuki-AI-Laboratory/PeekingDuckReborn/wiki/PeekingDuckReborn-Nodes">PeekingDuckReborn wiki</a> for more information on the nodes currently available.


## Important Things to Note
### Images Are Numpy Arrays with BGR Channels
In the pipeline, images are `numpy` arrays with shape `[H, W, C]`, where `C` corresponds to Blue-Green-Red channels following OpenCV convention. Please keep this in mind when implementing new nodes or tests.

### Buggy Original Pose Estimation Models
The original PeekingDuck pose estimation models `hrnet`, `movenet` and `posenet` are extremely buggy and should not be used.


## Known Issues
1. The original pose estimation models (`hrnet`, `movenet`, `posenet`) will crash when more than one person exists in the frame.
2. Models implemented in TensorFlow (`yolo`, `yolo_face`, `yolo_license_plate`) might be buggy and not work properly.
3. `yolact_edge` implementation has issues with CUDA on Windows, resulting in the following error `ERROR:  RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!`.


## License
This project is a fork of [PeekingDuck](https://github.com/aisingapore/PeekingDuck), originally licensed under the Apache License 2.0.

PeekingDuckReborn contains substantial modifications and new code, and is distributed under the GNU General Public License v3.0 (GPL-3.0).

- Original code: Apache License 2.0 (see [LICENSE.Apache-2.0](./LICENSE.Apache-2.0.txt))
- This fork: GPL-3.0 (see [LICENSE](./LICENSE.txt))

See [NOTICE](./NOTICE.txt) for details.


## Acknowledgements
PeekingDuckReborn is an independently maintained and significantly modernized fork of [PeekingDuck](https://github.com/aisingapore/PeekingDuck), originally developed by AI Singapore. The original project has not been updated in several years and no longer works with current dependencies. This fork fixes bugs, adds new features, and updates the entire pipeline to work with modern Python versions, computer vision libraries, and deployment environments.

Any opinions, findings, conclusions, or recommendations expressed in this project are solely those of the authors of PeekingDuckReborn and do not reflect the views of AI Singapore or the original authors of [PeekingDuck](https://github.com/aisingapore/PeekingDuck).

PeekingDuckReborn is neither supported, funded by, nor affiliated with AI Singapore or the original authors of [PeekingDuck](https://github.com/aisingapore/PeekingDuck).
