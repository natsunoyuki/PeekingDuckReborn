# Modifications copyright 2025 Natsunoyuki AI Laboratory
#
# PeekingDuckReborn is free software: you can redistribute it and/or modify it 
# under the terms of the GNU General Public License as published by the Free 
# Software Foundation, either version 3 of the License, or (at your option) any 
# later version.
#
# PeekingDuckReborn is distributed in the hope that it will be useful, but 
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or 
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more 
# details.
#
# You should have received a copy of the GNU General Public License along with 
# PeekingDuckReborn. If not, see <https://www.gnu.org/licenses/>.

# Original copyright 2022 AI Singapore
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path
from unittest import mock
import platform
import cv2
import numpy as np
import numpy.testing as npt
import pytest
import yaml
from typeguard import TypeCheckError

from peekingduck.pipeline.nodes.base import WeightsDownloaderMixin
from peekingduck.pipeline.nodes.model.yolo import Node
from tests.conftest import PKD_DIR, get_groundtruth

GT_RESULTS = get_groundtruth(Path(__file__).resolve())


@pytest.fixture
def yolo_config():
    with open(PKD_DIR / "configs" / "model" / "yolo.yml") as infile:
        node_config = yaml.safe_load(infile)
    node_config["root"] = Path.cwd()
    # The following prevents pytest from creating `peekingduck_weights/`
    # in the parent directory of `PeekingDuckReborn/`.
    node_config["weights_parent_dir"] = str(PKD_DIR.parent)
    return node_config


@pytest.fixture(
    params=[
        {"key": "iou_threshold", "value": -0.5},
        {"key": "iou_threshold", "value": 1.5},
        {"key": "score_threshold", "value": -0.5},
        {"key": "score_threshold", "value": 1.5},
    ],
)
def yolo_bad_config_value(request, yolo_config):
    yolo_config[request.param["key"]] = request.param["value"]
    return yolo_config


@pytest.fixture(params=["v4", "v4tiny"])
def yolo_type(request, yolo_config):
    yolo_config["model_type"] = request.param
    return yolo_config


@pytest.mark.mlmodel
# TODO: fix this test for Windows.
# This test passes on Ubuntu and MacOS, but fails on Windows
# due to issues with TensorFlow.
@pytest.mark.skipif(platform.system()=="Windows", reason="This test consistently fails on Windows.")
class TestYolo:
    def test_no_human_image(self, no_human_image, yolo_type):
        no_human_img = cv2.imread(no_human_image)
        yolo = Node(yolo_type)
        output = yolo.run({"img": no_human_img})
        expected_output = {
            "bboxes": np.empty((0, 4), dtype=np.float32),
            "bbox_labels": np.empty((0)),
            "bbox_scores": np.empty((0), dtype=np.float32),
        }
        assert output.keys() == expected_output.keys()
        npt.assert_equal(output["bboxes"], expected_output["bboxes"])
        npt.assert_equal(output["bbox_labels"], expected_output["bbox_labels"])
        npt.assert_equal(output["bbox_scores"], expected_output["bbox_scores"])

    def test_detect_human_bboxes(self, human_image, yolo_type):
        human_img = cv2.imread(human_image)
        yolo = Node(yolo_type)
        output = yolo.run({"img": human_img})

        assert "bboxes" in output
        assert output["bboxes"].size > 0

        model_type = yolo.config["model_type"]
        image_name = Path(human_image).stem
        expected = GT_RESULTS[model_type][image_name]

        npt.assert_allclose(output["bboxes"], expected["bboxes"], atol=1e-3)
        npt.assert_equal(output["bbox_labels"], expected["bbox_labels"])
        npt.assert_allclose(output["bbox_scores"], expected["bbox_scores"], atol=1e-2)

    def test_get_detect_ids(self, yolo_type):
        yolo = Node(yolo_type)
        assert yolo.model.detect_ids == [0]

    def test_invalid_config_detect_ids(self, yolo_config):
        yolo_config["detect"] = 1
        with pytest.raises(TypeCheckError):
            _ = Node(config=yolo_config)

    def test_invalid_config_value(self, yolo_bad_config_value):
        with pytest.raises(ValueError) as excinfo:
            _ = Node(config=yolo_bad_config_value)
        assert "_threshold must be between [0.0, 1.0]" in str(excinfo.value)

    @mock.patch.object(WeightsDownloaderMixin, "_has_weights", return_value=True)
    def test_invalid_config_model_files(self, _, yolo_config):
        with pytest.raises(ValueError) as excinfo:
            yolo_config["weights"][yolo_config["model_format"]]["model_file"][
                yolo_config["model_type"]
            ] = "some/invalid/path"
            _ = Node(config=yolo_config)
        assert "Graph file does not exist. Please check that" in str(excinfo.value)
