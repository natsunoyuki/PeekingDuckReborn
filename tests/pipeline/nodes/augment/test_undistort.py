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

"""
Test for augment undistort node
"""

import gc
import cv2
import numpy as np
import pytest
from pathlib import Path
import tensorflow.keras.backend as K
from tests.conftest import PKD_DIR, TEST_DATA_DIR, TEST_IMAGES_DIR

from peekingduck.pipeline.nodes.augment.undistort import Node

UNDISTORT_BEFORE = ["undistort_before.jpg"]
CAMERA_COEFFICIENTS = ["camera_calibration_coeffs.yml"]


@pytest.fixture(params=UNDISTORT_BEFORE)
def undistort_before(request):
    yield str(TEST_IMAGES_DIR / request.param)
    K.clear_session()
    gc.collect()


@pytest.fixture(params=CAMERA_COEFFICIENTS)
def camera_coefficients(request):
    yield str(TEST_DATA_DIR / "undistort" / request.param)
    K.clear_session()
    gc.collect()


class TestUndistort:
    def test_undistort(self, undistort_before, camera_coefficients):
        before_img = cv2.imread(undistort_before)
        file_path = str(camera_coefficients)
        node = Node({"input": ["img"], "output": ["img"], "file_path": file_path})

        outputs = node.run({"img": before_img})
        assert before_img.shape != outputs["img"].shape

    def test_file_io(self):
        with pytest.raises(ValueError) as excinfo:
            Node({"input": ["img"], "output": ["img"], "file_path": "file.txt"})
        assert str(excinfo.value) == "Filepath must have a '.yml' extension."

        par_dir = Path(__file__).parent / "undistort" / "nonexistent_file.yml"
        file_path = str(par_dir)

        with pytest.raises(FileNotFoundError) as excinfo:
            Node({"input": ["img"], "output": ["img"], "file_path": file_path})
        assert (
            str(excinfo.value)
            == f"File {file_path} does not exist. Please run the camera calibration again "
            "with the dabble.camera_calibation node. You may refer to this tutorial: "
            "https://peekingduck.readthedocs.io/en/stable/nodes/dabble.camera_calibration.html"
        )
