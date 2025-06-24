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

import numpy as np
import pytest

from peekingduck.pipeline.nodes.dabble.bbox_to_btm_midpoint import Node


@pytest.fixture
def size():
    return (400, 600, 3)


@pytest.fixture
def bbox_to_btm_midpoint():
    node = Node(
        {
            "input": ["bboxes", "img"],
            "output": ["btm_midpoint"],
        }
    )
    return node


class TestBboxToBtmMidpoint:
    def test_no_bboxes(self, create_image, bbox_to_btm_midpoint, size):
        array1 = []
        img1 = create_image(size)
        input1 = {"bboxes": array1, "img": img1}

        assert bbox_to_btm_midpoint.run(input1)["btm_midpoint"] == []
        np.testing.assert_equal(input1["img"], img1)
        np.testing.assert_equal(input1["bboxes"], array1)

    def test_multi_bboxes(self, create_image, bbox_to_btm_midpoint, size):
        array1 = [np.array([0.1, 0.2, 0.3, 0.4]), np.array([0.5, 0.6, 0.7, 0.8])]
        img1 = create_image(size)
        input1 = {"bboxes": array1, "img": img1}

        assert len(bbox_to_btm_midpoint.run(input1)["btm_midpoint"]) == 2
        np.testing.assert_equal(input1["img"], img1)
        np.testing.assert_equal(input1["bboxes"], array1)

    def test_formula(self, create_image, bbox_to_btm_midpoint, size):
        array1 = [np.array([0.1, 0.2, 0.3, 0.4])]
        img1 = create_image(size)
        input1 = {"bboxes": array1, "img": img1}

        assert bbox_to_btm_midpoint.run(input1)["btm_midpoint"] == [(120, 160)]
        np.testing.assert_equal(input1["img"], img1)
        np.testing.assert_equal(input1["bboxes"], array1)
