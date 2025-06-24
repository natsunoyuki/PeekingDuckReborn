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
Test for group bbox and tag node
"""

import numpy as np
import pytest

from peekingduck.pipeline.nodes.draw.group_bbox_and_tag import Node


@pytest.fixture
def draw_group_bbox_and_tag():
    node = Node(
        {
            "input": ["img", "bboxes", "obj_attrs", "large_groups"],
            "output": ["none"],
            "bbox_color": [0, 140, 255],
            "bbox_thickness": 4,
            "tag": "LARGE GROUP!",
        }
    )
    return node


class TestTag:
    def test_no_groups(self, draw_group_bbox_and_tag, create_image):
        original_img = create_image((28, 28, 3))
        output_img = original_img.copy()
        input1 = {
            "img": output_img,
            "bboxes": [],
            "obj_attrs": {"groups": []},
            "large_groups": [],
        }
        draw_group_bbox_and_tag.run(input1)
        assert original_img.shape == output_img.shape
        np.testing.assert_equal(original_img, output_img)

    # formula: processed image = contrast * image + brightness
    def test_draw_group_and_tag(self, draw_group_bbox_and_tag, create_image):
        original_img = create_image((28, 28, 3))
        output_img = original_img.copy()
        input1 = {
            "img": output_img,
            "bboxes": [np.array([0, 0, 10, 10])],
            "obj_attrs": {"groups": [0]},
            "large_groups": [0],
        }
        draw_group_bbox_and_tag.run(input1)

        assert original_img.shape == output_img.shape
        np.testing.assert_raises(
            AssertionError, np.testing.assert_equal, original_img, output_img
        )
