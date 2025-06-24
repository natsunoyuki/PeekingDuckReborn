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

from peekingduck.pipeline.nodes.dabble.bbox_count import Node


@pytest.fixture
def bbox_count():
    node = Node(
        {
            "input": ["bboxes"],
            "output": ["count"],
        }
    )
    return node


class TestBboxCount:
    def test_no_bboxes(self, bbox_count):
        array1 = []
        input1 = {"bboxes": array1}

        assert bbox_count.run(input1)["count"] == 0
        np.testing.assert_equal(input1["bboxes"], array1)

    def test_multi_bboxes(self, bbox_count):
        array1 = [np.array([0.1, 0.2, 0.3, 0.4]), np.array([0.5, 0.6, 0.7, 0.8])]
        input1 = {"bboxes": array1}

        assert bbox_count.run(input1)["count"] == 2
        np.testing.assert_equal(input1["bboxes"], array1)
