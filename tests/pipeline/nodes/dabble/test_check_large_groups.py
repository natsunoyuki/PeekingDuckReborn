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

import pytest

from peekingduck.pipeline.nodes.dabble.check_large_groups import Node


@pytest.fixture
def check_large_groups():
    node = Node(
        {"input": ["obj_attrs"], "output": ["large_groups"], "group_size_threshold": 3}
    )
    return node


class TestCheckLargeGroups:
    def test_no_obj_groups(self, check_large_groups):
        array1 = []
        input1 = {"obj_attrs": {"groups": array1}}

        assert check_large_groups.run(input1)["large_groups"] == []
        assert input1["obj_attrs"]["groups"] == array1

    def test_no_large_groups(self, check_large_groups):
        array1 = [0, 1, 2, 3, 4, 5]
        input1 = {"obj_attrs": {"groups": array1}}

        assert check_large_groups.run(input1)["large_groups"] == []
        assert input1["obj_attrs"]["groups"] == array1

    def test_multi_large_groups(self, check_large_groups):
        array1 = [0, 1, 0, 3, 1, 0, 1, 2, 1, 0]
        input1 = {"obj_attrs": {"groups": array1}}

        assert check_large_groups.run(input1)["large_groups"] == [0, 1]
        assert input1["obj_attrs"]["groups"] == array1
