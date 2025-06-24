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

import numpy as np
import numpy.testing as npt
import pytest

from peekingduck.pipeline.nodes.model.posenetv1.posenet_files.decode import (
    _clip_to_indices,
    _traverse_to_target_keypoint,
)

NP_FILE = np.load(Path(__file__).resolve().parent / "posenet.npz")


@pytest.fixture
def source_keypoint():
    return np.array([76.23, 70.98])

# TODO
@pytest.mark.skip("The implementation of PoseNet in PKD is problematic and should not be used.")
class TestDecode:
    def test_clip_to_indices(self, source_keypoint):
        source_keypoint_indices = _clip_to_indices(
            keypoints=source_keypoint, output_stride=16, width=14, height=14
        )
        npt.assert_almost_equal(
            source_keypoint_indices,
            np.array([5, 4]),
            4,
            err_msg="Unexpected output from clipping to indices",
        )

    def test_traverse_to_target_keypoint(self, source_keypoint):
        score, coords = _traverse_to_target_keypoint(
            edge_id=0,
            source_keypoint=source_keypoint,
            target_keypoint_id=0,
            scores=NP_FILE["scores"],
            offsets=NP_FILE["offsets"],
            output_stride=16,
            displacements=NP_FILE["displacements_bwd"],
        )
        assert score == pytest.approx(0.9706, 0.01), "Score did not meet expected value"
        npt.assert_almost_equal(
            score, 0.9706, 4, err_msg="Score did not meet expected value"
        )
        npt.assert_almost_equal(
            coords,
            np.array([74.90, 72.22]),
            2,
            err_msg="Coordinates of incorrect values",
        )

        score, coords = _traverse_to_target_keypoint(
            edge_id=1,
            source_keypoint=np.array([76.23, 70.98]),
            target_keypoint_id=3,
            scores=NP_FILE["scores"],
            offsets=NP_FILE["offsets"],
            output_stride=16,
            displacements=NP_FILE["displacements_fwd"],
        )
        npt.assert_almost_equal(
            score, 0.4393, 4, err_msg="Score did not meet expected value"
        )
        npt.assert_almost_equal(
            coords,
            np.array([77.11, 71.62]),
            2,
            err_msg="Coordinates of incorrect values",
        )
