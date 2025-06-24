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
import torch

import pytest

from peekingduck.pipeline.nodes.model.jdev1.jde_files import tracker
from peekingduck.pipeline.nodes.model.jdev1.jde_files.track import STrack

# TODO
@pytest.mark.skip("Too many test failures. Needs to be cleaned up.")
class TestJDETracker:
    def test_remove_duplicate_stracks(self):
        tlwhs = [
            np.array([10, 20, 30, 40]),
            np.array([20, 40, 60, 80]),
            np.array([40, 80, 120, 160]),
        ]
        score = torch.ones((1,), dtype=torch.float, device=torch.device("cpu"))
        feature = np.ones((10,), dtype=float)
        # Create 2 list of elementwise overlapping stracks
        stracks_1 = [STrack(tlwh, score, feature) for tlwh in tlwhs]
        stracks_2 = [STrack(tlwh, score, feature) for tlwh in tlwhs]

        frame_id = 10
        early_start = 2
        late_start = 5
        for i, _ in enumerate(tlwhs):
            stracks_1[i].frame_id = frame_id
            stracks_2[i].frame_id = frame_id
            # Alternate between which STrack is older so we cover more branches
            if i % 2 == 0:
                stracks_1[i].start_frame = early_start
                stracks_2[i].start_frame = late_start
            else:
                stracks_1[i].start_frame = late_start
                stracks_2[i].start_frame = early_start
        out_stracks_1, out_stracks_2 = tracker._remove_duplicate_stracks(
            stracks_1, stracks_2
        )
        # stracks_1 has more older tracks so more elements will be leftover
        assert len(out_stracks_1) == 2
        assert len(out_stracks_2) == 1
