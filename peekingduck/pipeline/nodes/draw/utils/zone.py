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
Draws utilities regarding zoning functions
"""

from typing import Any, List, Tuple

import cv2
import numpy as np

from peekingduck.pipeline.nodes.draw.utils.constants import (
    PRIMARY_PALETTE,
    PRIMARY_PALETTE_LENGTH,
    VERY_THICK,
)


def _draw_zone_area(
    frame: np.ndarray, points: List[Tuple[int]], zone_index: int
) -> None:
    num_points = len(points)
    for i in range(num_points):
        cv2.line(
            frame,
            points[i],
            points[(i + 1) % num_points],
            PRIMARY_PALETTE[(zone_index + 1) % PRIMARY_PALETTE_LENGTH],
            VERY_THICK,
        )


def draw_zones(frame: np.ndarray, zones: List[Any]) -> None:
    """draw the boundaries of the zones used in zoning analytics

    Args:
        frame (np.array): image of current frame
        zones (Zone): zones used in the zoning analytics. possible
        classes are Area and Divider.
    """
    for i, zone_pts in enumerate(zones):
        _draw_zone_area(frame, zone_pts, i)
