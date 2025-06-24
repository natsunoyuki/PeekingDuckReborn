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
Preprocessing functions for input nodes
"""

import logging
from typing import Any, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def get_res(stream: Any) -> Tuple[int, int]:
    """Gets the resolution for the video frame"""
    width = int(stream.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT))

    return width, height


def mirror(frame: np.ndarray) -> np.ndarray:
    """Mirrors a video frame."""
    return cv2.flip(frame, 1)


def resize_image(frame: np.ndarray, desired_width: int, desired_height: int) -> Any:
    """function that resizes the image input
    to the desired dimensions

    Args:
        frame (np.array): image
        desired_width: width of the resized image
        desired_height: height of the resized image

    Returns:
        image (np.array): returns a scaled image depending on the
        desired wight and height
    """
    return cv2.resize(frame, (desired_width, desired_height))
