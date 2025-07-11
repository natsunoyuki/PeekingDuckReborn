# Copyright 2025 Natsunoyuki AI Laboratory
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

"""Utility functions which convert keypoint connection coordinates from one 
format to another. This is for keypoint connections and not keypoints!
For keypoint coordinates, see keypoint/transforms.py."""

import numpy as np


def xy2xyn(inputs: np.ndarray, height: float, width: float) -> np.ndarray:
    """Converts from [x, y] to normalized [xn, yn].
    Normalized coordinates are w.r.t. original image size.

    Args:
        inputs (np.ndarray): Input keypoints (N, D, 2, 2) each with the
            format `(x, y)`.
        height (int): Height of the image frame.
        height (int): Width of the image frame.

    Returns:
        (np.ndarray): Keypoints (N, D, 2, 2) with the format `normalized (x, y)`.
    """
    outputs = np.empty_like(inputs)
    outputs[:, :, 0, 0] = inputs[:, :, 0, 0] / width
    outputs[:, :, 1, 1] = inputs[:, :, 1, 1] / height
    return outputs


def xyn2xy(inputs: np.ndarray, height: float, width: float) -> np.ndarray:
    """Converts from normalized [xn, yn] to pixel coordinates [x, y] format. 

    Args:
        inputs (np.ndarray): Keypoints (N, D, 2, 2) with the format 
            `normalized (x, y)`.
        height (int): Height of the image frame.
        height (int): Width of the image frame.

    Returns:
        (np.ndarray): Keypoints (N, D, 2, 2) each with the format `(x, y)`.
    """
    outputs = np.empty_like(inputs)
    outputs[:, :, 0, 0] = inputs[:, :, 0, 0] * width
    outputs[:, :, 1, 1] = inputs[:, :, 1, 1] * height
    return outputs.astype(int)
