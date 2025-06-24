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
Adjusts the brightness of an image.
"""


from typing import Any, Dict

import cv2
import numpy as np

from peekingduck.pipeline.nodes.abstract_node import AbstractNode
from peekingduck.pipeline.nodes.base import ThresholdCheckerMixin


class Node(ThresholdCheckerMixin, AbstractNode):
    """Adjusts the brightness of an image, by adding a bias/`beta parameter
    <https://docs.opencv.org/4.x/d3/dc1/tutorial_basic_
    linear_transform.html>`_.

    Inputs:
        |img_data|

    Outputs:
        |img_data|

    Configs:
        beta (:obj:`int`): **[-100, 100], default = 0**. |br|
            Increasing the value of beta increases image brightness, and vice
            versa.
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)

        self.check_bounds("beta", "[-100, 100]")

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Adjusts the brightness of an image frame.

        Args:
            inputs (Dict): Inputs dictionary with the key `img`.

        Returns:
            (Dict): Outputs dictionary with the key `img`.
        """
        orig_shape = inputs["img"].shape
        img_vector = np.reshape(inputs["img"], (1, -1))
        cv2.add(img_vector, self.beta, img_vector)
        img = np.reshape(img_vector, orig_shape)

        return {"img": img}

    def _get_config_types(self) -> Dict[str, Any]:
        """Returns dictionary mapping the node's config keys to respective types."""
        return {"beta": int}
