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
Draws the 2D boundaries of a zone.
"""

from typing import Any, Dict

from peekingduck.pipeline.nodes.draw.utils.zone import draw_zones
from peekingduck.pipeline.nodes.abstract_node import AbstractNode


class Node(AbstractNode):
    """Draws the boundaries of each specified zone onto the image.

    The ``draw.zones`` node uses the :term:`zones` output from the
    ``dabble.zone_count`` node to draw a bounding box that represents the zone
    boundaries onto the image.

    Inputs:
        |img_data|

        |zones_data|

    Outputs:
        |none_output_data|

    Configs:
        None.
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Draws the boundaries of each specified zone onto the image.

        Args:
            inputs (dict): Dictionary with keys "zones", "img".

        Returns:
            outputs (dict): Dictionary with keys "none".
        """

        draw_zones(inputs["img"], inputs["zones"])  # type: ignore
        return {}
