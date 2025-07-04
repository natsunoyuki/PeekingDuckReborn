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
Loads configurations for individual nodes.
"""

from pathlib import Path
from typing import Any, Dict

import logging
import yaml


class ConfigLoader:  # pylint: disable=too-few-public-methods
    """A helper class to create pipeline.

    The config loader class is used to allow for instantiation of Node classes
    directly instead of reading configurations from the run config yaml.

    Args:
        base_dir (:obj:`pathlib.Path`): Base directory of ``peekingduck``
    """

    def __init__(self, base_dir: Path) -> None:
        self._base_dir = base_dir
        self.logger = logging.getLogger(__name__)

    def _get_config_path(self, node: str) -> Path:
        """Based on the node, return the corresponding node config path"""
        configs_folder = self._base_dir / "configs"
        node_type, node_name = node.split(".")
        file_path = configs_folder / node_type / f"{node_name}.yml"

        return file_path

    def get(self, node_name: str) -> Dict[str, Any]:
        """Gets node configuration for specified node.

        Args:
            node_name (:obj:`str`): Name of node.

        Returns:
            node_config (:obj:`Dict[str, Any]`): A dictionary of node
            configurations for the specified node.
        """
        file_path = self._get_config_path(node_name)

        with open(file_path) as file:
            node_config = yaml.safe_load(file)

        # some models require the knowledge of where the root is for loading
        node_config["root"] = self._base_dir
        return node_config
