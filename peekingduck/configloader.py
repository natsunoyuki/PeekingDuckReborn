# Copyright 2021 AI Singapore
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
from typing import Any, Dict, List, Tuple

import yaml

# Master map file for class name to object IDs for object detection models
MASTER_MAP = "pipeline/models/master_map.yml"


class ConfigLoader:  # pylint: disable=too-few-public-methods
    """A helper class to create pipeline.

    The config loader class is used to allow for instantiation of Node classes
    directly instead of reading configurations from the run config yaml.

    Args:
        base_dir (:obj:`pathlib.Path`): Base directory of ``peekingduck``
    """

    def __init__(self, base_dir: Path) -> None:
        self._base_dir = base_dir

    def _get_config_path(self, node: str) -> Path:
        """Based on the node, return the corresponding node config path"""
        configs_folder = self._base_dir / "configs"
        node_type, node_name = node.split(".")
        file_path = configs_folder / node_type / f"{node_name}.yml"

        return file_path

    def _load_mapping(self, node_name: str) -> Dict[str, int]:
        """Loads class name to object ID mapping from the file
        peekingduck/pipeline/nodes/model/master_map.yml

        Args:
            node_name (str): Tells function which mapping to load,
                             Possible values = { model.efficientdet, model.yolo }.

        Returns:
            Dict[str, int]: Mapping of class names to object IDs relevant to given node_name
        """
        assert node_name in (
            "model.yolo",
            "model.efficientdet",
        ), f"Name Error: expect model.yolo or model.efficientdet but got {node_name}"

        master_map_file = self._base_dir / MASTER_MAP
        with master_map_file.open() as map_file:
            model_map, class_id_map = yaml.safe_load_all(map_file)
        print("** model_map")
        print(model_map)
        print("** class_id_map")
        print(class_id_map)
        return {}

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

    def change_class_name_to_id(
        self, node_name: str, key: str, value: List[Any]
    ) -> Tuple[str, List[int]]:
        """Process object detection model node's detect_ids key and check for
        any class names to be converted to object IDs.
        E.g. person to 0, car to 2

        Args:
            node_name (str): to determine if node is efficientdet or yolo,
                             because both lists of object IDs are different.
            key (str): expected to be "detect_ids"; error otherwise.
            value (List[Any]): list of class names or object IDs for detection.
                               If object IDs, do nothing.
                               If class names, convert to object IDs.

        Returns:
            Tuple[str, List[int]]: "detect_ids", list of sorted object IDs.
        """
        class_id_map = self._load_mapping(node_name)
        obj_ids_set = {
            x if isinstance(x, int) else class_id_map.get(x, 0) for x in value
        }
        obj_ids_sorted_list = sorted(list(obj_ids_set))
        return key, obj_ids_sorted_list
