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

"""Records the nodes' outputs to a CSV file."""

import logging
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from peekingduck.pipeline.nodes.abstract_node import AbstractNode
from peekingduck.pipeline.utils.bbox.transforms import xyxyn2xyxy
from peekingduck.pipeline.utils.keypoint.transforms import xyn2xy as xyn2xy_kpts
from peekingduck.pipeline.utils.keypoint_conn.transforms import xyn2xy as xyn2xy_conns
from peekingduck.pipeline.nodes.output.utils.csvlogger import CSVLogger


class Node(AbstractNode):
    """Tracks user-specified parameters and outputs the results in a CSV file.

    Inputs:
        ``all`` (:obj:`List`): A placeholder that represents a flexible input.
        Actual inputs to be written into the CSV file can be configured in
        ``stats_to_track``.

    Outputs:
        |none_output_data|

    Configs:
        stats_to_track (:obj:`List[str]`):
            **default = ["keypoints", "bboxes", "bbox_labels"]**. |br|
            Parameters to log into the CSV file. The chosen parameters must be
            present in the data pool.
        file_path (:obj:`str`):
            **default = "PeekingDuck/data/stats.csv"**. |br|
            Path of the CSV file to be saved. The resulting file name would have an appended
            timestamp.
        logging_interval (:obj:`int`): **default = 1**. |br|
            Interval between each log, in terms of seconds.
    """
    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)

        self.logger = logging.getLogger(__name__)
        self.save_pixel_coords = self.save_pixel_coords
        self.logging_interval = int(self.logging_interval)  # type: ignore
        self.file_path = Path(self.file_path)  # type: ignore
        # check if file_path has a '.csv' extension
        if self.file_path.suffix != ".csv":
            raise ValueError("Filepath must have a '.csv' extension.")

        self._file_path_datetime = self._append_datetime_file_path(self.file_path)
        self._stats_checked = False
        self.stats_to_track: List[str]
        self.csv_logger = CSVLogger(
            self._file_path_datetime, self.stats_to_track, self.logging_interval
        )


    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Writes the current state of the tracked statistics into
        the csv file as a row entry

        Args:
            inputs (dict): The data pool of the pipeline.

        Returns:
            outputs: [None]
        """
        # reset and terminate when there are no more data
        if inputs["pipeline_end"]:
            self._reset()
            return {}

        if not self._stats_checked:
            self._check_tracked_stats(inputs)
            # self._stats_to_track might change after the check
            self.csv_logger = CSVLogger(
                self._file_path_datetime, self.stats_to_track, self.logging_interval
            )

        if self.save_pixel_coords is True:
            inputs = self._norm_to_pixel_coords(inputs=inputs)

        self.csv_logger.write(inputs, self.stats_to_track)
        return {}


    def _norm_to_pixel_coords(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Converts normalized [x, y] coordinates to pixel coordinates for
        `bboxes`, `keypoints` and `keypoint_conns`."""
        H, W = inputs["img"].shape[:2] 
        if "bboxes" in inputs and "bboxes" in self.stats_to_track:
            inputs["bboxes"] = xyxyn2xyxy(inputs["bboxes"], H, W)
        if "keypoints" in inputs and "keypoints" in self.stats_to_track:
            inputs["keypoints"] = xyn2xy_kpts(inputs["keypoints"], H, W, clip_min=-1)
        if "keypoint_conns" in inputs and "keypoint_conns" in self.stats_to_track:
            inputs["keypoint_conns"] = xyn2xy_kpts(inputs["keypoint_conns"], H, W)
        return inputs


    def _check_tracked_stats(self, inputs: Dict[str, Any]) -> None:
        """Checks whether user input statistics is present in the data pool
        of the pipeline. Statistics not present in data pool will be
        ignored and dropped.
        """
        valid = []
        invalid = []

        for stat in self.stats_to_track:
            if stat in inputs:
                valid.append(stat)
            else:
                invalid.append(stat)

        if invalid:
            msg = textwrap.dedent(
                f"""\
                {invalid} are not valid outputs.
                Data pool only has this outputs: {list(inputs.keys())}
                Only {valid} will be logged in the csv file.
                """
            )
            self.logger.warning(msg)

        # update stats_to_track with valid stats found in data pool
        self.stats_to_track = valid
        self._stats_checked = True


    def _get_config_types(self) -> Dict[str, Any]:
        """Returns dictionary mapping the node's config keys to respective types."""
        return {"stats_to_track": List[str], "file_path": str, "logging_interval": int}


    def _reset(self) -> None:
        del self.csv_logger
        # initialize for use in run
        self._stats_checked = False


    @staticmethod
    def _append_datetime_file_path(file_path: Path) -> Path:
        """Append time stamp to the filename."""
        current_time = datetime.now()
        # output as '240621-15-09-13'
        time_str = current_time.strftime("%d%m%y-%H-%M-%S")

        # append timestamp to filename before extension
        # Format: filename_timestamp.extension
        file_path_with_timestamp = file_path.with_name(
            f"{file_path.stem}_{time_str}{file_path.suffix}"
        )
        return file_path_with_timestamp
