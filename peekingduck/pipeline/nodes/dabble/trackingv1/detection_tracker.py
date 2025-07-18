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

"""Tracker for object detector bounding boxes."""

import logging
from typing import Any, Dict, List

from peekingduck.pipeline.nodes.base import ThresholdCheckerMixin
from peekingduck.pipeline.nodes.dabble.trackingv1.tracking_files.iou_tracker import (
    IOUTracker,
)

# legacy_TrackerMOSSE is found only in the opencv-contrib-python package.
# IoU tracker will be used instead if opencv-python is installed.
try:
    from peekingduck.pipeline.nodes.dabble.trackingv1.tracking_files.opencv_tracker import (
        OpenCVTracker,
    )
except AttributeError as e:
    logging.warning(
        "{}. IoU tracker will be used. Install opencv-contrib-python to use the MOSSE tracker.".format(e)
    )
    OpenCVTracker = IOUTracker


class DetectionTracker(ThresholdCheckerMixin):  # pylint: disable=too-few-public-methods
    """Tracks detection bounding boxes using the chosen algorithm.

    Args:
        config (Dict[str, Any]): Configration dict containing the following:
            tracking_type (str): Type of tracking algorithm to be used, one of
                ["iou", "mosse"].
            iou_threshold (float): Minimum IoU value to be used with the
                matching logic.
            max_lost (int): Maximum number of frames to keep "lost" tracks
                after which they will be removed. Only used in IOUTracker.

    Raises:
        ValueError: `tracking_type` is not one of ["iou", "mosse"].
        ValueError: `iou_threshold` is not within [0, 1].
        ValueError: `max_lost` is negative.
    """

    tracker_constructors = {"iou": IOUTracker, "mosse": OpenCVTracker}


    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.logger = logging.getLogger(__name__)

        self.check_bounds("iou_threshold", "[0, 1]")
        self.check_bounds("max_lost", "[0, +inf)")
        self.check_valid_choice("tracking_type", {"iou", "mosse"})

        self.tracker = self.tracker_constructors[config["tracking_type"]](config)


    def track_detections(self, inputs: Dict[str, Any]) -> List[int]:
        """Tracks detections using the selected algorithm.

        Args:
            inputs (Dict[str, Any]): Dictionary with keys "img", "bboxes", and
                "bbox_scores.

        Returns:
            (List[int]): Tracking IDs of the detection bounding boxes.
        """
        track_ids = self.tracker.track_detections(inputs)
        return track_ids
