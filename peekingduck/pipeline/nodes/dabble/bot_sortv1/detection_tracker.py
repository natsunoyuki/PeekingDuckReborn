import logging
from typing import Any, Dict, List

from peekingduck.pipeline.nodes.base import ThresholdCheckerMixin
from peekingduck.pipeline.nodes.dabble.bot_sortv1.bot_sort_tracker import (
    BoTSORTTracker,
)


class DetectionTracker(ThresholdCheckerMixin):  # pylint: disable=too-few-public-methods
    """Tracks detection bounding boxes using the chosen algorithm.

    Args:
        config (Dict[str, Any]): Configration dict containing the following:
            track_high_thresh (float): Detection confidence score threshold for
                a good detection.
            track_low_thresh (float): Detection confidence score threshold for a
                weak detection.
            new_track_thresh (float): New track threshold.
            match_thresh (float): Track match threshold.
            track_buffer (int): Track buffer in frame rate. Should be the same
                as frame_rate in most situations.
            frame_rate (int): Video frame rate.

    Raises:
        ValueError: `track_high_thresh` is not within [0, 1].
        ValueError: `track_low_thresh` is not within [0, 1].
        ValueError: `new_track_thresh` is not within [0, 1].
        ValueError: `match_thresh` is not within [0, 1].
        ValueError: `track_buffer` is not positive.
        ValueError: `frame_rate` is not positive.
    """
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.logger = logging.getLogger(__name__)

        self.check_bounds("track_high_thresh", "[0, 1]")
        self.check_bounds("track_low_thresh", "[0, 1]")
        self.check_bounds("new_track_thresh", "[0, 1]")
        self.check_bounds("match_thresh", "[0, 1]")
        self.check_bounds("track_buffer", "[1, +inf)")
        self.check_bounds("frame_rate", "[1, +inf)")

        self.tracker = BoTSORTTracker(config)


    def track_detections(self, inputs: Dict[str, Any]) -> List[int]:
        """Tracks detections using the selected algorithm.

        Args:
            inputs (Dict[str, Any]): Dictionary with keys "img", "bboxes",
                "bbox_scores", "bbox_labels".

        Returns:
            (List[int]): Tracking IDs of the detection bounding boxes.
        """
        return self.tracker.track_detections(inputs)
