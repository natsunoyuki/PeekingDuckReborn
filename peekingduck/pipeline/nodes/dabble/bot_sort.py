"""🎯 Performs multiple object tracking for detected bboxes."""

from typing import Any, Dict

from peekingduck.pipeline.nodes.abstract_node import AbstractNode
from peekingduck.pipeline.nodes.dabble.bot_sortv1.detection_tracker import (
    DetectionTracker,
)


class Node(AbstractNode):
    """Uses bounding boxes detected by an object detector model to track
    multiple objects with the BoT-SORT algorithm.

    Inputs:
        |img_data|
        |bboxes_data|
        |bboxes_labels_data|
        |bboxes_scores_data|


    Outputs:
        |obj_attrs_data|
        :mod:`dabble.tracking` produces the ``ids`` attribute which contains
        the tracking IDs of the detections.
        |bboxes_data|
        |bboxes_labels_data|
        |bboxes_scores_data|   


    Configs:
        track_high_thresh (:obj:`float`): **[0, 1], default=0.6**. |br|
            Detection confidence score threshold for a good detection.
        track_low_thresh (:obj:`float`): **[0, 1], default=0.1**. |br|
            Minimum detection confidence score required. Any detections lower
            than this score will be discarded.
        new_track_thresh (:obj:`float`): **[0, 1], default=0.7**. |br|
        match_thresh (:obj:`float`): **[0, 1], default=0.8**. |br|
        track_buffer (:obj:`int`): **[1, +inf), default=30**. |br|
        frame_rate (:obj:`int`): **[1, +inf), default=30**. |br|
    """
    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)
        self.tracker = DetectionTracker(self.config)


    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Tracks detection bounding boxes.

        Args:
            inputs (Dict[str, Any]): Dictionary with keys "img", "bboxes",
                "bbox_scores", "bbox_labels".

        Returns:
            outputs (Dict[str, Any]): Tracking IDs of bounding boxes.
            "obj_attrs" key is used for compatibility with draw nodes.
        """
        outputs = self.tracker.track_detections(inputs)

        return {
            "obj_attrs": {"ids": outputs.get("ids")},
            "bboxes": outputs.get("bboxes"),
            "bbox_labels": outputs.get("bbox_labels"),
            "bbox_scores": outputs.get("bbox_scores"),
        }
    

    def _get_config_types(self) -> Dict[str, Any]:
        """Returns dictionary mapping the node's config keys to respective types."""
        return {
            "track_high_thresh": float,
            "track_low_thresh": float,
            "new_track_thresh": float,
            "match_thresh": float,
            "track_buffer": int,
            "frame_rate": int,
        }
    

    def _reset_model(self) -> None:
        """Creates a new instance of DetectionTracker."""
        self.logger.info(f"Creating new {self.config['tracking_type']} tracker...")
        self.tracker = DetectionTracker(self.config)
