from collections import OrderedDict
from typing import Any, Dict, List, Tuple

import numpy as np

from bot_sort import BoTSORT


class BoTSORTTracker:
    """Multi Object Tracking (MOT) with the BoT-SORT algorithm.

    This method is based on the assumption that the detector produces a
    detection per frame for every object to be tracked. Furthermore, it is
    assumed that detections of an object in consecutive frames have an
    unmistakably high overlap IoU which is commonly the case when using
    sufficiently high frame rates.

    References:
        BoT-SORT: Robust Associations Multi-Pedestrian Tracking
        https://arxiv.org/pdf/2206.14651

        We use the implementation:
        https://github.com/natsunoyuki/BoT-SORT
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.track_high_thresh = config.get("track_high_thresh", 0.6)
        self.track_low_thresh = config.get("track_low_thresh", 0.1)
        self.new_track_thresh = config.get("track_low_thresh", 0.7)
        self.match_thresh = config.get("match_thresh", 0.8)
        self.track_buffer = config.get("track_buffer", 30)
        self.frame_rate = config.get("frame_rate", 30)


        self.tracker = BoTSORT(
            track_high_thresh=self.track_high_thresh,
            track_low_thresh=self.track_low_thresh,
            new_track_thresh=self.new_track_thresh,
            match_thresh=self.match_thresh,
            track_buffer=self.track_buffer,
            frame_rate=self.frame_rate,
        )


    def track_detections(self, inputs: Dict[str, Any]) -> List[int]:
        """Initializes and updates tracker on each frame.

        Args:
            inputs (Dict[str, Any]): Dictionary with keys "img", "bboxes",
                "bbox_labesl", "bbox_scores".

        Returns:
            (List[int]]): List of track IDs.
        """
        bgr_frame = inputs["img"]
        frame_size = bgr_frame.shape[:2]

        bboxes = xyxyn2xyxy(inputs["bboxes"], *frame_size)
        labels = inputs["bbox_labels"]
        scores = inputs["bbox_scores"]

        tracked_objects = self.tracker.update(bboxes, labels, scores)

        ids = []
        bboxes = []
        bbox_labels = []
        bbox_scores = []
        for t in tracked_objects:
            ids.append(t.track_id)
            bboxes.append(t.tlbr.tolist())
            bbox_labels.append(t.label)
            bbox_scores.append(t.score)
            
        return {
            "ids": ids,
            "bboxes": xyxy2xyxyn(np.array(bboxes), *frame_size),
            "bbox_labels": np.array(bbox_labels),
            "bbox_scores": np.array(bbox_scores),
        }


def xyxyn2xyxy(inputs: np.ndarray, height: int, width: int) -> np.ndarray:
    """Converts bounding boxes format from (x1, y1, x2, y2) to (X1, Y1, X2, Y2).
    (x1, y1) is the normalized coordinates of the top-left corner, (x2, y2) is
    the normalized coordinates of the bottom-right corner. (X1, Y1) is the
    original coordinates of the top-left corner, (X2, Y2) is the original 
    coordinates of the bottom-right corner.

    Args:
        inputs (np.ndarray): Bounding box coordinates with (x1, y1, x2, y2)
            format.
        height (int): Original height of bounding box.
        width (int): Original width of bounding box.

    Returns:
        (np.ndarray): Converted bounding box coordinates with (X1, Y1, X2, Y2)
            format.
    """
    outputs = np.empty_like(inputs)
    outputs[:, 0] = inputs[:, 0] * width
    outputs[:, 1] = inputs[:, 1] * height
    outputs[:, 2] = inputs[:, 2] * width
    outputs[:, 3] = inputs[:, 3] * height
    return outputs


def xyxy2xyxyn(inputs: np.ndarray, height: int, width: int) -> np.ndarray:
    outputs = np.empty_like(inputs)
    outputs[:, 0] = inputs[:, 0] / width
    outputs[:, 1] = inputs[:, 1] / height
    outputs[:, 2] = inputs[:, 2] / width
    outputs[:, 3] = inputs[:, 3] / height
    return outputs
