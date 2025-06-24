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

from typing import Any, Dict, List

import numpy as np

from bot_sort import BoTSORT

from peekingduck.pipeline.nodes.dabble.bot_sortv1.utils import (
    xyxy2xyxyn, 
    xyxyn2xyxy,
)


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
        bgr_frame = inputs.get("img")
        bboxes = inputs.get("bboxes")
        labels = inputs.get("bbox_labels")
        scores = inputs.get("bbox_scores")

        # Denormalize bounding box coordinates.
        frame_size = bgr_frame.shape[:2]
        bboxes = xyxyn2xyxy(bboxes, *frame_size)

        # Update tracker.
        tracked_objects = self.tracker.update(bboxes, labels, scores)

        # Format tracks into something PeekingDuck's pipeline can ingest.
        ids, boxes, box_labels, box_scores = self.postprocess(tracked_objects)

        boxes = xyxy2xyxyn(boxes, *frame_size)
            
        return {
            "ids": ids,
            "bboxes": boxes,
            "bbox_labels": box_labels,
            "bbox_scores": box_scores,
        }


    def postprocess(self, tracked_objects=[]):
        """Post processes the tracked objects from BoT-SORT into a format which
        the PeekingDuck pipeline can ingest."""
        ids = []
        bboxes = []
        bbox_labels = []
        bbox_scores = []
        for t in tracked_objects:
            ids.append(t.track_id)
            bboxes.append(t.tlbr.tolist())
            bbox_labels.append(t.label)
            bbox_scores.append(t.score)
        
        if len(bboxes) == 0:
            bboxes = np.empty([0, 4])
        else:
            bboxes = np.array(bboxes)
        bbox_labels = np.array(bbox_labels)
        bbox_scores = np.array(bbox_scores)

        return ids, bboxes, bbox_labels, bbox_scores
