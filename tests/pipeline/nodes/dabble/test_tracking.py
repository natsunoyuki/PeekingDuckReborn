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

from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest

from peekingduck.pipeline.nodes.dabble.tracking import Node

# Frame index for manual manipulation of detections to trigger some
# branches
SEQ_IDX = 6
SIZE = (400, 600, 3)


@pytest.fixture
def tracking_config():
    return {
        "root": Path.cwd(),
        "input": ["img", "bboxes", "bbox_scores"],
        "output": ["obj_tags"],
    }


@pytest.fixture(params=["iou", "mosse"])
def tracker(request, tracking_config):
    tracking_config["tracking_type"] = request.param
    node = Node(tracking_config)
    return node


class TestTracking:
    def test_should_raise_for_invalid_tracking_type(self, tracking_config):
        tracking_config["tracking_type"] = "invalid type"
        with pytest.raises(ValueError) as excinfo:
            _ = Node(tracking_config)
        assert str(excinfo.value) == "tracker_type must be one of ['iou', 'mosse']"

    def test_no_tags(self, create_image, tracker):
        img1 = create_image(SIZE)

        inputs = {
            "img": img1,
            "bboxes": np.empty((0, 4), dtype=np.float32),
            "bbox_scores": np.empty((0), dtype=np.float32),
        }
        outputs = tracker.run(inputs)

        assert not outputs["obj_tags"]

    def test_tracking_ids_should_be_consistent_across_frames(
        self, tracker, test_human_video_sequences
    ):
        _, detections = test_human_video_sequences
        prev_tags = []
        for i, inputs in enumerate(detections):
            outputs = tracker.run(inputs)
            assert len(outputs["obj_tags"]) == len(inputs["bboxes"])
            if i > 0:
                assert outputs["obj_tags"] == prev_tags
            prev_tags = outputs["obj_tags"]

    def test_should_track_new_detection(self, tracker, test_human_video_sequences):
        _, detections = test_human_video_sequences
        # Add a new detection at the specified SEQ_IDX
        detections[SEQ_IDX]["bboxes"] = np.append(
            detections[SEQ_IDX]["bboxes"],
            [[0.1, 0.2, 0.3, 0.4]],
            axis=0,
        )
        detections[SEQ_IDX]["bbox_scores"] = np.append(
            detections[SEQ_IDX]["bbox_scores"], [0.4], axis=0
        )
        prev_tags = []
        for i, inputs in enumerate(detections):
            outputs = tracker.run(inputs)
            assert len(outputs["obj_tags"]) == len(inputs["bboxes"])
            # Special handling of comparing tag during and right after
            # seq_idx since a detection got added and removed
            if i == SEQ_IDX:
                assert outputs["obj_tags"] == prev_tags + ["2"]
            elif i == SEQ_IDX + 1:
                assert outputs["obj_tags"] == prev_tags[:-1]
            elif i > 0:
                assert outputs["obj_tags"] == prev_tags
            prev_tags = outputs["obj_tags"]

    def test_should_remove_lost_tracks(
        self, tracking_config, test_human_video_sequences
    ):
        """This only applies to IOU Tracker.

        NOTE: We are manually making a track to be lost since we don't
        have enough frames for it to occur naturally.
        """
        _, detections = test_human_video_sequences
        # Add a new detection at the specified SEQ_IDX
        detections[SEQ_IDX]["bboxes"] = np.append(
            detections[SEQ_IDX]["bboxes"],
            [[0.1, 0.2, 0.3, 0.4]],
            axis=0,
        )
        detections[SEQ_IDX]["bbox_scores"] = np.append(
            detections[SEQ_IDX]["bbox_scores"], [0.4], axis=0
        )
        tracking_config["tracking_type"] = "iou"
        tracker = Node(tracking_config)
        prev_tags = []
        for i, inputs in enumerate(detections):
            # Set the track which doesn't have a detection to be "lost"
            # by setting `lost > max_lost`
            if i == SEQ_IDX + 1:
                tracker.tracker.tracker.tracks[2].lost = (
                    tracker.tracker.tracker.max_lost + 1
                )
            outputs = tracker.run(inputs)
            assert len(outputs["obj_tags"]) == len(inputs["bboxes"])
            # This happens to be true for the test case, not a guaranteed
            # behaviour during normal operation.
            assert len(tracker.tracker.tracker.tracks) == len(inputs["bboxes"])
            # Special handling of comparing tag during and right after
            # seq_idx since a detection got added and removed
            if i == SEQ_IDX:
                assert outputs["obj_tags"] == prev_tags + ["2"]
            elif i == SEQ_IDX + 1:
                assert outputs["obj_tags"] == prev_tags[:-1]
            elif i > 0:
                assert outputs["obj_tags"] == prev_tags
            prev_tags = outputs["obj_tags"]

    def test_should_remove_update_failures(
        self, tracking_config, test_human_video_sequences
    ):
        """This only applies to OpenCV Tracker.

        NOTE: We are manually making a track to be lost since we don't
        have enough frames for it to occur naturally.

        NOTE: The bbox modification only applies to the two_people_crossing
        video sequence.
        """
        sequence_name, detections = test_human_video_sequences
        if sequence_name != "two_people_crossing":
            return
        # Add a new detection at the specified SEQ_IDX
        # This is the bbox of a small road divider that gets occluded the
        # next frame
        detections[SEQ_IDX]["bboxes"] = np.append(
            detections[SEQ_IDX]["bboxes"],
            [[0.65, 0.45, 0.7, 0.5]],
            axis=0,
        )
        detections[SEQ_IDX]["bbox_scores"] = np.append(
            detections[SEQ_IDX]["bbox_scores"], [0.4], axis=0
        )
        tracking_config["tracking_type"] = "mosse"
        tracker = Node(tracking_config)
        prev_tags = []
        for i, inputs in enumerate(detections):
            # Set the track which doesn't have a detection to be "lost"
            # by setting `lost > max_lost`
            outputs = tracker.run(inputs)
            assert len(outputs["obj_tags"]) == len(inputs["bboxes"])
            # This happens to be true for the test case, not a guaranteed
            # behaviour during normal operation.
            assert len(tracker.tracker.tracker.tracks) == len(inputs["bboxes"])
            # Special handling of comparing tag during and right after
            # seq_idx since a detection got added and removed
            if i == SEQ_IDX:
                assert outputs["obj_tags"] == prev_tags + ["2"]
            elif i == SEQ_IDX + 1:
                assert outputs["obj_tags"] == prev_tags[:-1]
            elif i > 0:
                assert outputs["obj_tags"] == prev_tags
            prev_tags = outputs["obj_tags"]
