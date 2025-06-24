# Copyright 2025 Natsunoyuki AI Laboratory
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

from pathlib import Path
from unittest import TestCase

import numpy as np
import pytest
from typeguard import TypeCheckError

from peekingduck.pipeline.nodes.dabble.bot_sort import Node


# Frame index for manual manipulation of detections to trigger some
# branches
SEQ_IDX = 3
SIZE = (400, 600, 3)


@pytest.fixture(params=[-0.1, 1.1])
def invalid_threshold(request):
    yield request.param


@pytest.fixture(params=[-1])
def invalid_positive_value(request):
    yield request.param


@pytest.fixture(params=[1.23])
def invalid_int_value(request):
    yield request.param


@pytest.fixture
def bot_sort_config():
    return {
        "root": Path.cwd(),
        "input": ["img", "bboxes", "bbox_labels", "bbox_scores"],
        "output": ["obj_attrs", "bboxes", "bbox_labels", "bbox_scores"],
        "track_high_thresh": 0.6,
        "track_low_thresh": 0.1,
        "new_track_thresh": 0.7,
        "match_thresh": 0.8,
        "track_buffer": 30,
        "frame_rate": 30,
    }


@pytest.fixture
def tracker(bot_sort_config):
    node = Node(bot_sort_config)
    return node


class TestBotSortTracking:
    def test_should_raise_for_invalid_track_high_thresh(
        self, bot_sort_config, invalid_threshold
    ):
        bot_sort_config["track_high_thresh"] = invalid_threshold
        with pytest.raises(ValueError) as excinfo:
            _ = Node(bot_sort_config)
        assert str(excinfo.value) == "track_high_thresh must be between [0.0, 1.0]"


    def test_should_raise_for_invalid_track_low_thresh(
        self, bot_sort_config, invalid_threshold
    ):
        bot_sort_config["track_low_thresh"] = invalid_threshold
        with pytest.raises(ValueError) as excinfo:
            _ = Node(bot_sort_config)
        assert str(excinfo.value) == "track_low_thresh must be between [0.0, 1.0]"


    def test_should_raise_for_invalid_new_track_thresh(
        self, bot_sort_config, invalid_threshold
    ):
        bot_sort_config["new_track_thresh"] = invalid_threshold
        with pytest.raises(ValueError) as excinfo:
            _ = Node(bot_sort_config)
        assert str(excinfo.value) == "new_track_thresh must be between [0.0, 1.0]"


    def test_should_raise_for_invalid_match_thresh(
        self, bot_sort_config, invalid_threshold
    ):
        bot_sort_config["match_thresh"] = invalid_threshold
        with pytest.raises(ValueError) as excinfo:
            _ = Node(bot_sort_config)
        assert str(excinfo.value) == "match_thresh must be between [0.0, 1.0]"


    def test_should_raise_for_negative_track_buffer(
        self, bot_sort_config, invalid_positive_value
    ):
        bot_sort_config["track_buffer"] = invalid_positive_value
        with pytest.raises(ValueError) as excinfo:
            _ = Node(bot_sort_config)
        assert str(excinfo.value) == "track_buffer must be between [1.0, inf)"


    def test_should_raise_for_non_int_track_buffer(
        self, bot_sort_config, invalid_int_value
    ):
        bot_sort_config["track_buffer"] = invalid_int_value
        with pytest.raises(TypeCheckError) as excinfo:
            _ = Node(bot_sort_config)
        assert str(excinfo.value) == "float is not an instance of int"


    def test_should_raise_for_negative_frame_rate(
        self, bot_sort_config, invalid_positive_value
    ):
        bot_sort_config["frame_rate"] = invalid_positive_value
        with pytest.raises(ValueError) as excinfo:
            _ = Node(bot_sort_config)
        assert str(excinfo.value) == "frame_rate must be between [1.0, inf)"


    def test_should_raise_for_non_int_frame_rate(
        self, bot_sort_config, invalid_int_value
    ):
        bot_sort_config["frame_rate"] = invalid_int_value
        with pytest.raises(TypeCheckError) as excinfo:
            _ = Node(bot_sort_config)
        assert str(excinfo.value) == "float is not an instance of int"


    def test_no_tags(self, create_image, tracker):
        img1 = create_image(SIZE)

        inputs = {
            "img": img1, 
            "bboxes": np.empty((0, 4), dtype=np.float32),
            "bbox_labels": np.empty(4, dtype=np.int32), 
            "bbox_scores": np.empty(4, dtype=np.float32),
        }
        outputs = tracker.run(inputs)

        assert not outputs["obj_attrs"]["ids"]


    def test_tracking_ids_should_be_consistent_across_frames(
        self, tracker, human_video_sequence_2
    ):
        _, detections = human_video_sequence_2
        prev_tags = []
        for i, inputs in enumerate(detections):
            outputs = tracker.run(inputs)
            assert len(outputs["obj_attrs"]["ids"]) == len(inputs["bboxes"])
            if i > 0:
                assert outputs["obj_attrs"]["ids"] == prev_tags
            prev_tags = outputs["obj_attrs"]["ids"]


    def test_should_track_new_detection(self, tracker, human_video_sequence_2):
        _, detections = human_video_sequence_2
        # Add a new detection at the specified SEQ_IDX
        detections[SEQ_IDX]["bboxes"] = np.append(
            detections[SEQ_IDX]["bboxes"], [[0.1, 0.2, 0.3, 0.4]], axis=0
        )
        detections[SEQ_IDX]["bbox_labels"] = np.append(
            detections[SEQ_IDX]["bbox_labels"], [0], axis=0
        )
        detections[SEQ_IDX]["bbox_scores"] = np.append(
            detections[SEQ_IDX]["bbox_scores"], [0.9], axis=0
        )   

        prev_tags = []
        for i, inputs in enumerate(detections):
            outputs = tracker.run(inputs)
            assert len(outputs["obj_attrs"]["ids"]) == len(inputs["bboxes"])
            # Special handling of comparing tag during and right after
            # seq_idx since a detection got added and removed
            if i == SEQ_IDX:
                assert outputs["obj_attrs"]["ids"] == prev_tags + [4]
            elif i == SEQ_IDX + 1:
                assert outputs["obj_attrs"]["ids"] == prev_tags[:-1]
            elif i > 0:
                assert outputs["obj_attrs"]["ids"] == prev_tags
            prev_tags = outputs["obj_attrs"]["ids"]

    def test_should_remove_lost_tracks(self, bot_sort_config, human_video_sequence_2):
        """NOTE: We are manually making a track to be lost since we don't
        have enough frames for it to occur naturally.
        
        Please refer to the BoT-SORT source code to see how tracks are removed.
        https://github.com/natsunoyuki/BoT-SORT/blob/main/src/bot_sort/bot_sort.py#L282
        """
        _, detections = human_video_sequence_2
        # Add a new detection at the specified SEQ_IDX
        detections[SEQ_IDX]["bboxes"] = np.append(
            detections[SEQ_IDX]["bboxes"], [[0.1, 0.2, 0.3, 0.4]], axis=0
        )
        detections[SEQ_IDX]["bbox_labels"] = np.append(
            detections[SEQ_IDX]["bbox_labels"], [0], axis=0
        )
        detections[SEQ_IDX]["bbox_scores"] = np.append(
            detections[SEQ_IDX]["bbox_scores"], [0.9], axis=0
        )   

        tracker = Node(bot_sort_config)
        # Artificially shorten the max_time_lost for testing.
        tracker.tracker.tracker.tracker.max_time_lost = 1

        prev_tags = []
        for i, inputs in enumerate(detections):
            outputs = tracker.run(inputs)

            assert len(outputs["obj_attrs"]["ids"]) == len(inputs["bboxes"])
            # This happens to be true for the test case, not a guaranteed
            # behaviour during normal operation.
            assert len(tracker.tracker.tracker.tracker.tracked_stracks) == len(inputs["bboxes"])
            # Special handling of comparing tag during and right after
            # seq_idx since a detection got added and removed.
            if i == SEQ_IDX:
                assert outputs["obj_attrs"]["ids"] == prev_tags + [4]
            elif i == SEQ_IDX + 1:
                assert outputs["obj_attrs"]["ids"] == prev_tags[:-1]
                assert tracker.tracker.tracker.tracker.removed_stracks[0].track_id == 4
            elif i > 0:
                assert outputs["obj_attrs"]["ids"] == prev_tags
            prev_tags = outputs["obj_attrs"]["ids"]

    # TODO
    @pytest.mark.skip("Not implemented in the bot-sort source code yet.")
    def test_reset_model(self, tracker, human_video_sequence):
        mot_metadata = {"reset_model": True}
        _, detections = human_video_sequence
        prev_tags = []
        with TestCase.assertLogs(
            "peekingduck.pipeline.nodes.dabble.bot_sort.logger"
        ) as captured:
            for i, inputs in enumerate(detections):
                # Insert mot_metadata in input to signal a new model should be
                # created
                if i == 0:
                    inputs["mot_metadata"] = mot_metadata
                outputs = tracker.run(inputs)
                assert len(outputs["obj_attrs"]["ids"]) == len(inputs["bboxes"])
                if i == 0:
                    assert captured.records[0].getMessage() == (
                        f"Creating new {tracker.tracking_type} tracker..."
                    )
                if i > 0:
                    assert outputs["obj_attrs"]["ids"] == prev_tags
                prev_tags = outputs["obj_attrs"]["ids"]

    def test_handle_empty_detections(
        self, tracker, human_video_sequence_with_empty_frames_2
    ):
        _, detections = human_video_sequence_with_empty_frames_2
        for inputs in detections:
            outputs = tracker.run(inputs)
            assert len(outputs["obj_attrs"]["ids"]) == len(inputs["bboxes"])
