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

"""Object detection class using YOLOv4 model to detect human faces."""

import logging
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants


class Detector:  # pylint: disable=too-few-public-methods,too-many-instance-attributes
    """Object detection class using yolo model to find human faces."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        model_dir: Path,
        class_names: List[str],
        detect_ids: List[int],
        model_type: str,
        model_file: Dict[str, str],
        max_output_size_per_class: int,
        max_total_size: int,
        input_size: int,
        iou_threshold: float,
        score_threshold: float,
    ) -> None:
        self.logger = logging.getLogger(__name__)

        self.class_names = class_names
        self.model_type = model_type
        self.model_path = model_dir / model_file[self.model_type]

        self.max_output_size_per_class = max_output_size_per_class
        self.max_total_size = max_total_size
        self.input_size = (input_size, input_size)
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold

        self.detect_ids = detect_ids
        self.yolo = self._create_yolo_model()

    def predict_object_bbox_from_image(
        self, image: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predicts face bboxes, labels and scores.

        Args:
            image (np.ndarray): Input image.

        Returns:
            bboxes (np.ndarray): Detected bboxes
            labels (np.ndarray): Class labels for each detected bbox.
            scores (np.ndarray): Confidence scores for each detected bbox.
        """
        image = self._preprocess(image)

        pred = self.yolo(tf.constant(image))
        pred = next(iter(pred.values()))

        bboxes, scores, classes = self._postprocess(pred[:, :, :4], pred[:, :, 4:])
        labels = np.array([self.class_names[int(i)] for i in classes])

        return bboxes, labels, scores

    def _create_yolo_model(self) -> Callable:
        self.logger.info(
            "Yolo model loaded with following configs:\n\t"
            f"Model type: {self.model_type},\n\t"
            f"Input resolution: {self.input_size},\n\t"
            f"IDs being detected: {self.detect_ids},\n\t"
            f"Max detections per class: {self.max_output_size_per_class},\n\t"
            f"Max total detections: {self.max_total_size},\n\t"
            f"IOU threshold: {self.iou_threshold},\n\t"
            f"Score threshold: {self.score_threshold}"
        )

        return self._load_yolo_weights()

    def _load_yolo_weights(self) -> Callable:
        self.model = tf.saved_model.load(
            str(self.model_path), tags=[tag_constants.SERVING]
        )
        return self.model.signatures["serving_default"]

    def _postprocess(
        self,
        pred_boxes: tf.Tensor,
        pred_scores: tf.Tensor,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        bboxes, scores, classes, valid_dets = tf.image.combined_non_max_suppression(
            tf.reshape(pred_boxes, (tf.shape(pred_boxes)[0], -1, 1, 4)),
            tf.reshape(
                pred_scores, (tf.shape(pred_scores)[0], -1, tf.shape(pred_scores)[-1])
            ),
            self.max_output_size_per_class,
            self.max_total_size,
            self.iou_threshold,
            self.score_threshold,
        )
        num_valid = valid_dets[0]

        classes = classes.numpy()[0]
        classes = classes[:num_valid]
        # only identify objects we are interested in
        mask = np.isin(classes, self.detect_ids)

        scores = scores.numpy()[0]
        scores = scores[:num_valid]
        scores = scores[mask]

        bboxes = bboxes.numpy()[0]
        bboxes = bboxes[:num_valid]
        bboxes = bboxes[mask]

        # swapping x and y axes
        bboxes[:, [0, 1]] = bboxes[:, [1, 0]]
        bboxes[:, [2, 3]] = bboxes[:, [3, 2]]

        return bboxes, scores, classes

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        image = cv2.resize(image, self.input_size)
        image = np.asarray([image]).astype(np.float32) / 255.0

        return image
