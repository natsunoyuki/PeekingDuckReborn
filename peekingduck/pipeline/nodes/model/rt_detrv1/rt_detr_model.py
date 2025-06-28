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

"""RT-DETR models with model types: rtdetr_r18vd, rtdetr_r34vd, rtdetr_r50vd,
rtdetr_r101vd, rtdetr_r18vd_coco_o365, rtdetr_r50vd_coco_o365, 
rtdetr_r101vd_coco_o365."""

import logging
from typing import Any, Dict, List, Tuple

from pathlib import Path
import numpy as np

from peekingduck.pipeline.nodes.base import (
    ThresholdCheckerMixin,
    WeightsDownloaderMixin,
)
from peekingduck.pipeline.nodes.model.rt_detrv1.rt_detr_files.detector import Detector


HF_REPO = "PekingU"


class RTDETRModel(ThresholdCheckerMixin, WeightsDownloaderMixin):
    """Validates configuration, loads RT-DETR model, and performs inference.

    Configuration options are validated to ensure they have valid types and
    values. Model weights files are downloaded if not found in the location
    indicated by the `weights_dir` configuration option.

    Attributes:
        class_names (List[str]): Human-friendly class names of the object
            categories.
        detect_ids (List[int]): List of selected object category IDs. IDs not
            found in this list will be filtered away from the results. An empty
            list indicates that all object categories should be detected.
        detector (Detector): RT-DETR detector object to infer bboxes from a
            provided image frame.
    """
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.logger = logging.getLogger(__name__)

        self.check_bounds(["score_threshold"], "[0, 1]")

        local_weights_path = self.config.get("local_weights_path", None)
        if local_weights_path is None:
            if self.config.get("huggingface", True) is True:
                # Download pre-trained weights from HuggingFace.
                model_dir = Path(self.config.get("huggingface_model_dir", HF_REPO))
                # Account for windows `\`, as HF repos use only `/`.
                model_path = str(model_dir / self.config["model_type"]).replace("\\", "/")
            else:
                # Load HuggingFace pre-trained weights from a local directory.
                model_dir = self._find_paths()
                model_path = model_dir / self.config["model_type"]
        else:
            # Absolute path to custom local weights directory.
            model_path = Path(local_weights_path)

        self.detect_ids = self.config["detect"]

        self.detector = Detector(
            model_path,
            self.detect_ids,
            self.config["input_size"],
            self.config["score_threshold"],
        )


    @property
    def detect_ids(self) -> List[int]:
        """The list of selected object category IDs."""
        return self._detect_ids


    @detect_ids.setter
    def detect_ids(self, ids: List[int]) -> None:
        if not isinstance(ids, list):
            raise TypeError("detect_ids has to be a list")
        self._detect_ids = ids


    def predict(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predicts bboxes from image.

        Args:
            image (np.ndarray): Input image frame.

        Returns:
            (Tuple[np.ndarray, np.ndarray, np.ndarray]): Returned tuple
            contains:
            - An array of detection bboxes
            - An array of human-friendly detection class names
            - An array of detection scores

        Raises:
            TypeError: The provided `image` is not a numpy array.
        """
        if not isinstance(image, np.ndarray):
            raise TypeError("image must be a np.ndarray")
        return self.detector.predict_object_bbox_from_image(image)
