"""Detector module to predict object bbox from an image using RT-DETR."""

import logging
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import torch
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

from peekingduck.pipeline.utils.bbox.transforms import xyxy2xyxyn


class Detector:  # pylint: disable=too-many-instance-attributes
    """Object detection class using RE-DETR to predict object bboxes.

    Attributes:
        logger (logging.Logger): Events logger.
        config (Dict[str, Any]): RT-DETR node configuration.
        model_dir (pathlib.Path): Path to directory of model weights files.
        device (torch.device): Represents the device on which the torch.Tensor
            will be allocated.
        model (RTDetrForObjectDetection): The RT-DETR model for performing inference.
        image_processor (RTDetrImageProcessor): The RT-DETR image processor.
    """
    def __init__(  # pylint: disable=too-many-arguments
        self,
        model_dir: Union[Path, str],
        detect_ids: List[int],
        model_type: str,
        input_size: int=640,
        score_threshold: float=0.5,
    ) -> None:
        self.logger = logging.getLogger(__name__)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model_dir = Path(model_dir)
        self.model_type = model_type
        self.model_path = str(self.model_dir / model_type)

        self.detect_ids = detect_ids

        self.input_size = (input_size, input_size)
        self.score_threshold = score_threshold

        self.model, self.image_processor = self.create_rtdetr_model()
        self.id2label = self.model.config.id2label


    @torch.no_grad()
    def predict_object_bbox_from_image(
        self, image: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Detects bounding boxes of selected object categories from an image.

        The input image is first scaled according to the `input_size`
        configuration option. Detection results will be filtered according to
        `iou_threshold`, `score_threshold`, and `detect_ids` configuration
        options. Bounding boxes coordinates are then normalized w.r.t. the
        input `image` size.

        Args:
            image (np.ndarray): Input image.

        Returns:
            (Tuple[np.ndarray, np.ndarray, np.ndarray]): Returned tuple
            contains:
            - An array of detection bboxes
            - An array of human-friendly detection class names
            - An array of detection scores
        """
        # Store the original image size to normalize bbox later
        image_shape = image.shape[:2]

        inputs = self.image_processor(images=image, return_tensors="pt")
        inputs = inputs.to(self.device)
        
        self.model = self.model.to(self.device)
        with torch.no_grad():
            result = self.model(**inputs)

        result = self.image_processor.post_process_object_detection(
            result, 
            target_sizes=torch.tensor([(image_shape[0], image_shape[1])]), 
            threshold=self.score_threshold,
        )[0]
        bboxes = result["boxes"].detach().cpu().numpy()
        classes = result["labels"].detach().cpu().numpy()
        scores = result["scores"].detach().cpu().numpy()

        want = np.isin(classes, self.detect_ids)
        bboxes = bboxes[want]
        classes = classes[want]
        scores = scores[want]

        bboxes = xyxy2xyxyn(bboxes, image_shape[0], image_shape[1])
        classes = np.array([self.id2label[c] for c in classes])
        
        return bboxes, classes, scores


    def create_rtdetr_model(self) -> Tuple[RTDetrForObjectDetection, RTDetrImageProcessor]:
        """Creates a RT-DETR model and loads its weights. Also loads the image
        processor required to preprocess and postprocess the inference results.

        Creates `detect_ids` as a `torch.Tensor`. Sets up `input_size` to a
        square shape. Logs model configurations.

        Returns:
            (RTDetrForObjectDetection): RE-DETR model.
        """
        self.logger.info(
            "RE-DETR model loaded with the following configs:\n\t"
            f"Model type: {self.model_type}\n\t"
            f"Input resolution: {self.input_size}\n\t"
            f"IDs being detected: {self.detect_ids}\n\t"
            f"Score threshold: {self.score_threshold}\n\t"
        )
        return (
            RTDetrForObjectDetection.from_pretrained(self.model_path),
            RTDetrImageProcessor.from_pretrained(self.model_path),
        )
