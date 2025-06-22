import numpy as np


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