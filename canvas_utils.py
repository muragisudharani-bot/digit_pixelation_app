from typing import Any

import cv2
import numpy as np


def canvas_json_or_rgba_to_grayscale(canvas_data: Any) -> np.ndarray:
    """Convert drawable-canvas image data or RGBA JSON-like data to grayscale."""
    if canvas_data is None:
        return np.zeros((280, 280), dtype=np.uint8)

    array = np.asarray(canvas_data)
    if array.ndim == 2:
        return np.clip(array, 0, 255).astype(np.uint8)

    if array.ndim == 3:
        if array.shape[2] == 4:
            rgb = array[:, :, :3].astype(np.float32)
            alpha = array[:, :, 3:4].astype(np.float32) / 255.0
            gray = (0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]) * alpha
            return np.clip(gray, 0, 255).astype(np.uint8)
        if array.shape[2] == 3:
            return cv2.cvtColor(array.astype(np.uint8), cv2.COLOR_RGB2GRAY)

    return np.zeros((280, 280), dtype=np.uint8)
