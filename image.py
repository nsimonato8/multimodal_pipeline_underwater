"""
    Image wraps the image, its path, its object detection, its object segmentation and other useful stuff.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import cv2


@dataclass
class Image:
    path: Path
    object_segmentation: Dict[str, str]
    object_detection: Dict[str, str]
    image: np.ndarray

    @staticmethod
    def from_workflow_result(workflow_result: List[Dict[str, Any]]) -> Image:
        raise NotImplementedError("from_workflow_result is not implemented yet")


def check_image_integrity(img: Path) -> bool:
    try:
        return cv2.imread(img) is not None
    except:
        return False
