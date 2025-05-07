"""
    Image wraps the image, its path, its object detection, its object segmentation and other useful stuff.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict
import numpy as np
import cv2


@dataclass
class Image:
    path: Path
    object_segmentation: Dict[str, str]
    object_detection: Dict[str, str]
    image: np.ndarray


def check_image_integrity(img: Path) -> bool:
    try:
        return cv2.imread(img) is not None
    except:
        return False
