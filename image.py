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
    def from_workflow_result(result: Dict[str, Any], image: np.ndarray) -> Image:
        def parse_detection_results(result: Dict[str, Any]) -> Dict[str, Any]:
            return result.get("model",{}).get("parsed_output",{})    

        def parse_segmentation_results(result: Dict[str, Any]) -> Dict[str, Any]:   
            return result.get("model_1",{})

        return Image(image=image,
                    object_detection=parse_detection_results(result),
                    object_segmentation=parse_segmentation_results(result))


def check_image_integrity(img: Path) -> bool:
    try:
        return cv2.imread(img) is not None
    except:
        return False
