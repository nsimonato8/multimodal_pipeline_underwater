"""
    Image wraps the image, its path, its object detection, its object segmentation and other useful stuff.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import cv2


@dataclass
class Image:
    image: np.ndarray
    path: Optional[Path] = field(default_factory=Path)
    object_segmentation: Optional[Dict[str, str]] = field(default_factory=dict)
    object_detection: Optional[Dict[str, str]] = field(default_factory=dict)

    @staticmethod
    def from_workflow_result(result: Dict[str, Any], image: np.ndarray):
        def parse_detection_results(result: Dict[str, Any]) -> Dict[str, Any]:
            return result.get("model", {}).get("parsed_output", {})    

        def parse_segmentation_results(result: Dict[str, Any]) -> Dict[str, Any]:   
            return result.get("model_1", {})

        return Image(image=image,
                     object_detection=parse_detection_results(result),
                     object_segmentation=parse_segmentation_results(result))


def check_image_integrity(img: Path) -> bool:
    try:
        return cv2.imread(img) is not None
    except:
        return False
