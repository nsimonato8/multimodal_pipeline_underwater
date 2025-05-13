"""
    Image wraps the image, its path, its object detection, its object segmentation and other useful stuff.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np
import cv2
import logging


@dataclass
class Image:
    image: np.ndarray
    original: Optional[np.ndarray]
    path: Optional[Path] = field(default_factory=Path)
    object_segmentation: Optional[Dict[str, str]] = field(default_factory=dict)
    object_detection: Optional[Dict[str, str]] = field(default_factory=dict)

    @staticmethod
    def from_workflow_result(
        result: Dict[str, Any], image: np.ndarray, original: np.ndarray
    ):
        def parse_detection_results(result: Dict[str, Any]) -> Dict[str, Any]:
            return result.get("model", {}).get("parsed_output", {})

        def parse_segmentation_results(result: Dict[str, Any]) -> Dict[str, Any]:
            return result.get("model_1", {})

        return Image(
            image=image,
            original=original,
            object_detection=parse_detection_results(result),
            object_segmentation=parse_segmentation_results(result),
        )


def check_image_integrity(img: Path) -> bool:
    try:
        return cv2.imread(img) is not None
    except:
        return False


def select_best_image(images: List[Image], return_path=False) -> Optional[Image]:

    logger = logging.getLogger("image")

    if not images:
        logger.info("No images to select from.")
        return None

    idx = np.argmax([compute_psnr(image.original, image.image) for image in images])

    if return_path:
        return images[idx], images[idx].path

    return images[idx]


def compute_psnr(image1: np.ndarray, image2: np.ndarray) -> float:
    # Convert images from RGB to YCbCr
    image1_ycbcr = cv2.cvtColor(image1, cv2.COLOR_RGB2YCrCb)
    image2_ycbcr = cv2.cvtColor(image2, cv2.COLOR_RGB2YCrCb)

    # Use only the Y (luminance) channel for PSNR computation
    y1 = image1_ycbcr[:, :, 0]
    y2 = image2_ycbcr[:, :, 0]

    # Compute Mean Squared Error (MSE) on the Y channel
    mse = ((y1 - y2) ** 2).mean()
    if mse == 0:
        return float("inf")

    # Compute PSNR
    max_pixel_value = 255.0
    return 20 * np.log10(max_pixel_value / np.sqrt(mse))
