"""
    Image wraps the image, its path, its object detection, its object segmentation and other useful stuff.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict
import PIL

@dataclass
class Image:
	path: Path
    object_segmentation: Dict[str, str]
    object_detection: Dict[str, str]
	image: PIL.image
	
	
def check_image_integrity(img: Path) -> bool:
	try:
		PIL.Image.load(path).verify()
		return True
	except:
		return False