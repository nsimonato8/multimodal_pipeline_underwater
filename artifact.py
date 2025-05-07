"""
    Artifact wraps the concept of archeological artifact. It collects the list of images extracted from the video, its name, inventory number, PLY/GLB encoding, best image, etc.
"""

from dataclasses import dataclass
from typing import List, Any
from image import Image


@dataclass
class Artifact:
    images: List[Image]
    name: str
    best_image: Image
    model_3d: Any
