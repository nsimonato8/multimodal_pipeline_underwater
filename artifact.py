"""
    Artifact wraps the concept of archeological artifact. It collects the list of images extracted from the video, its name, inventory number, PLY/GLB encoding, best image, etc.
"""

from dataclasses import dataclass
from typing import List, Any, Optional
from image import Image


@dataclass
class Artifact:
    images: List[Image]
    name: str
    best_image: Optional[Image] = None
    caption: Optional[str] = None
    model_3d: Optional[Any] = None

    def __post_init__(self):
        self.best_image = Artifact.select_best_image(self.images)

    @staticmethod
    def select_best_image(images: List[Image]) -> Optional[Image]:
        # TODO: Implement a better selection algorithm for the best image
        if images:
            return images[0]

        return None
