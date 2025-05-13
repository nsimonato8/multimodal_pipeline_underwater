"""
    Artifact wraps the concept of archeological artifact. It collects the list of images extracted from the video, its name, inventory number, PLY/GLB encoding, best image, etc.
"""

from dataclasses import dataclass
from typing import List, Any, Optional
from pathlib import Path
from image import Image, select_best_image
import pickle
import cv2


@dataclass
class Artifact:
    images: List[Image]
    name: str
    best_image: Optional[Image] = None
    caption: Optional[str] = None
    model_3d: Optional[Any] = None

    def __post_init__(self):
        self.best_image, best_image_path = select_best_image(
            self.images, return_path=True
        )

        cv2.imwrite(
            "output/" + self.best_image.path.stem + "_processed.jpg",
            self.best_image.image,
        )

        print(f"Best image: {best_image_path}")

    def __repr__(self):
        return f"\nArtifact(name={self.name}, images={len(self.images)}, best_image={self.best_image.path if self.best_image else None}, caption={self.caption})\n"

    def to_pickle(self, saving_path: Path) -> None:
        with open(saving_path, "wb") as f:
            pickle.dump(self, f)
