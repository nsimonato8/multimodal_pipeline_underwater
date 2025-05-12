import cv2
from concurrent.futures import ThreadPoolExecutor
from typing import List, Callable
from image import Image
from preprocessing2 import simple_underwater_correction


def preprocess_image(image: Image) -> Image:
    """
    Applies simple underwater correction to the input image.

    Args:
        image (Image): Input image to be processed.

    Returns:
        Image: Processed image with simple underwater correction applied.
    """
    try:
        image.image = simple_underwater_correction(
            image.image
        )  # Update the image attribute with the processed image
        return image
    except Exception as e:
        raise RuntimeError(f"An error occurred during image preprocessing: {e}")


def preprocess_images_parallel(
    images: List[Image], preprocessing: Callable = preprocess_image
) -> List[Image]:
    """
    Runs the preprocess_image function in parallel using multithreading.

    Args:
        images (List[Any]): List of input images to be processed.

    Returns:
        List[Any]: List of processed images.
    """

    with ThreadPoolExecutor() as executor:
        results = executor.map(preprocessing, images)

    return list(results)
