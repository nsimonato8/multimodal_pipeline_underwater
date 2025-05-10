import cv2
from concurrent.futures import ThreadPoolExecutor
from typing import List, Callable
from image import Image
from preprocessing2 import simple_underwater_correction
from deprecated import deprecated


@deprecated(reason="Use simple_underwater_correction instead")
def _preprocess_image(image: Image) -> Image:
    """
    Performs histogram equalization and radial distortion correction on the input image.

    Args:
        image (Image): Input image to be processed.

    Returns:
        Image: Processed image with histogram equalization and radial distortion correction applied.
    """
    try:
        equalized_image = histogram_equalization(image)  # Apply histogram equalization

        # Get image dimensions
        height, width = equalized_image.image.shape[:2]

        # Ensure the image is large enough for cropping
        if height < 1500 or width < 1500:
            raise ValueError(
                "Image dimensions must be at least 1500x1500 for radial distortion correction."
            )

        # Crop to the central 1500x1500 pixels
        start_x = (width - 1500) // 2
        start_y = (height - 1500) // 2
        cropped_image = equalized_image.image[
            start_y : start_y + 1500, start_x : start_x + 1500
        ]

        image.image = (
            cropped_image  # Update the image attribute with the processed image
        )

        return image

    except Exception as e:
        raise RuntimeError(f"An error occurred during image preprocessing: {e}")


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
    images: List[Image], preprocessing: Callable = _preprocess_image
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


def histogram_equalization(image: Image) -> Image:

    img = image.image

    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

    # equalize the histogram of the Y channel
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])

    # convert the YUV image back to RGB format
    image.image = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    return image
