import cv2
from typing import Any
from concurrent.futures import ThreadPoolExecutor
from typing import List
from image import Image

def preprocess_image(image: Image) -> Image:
    """
    Performs histogram equalization and radial distortion correction on the input image.

    Args:
        image (Image): Input image to be processed.

    Returns:
        Image: Processed image with histogram equalization and radial distortion correction applied.
    """
    try:
        # Convert to grayscale if the image is not already
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image.image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image.image

        # Apply histogram equalization
        equalized_image = cv2.equalizeHist(gray_image)

        # Get image dimensions
        height, width = equalized_image.shape[:2]

        # Ensure the image is large enough for cropping
        if height < 1500 or width < 1500:
            raise ValueError("Image dimensions must be at least 1500x1500 for radial distortion correction.")

        # Crop to the central 1500x1500 pixels
        start_x = (width - 1500) // 2
        start_y = (height - 1500) // 2
        cropped_image = equalized_image[start_y:start_y + 1500, start_x:start_x + 1500]

        image.image = cropped_image  # Update the image attribute with the processed image

        return image

    except Exception as e:
        raise RuntimeError(f"An error occurred during image preprocessing: {e}")
    

def preprocess_images_parallel(images: List[Image], preprocessing: function=preprocess_image) -> List[Image]:
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
    