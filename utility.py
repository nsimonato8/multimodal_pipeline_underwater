from typing import List, Dict
from pathlib import Path
from tqdm import tqdm
import cv2
import logging
import os
import json
import tqdm

from errors import InputError, ProcessingError
from image import Image, check_image_integrity
from artifact import Artifact


def load_prompts(prompt_folder: Path) -> Dict[str, str]:
    prompt_files = ['Artifacts.txt', 'Historical.txt', 'GeographicalEnvironment.txt']

    if not all(list(map(lambda x: os.path.exists(prompt_folder / x), prompt_files))):
        raise InputError(
            f"Expected 3 prompt files in {prompt_folder}."
        )

    try:
        prompts = []
        for file in prompt_files:  # Take first 3 prompt files
            with open(prompt_folder / file, "r", encoding="utf-8") as f:
                prompts.append(f.read().strip())

        DETECTION_PROMPT = '\n'.join(prompts)

        with open(prompt_folder / "ClassificationSchema.txt", "r", encoding="utf-8") as f:
            classification_schema = f.read().strip()

        return {
            "DETECTION_PROMPT": DETECTION_PROMPT,
            "CLASSIFICATION_PROMPT": classification_schema,
        }
    except Exception as e:
        raise InputError(f"Error loading prompt files: {str(e)}")


def get_image_paths(input_path: Path) -> List[Path]:
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    image_paths = [
        p
        for p in input_path.iterdir()
        if p.is_file()
        and p.suffix.lower() in image_extensions
        and check_image_integrity(p)
    ]

    if not image_paths:
        raise InputError(f"No images found in {input_path}")

    # Sort images by name to maintain sequence
    image_paths.sort()

    logging.info(f"Found {len(image_paths)} images in {input_path}")
    return image_paths


def load_frames(input_folder: Path) -> List[Image]:
    image_paths = get_image_paths(input_folder)

    images = []
    for path in image_paths:
        try:
            img = Image(path=path, image=cv2.imread(str(path)))
            if img.image is not None:
                images.append(img)
        except Exception as e:
            logging.error(f"Error loading image {path}: {str(e)}")

    if not images:
        raise InputError(f"No valid images found in {input_folder}")

    return images


def save_results(
    artifacts: List[Artifact],
    output_dir: Path,
) -> None:
    try:
        output_dir.mkdir(exist_ok=True, parents=True)

        for artifact in tqdm.tqdm(artifacts):
            artifact_dir = output_dir / artifact.name
            artifact_dir.mkdir(exist_ok=True)

        logging.info(f"Saving artifacts to {output_dir}")
        
        for i, artifact in tqdm(enumerate(artifacts)):
           artifact.to_pickle(saving_path=output_dir / f"artifact{i}.pickle")

        logging.info(f"Results saved to {output_dir}")
    except Exception as e:
        raise ProcessingError(f"Error saving results: {str(e)}")


def extract_frames_from_video(
    video_path: Path, output_folder: Path, sample_rate: int
) -> None:
    try:
        # Ensure output folder exists
        output_folder.mkdir(parents=True, exist_ok=True)

        # Open video file
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise InputError(f"Cannot open video file: {video_path}")

        frame_count = 0
        saved_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break  # End of video

            # Save every nth frame
            if frame_count % sample_rate == 0:
                frame_name = f"frame_{saved_count:03d}.png"
                frame_path = output_folder / frame_name
                cv2.imwrite(str(frame_path), frame)
                saved_count += 1

            frame_count += 1

        cap.release()
        logging.info(f"Extracted {saved_count} frames from {video_path}")

    except InputError as e:
        logging.error(str(e))
        raise
    except Exception as e:
        raise ProcessingError(f"Error extracting frames: {str(e)}")
