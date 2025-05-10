#!/usr/bin/env python3
"""
Image Processing Pipeline

This script implements a command-line interface for an image processing pipeline
that processes video frames or images using Roboflow and applies custom analysis functions.
"""

import os
import sys
import click
import logging
from typing import List
from pathlib import Path
from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient

from image import Image
from artifact import Artifact
from errors import *
from utility import (
    load_prompts,
    save_results,
    load_frames,
    extract_frames_from_video,
)
from preprocessing import preprocess_images_parallel
from processing import (
    detect_and_segmentation_workflow,
    frame_selection,
    generate_frame_description,
)
from reconstruction import reconstruct_image

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("multimodal_pipeline")


@click.command()
@click.option(
    "--input",
    "input_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to input video file or directory of image frames",
)
@click.option(
    "--prompt-dir",
    "prompt_dir",
    required=True,
    type=click.Path(exists=True),
    help="Path to directory containing prompt text files",
)
@click.option(
    "--output-dir",
    "output_dir",
    required=True,
    type=click.Path(),
    help="Path to output directory for results",
)
@click.option(
    "--is-video/",
    default=True,
    help="Specify if input is a video file or a directory of frames",
)
@click.option(
    "--sample-rate",
    default=1,
    type=int,
    help="If input is video, extract every nth frame",
)
def main(
    input_path: str,
    prompt_dir: str,
    output_dir: str,
    is_video: bool,
    sample_rate: int,
) -> None:
    load_dotenv()
    try:
        global client
        client = InferenceHTTPClient(
            api_url=os.getenv("ROBOFLOW_API_URL", "http://localhost:9001"),
            api_key=os.getenv("ROBOFLOW_API_KEY", ""),
        )
        input_path = Path(input_path)
        prompt_dir = Path(prompt_dir)
        output_dir = Path(output_dir)

        logger.info("Starting image processing pipeline")

        # 1. Load prompts
        prompts: dict = load_prompts(prompt_dir)
        logger.info(f"Loaded {len(prompts)} prompts")

        # 2. Get image paths
        if is_video:
            extract_frames_from_video(input_path, output_dir, sample_rate)

        # 2 Load and Pre-Process images locally
        frames: List[Image] = load_frames(input_path)
        frames: List[Image] = preprocess_images_parallel(frames)

        # 3 Process images with Roboflow workflow
        workflow_results: List[Image] = detect_and_segmentation_workflow(frames, prompts.get("DETECTION_PROMPT", ""))

        # 4. Select frames
        artifacts: List[Artifact] = frame_selection(workflow_results)

        # 5. Generate frame descriptions
        artifacts: List[Artifact] = generate_frame_description(artifacts)

        # 6. Save results
        save_results(artifacts, output_dir) # TODO: Change save_results according to the new updates.

        # 7. Reconstruction with WaterSplatting technique
        # TODO: define input arguments for "reconstruct_image" (this will be implemented in new versions, for now it's performed manually).

        logger.info("Pipeline completed.")

    except PipelineError as e:
        logger.error(f"Pipeline error: {str(e)}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
