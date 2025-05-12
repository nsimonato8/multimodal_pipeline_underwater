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
import cv2
import traceback
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
    "--prompt",
    "prompt_dir",
    required=True,
    type=click.Path(exists=True),
    help="Path to directory containing prompt text files",
)
@click.option(
    "--output",
    "output_dir",
    required=True,
    type=click.Path(),
    help="Path to output directory for results",
)
@click.option(
    "--is-video/",
    default=False,
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
        client = InferenceHTTPClient(
            api_url=os.getenv("ROBOFLOW_API_URL", "http://localhost:9001"),
            api_key=os.getenv("ROBOFLOW_API_KEY", ""),
        )
        logger.debug("Inference client created")
        input_path = Path(input_path)
        prompt_dir = Path(prompt_dir)
        output_dir = Path(output_dir)

        logger.info("Starting image processing pipeline")

        # 1. Load prompts
        prompts: dict = load_prompts(prompt_dir)
        logger.info(f"Loaded {len(prompts)} prompts")

        # 2. Get image paths
        if is_video:
            logger.info("Starting frame extraction...")
            extract_frames_from_video(input_path, output_dir, sample_rate)
            logger.info("Frame extraction performed successfully.")

        # 2 Load and Pre-Process images locally
        logger.info("Starting frame loading...")
        frames: List[Image] = load_frames(input_path)
        logger.info("Frame loading performed successfully.")
        logger.info("Starting frame preprocessing...")
        frames: List[Image] = preprocess_images_parallel(frames)
        logger.info("Frame preprocessing performed successfully.")

        # 3 Process images with Roboflow workflow
        logger.info("Starting detect_and_segmentation_workflow...")
        workflow_results: List[Image] = detect_and_segmentation_workflow(
            client, frames, prompts.get("DETECTION_PROMPT", "")
        )
        logger.info("detect_and_segmentation_workflow executed successfully.")

        # 4. Select frames
        logger.info("Starting frame_selection...")
        artifacts: List[Artifact] = frame_selection(workflow_results)
        logger.info(
            f"Artifacts identified: {type(artifacts)}, {len(artifacts)}, {type(artifacts[0])}"
        )
        logger.info("frame_selection executed successfully.")

        # 5. Generate frame descriptions
        logger.info("Starting generate_frame_description...")
        artifacts: List[Artifact] = generate_frame_description(client, artifacts)
        logger.info("generate_frame_description executed successfully.")

        # 6. Save results
        logger.info("Starting save_results...")
        save_results(artifacts, output_dir)
        logger.info("save_results executed successfully.")

        # 7. Reconstruction with WaterSplatting technique
        # TODO: define input arguments for "reconstruct_image" (this will be implemented in new versions, for now it's performed manually).

        logger.info("Pipeline completed.")

    except PipelineError as e:
        logger.error(f"Pipeline error: {str(e)}")
        sys.exit(1)
    except:
        logger.error(f"Unexpected error: {traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    main()
