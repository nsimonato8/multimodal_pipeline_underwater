#!/usr/bin/env python3
"""
Image Processing Pipeline

This script implements a command-line interface for an image processing pipeline
that processes video frames or images using Roboflow and applies custom analysis functions.
"""

import os
import sys
import json
import click
import logging
from typing import List, Dict, Union, Tuple, Optional
from pathlib import Path
import requests
import cv2
from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient

from image import Image
from artifact import Artifact

from errors import *
from utility import load_prompts, get_image_paths, save_results
from processing import detect_images, segment_images, frame_selection, frame_description
from reconstruction import reconstruct_image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@click.command()
@click.option(
    "--input", 
    "input_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to input video file or directory of image frames"
)
@click.option(
    "--prompt-dir", 
    "prompt_dir",
    required=True,
    type=click.Path(exists=True),
    help="Path to directory containing prompt text files"
)
@click.option(
    "--output-dir", 
    "output_dir",
    required=True,
    type=click.Path(),
    help="Path to output directory for results"
)
@click.option(
    "--api-key", 
    required=True,
    help="Roboflow API key"
)
@click.option(
    "--model-endpoint", 
    required=True,
    help="Roboflow model endpoint URL"
)
@click.option(
    "--is-video/",
    default=True,
    help="Specify if input is a video file or a directory of frames"
)
@click.option(
    "--sample-rate",
    default=1,
    type=int,
    help="If input is video, extract every nth frame"
)
@click.option(
    "--batch-size",
    default=4,
    type=int,
    help="Number of images to process in parallel"
)
def main(
    input_path: str,
    prompt_dir: str,
    output_dir: str,
    api_key: str,
    model_endpoint: str,
    is_video: bool,
    sample_rate: int,
    batch_size: int
) -> None:
    load_dotenv()    
    try:
        global client
        client = InferenceHTTPClient(
            api_url="https://detect.roboflow.com",
            api_key=os.getenv('ROBOFLOW_API_KEY', ''),
        )
        input_path = Path(input_path)
        prompt_dir = Path(prompt_dir)
        output_dir = Path(output_dir)
        
        logger.info("Starting image processing pipeline")
        
        # 1. Load prompts
        prompts = load_prompts(prompt_dir)
        logger.info(f"Loaded {len(prompts)} prompts")
        
        # 2. Get image paths
        if is_video:
            temp_frames_dir = output_dir / "extracted_frames"
            image_paths = extract_frames(input_path, temp_frames_dir, sample_rate)
            # TODO: Define "extract_frames" inside utility module
        else:
            image_paths = get_image_paths(input_path)
            
        # 3. Process images with Roboflow  
        
            # 3.1 Detect images with prompts, using Florence-2 or YOLOvX
        detection_results: List[Image] = detect_images(image_paths)
        
        # 4. Select frames
        selected_frames = frame_selection(detection_results)
        
            # 3.2 Segment the images, using SAM-2. Postponed to save resources
        segmentation_results = segment_images(selected_frames)
        
        # 5. Generate frame descriptions
        descriptions = frame_description(selected_frames, segmentation_results)
        
        # 6. Save results
        save_results(selected_frames, descriptions, segmentation_results, output_dir)
        
        # 7. Reconstruction with WaterSplatting technique
        # TODO: define input arguments for "reconstruct_image"
        # TODO: implement "reconstruct_image" with os.system calls
        
        logger.info("Pipeline completed.")
        
    except PipelineError as e:
        logger.error(f"Pipeline error: {str(e)}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()