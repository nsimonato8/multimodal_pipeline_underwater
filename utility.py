from typing import List, Dict
from pathlib import Path
from image import check_image_integrity
import cv2
import logging
import os
import json

from errors import InputError, ProcessingError

def load_prompts(prompt_folder: Path) -> Dict[str, str]:
    """
    Load prompts from .txt files in the given folder.
    
    Args:
        prompt_folder: Path to folder containing prompt files
        
    Returns:
        Dictionary mapping prompt names to their content
        
    Raises:
        InputError: If prompt files cannot be found or read
    """
    prompt_files = list(prompt_folder.glob("*.txt"))
    
    if len(prompt_files) < 3:
        raise InputError(f"Expected 3 prompt files in {prompt_folder}, found {len(prompt_files)}")
    
    prompts = {}
    try:
        for file in prompt_files[:3]:  # Take first 3 prompt files
            with open(file, 'r', encoding='utf-8') as f:
                prompts[file.stem] = f.read().strip()
        
        return prompts
    except Exception as e:
        raise InputError(f"Error loading prompt files: {str(e)}")



def get_image_paths(input_path: Path) -> List[Path]:
    """
    Get image paths from a directory.
    
    Args:
        input_path: Path to directory containing images
        
    Returns:
        List of paths to images
        
    Raises:
        InputError: If directory contains no images
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_paths = [
        p for p in input_path.iterdir() 
        if p.is_file() and p.suffix.lower() in image_extensions and check_image_integrity(p)
    ]
    
    if not image_paths:
        raise InputError(f"No images found in {input_path}")
    
    # Sort images by name to maintain sequence
    image_paths.sort()
    
    logging.info(f"Found {len(image_paths)} images in {input_path}")
    return image_paths


def save_results(
    selected_frames: List[Path],
    descriptions: Dict[str, str],
    segmentation_results: Dict[str, Dict],
    output_dir: Path
) -> None:
    """
    Save processing results to output directory.
    
    Args:
        selected_frames: List of paths to selected frames
        descriptions: Dictionary mapping frame paths to their descriptions
        segmentation_results: Dictionary mapping image paths to their segmentation results
        output_dir: Directory to save results
        
    Raises:
        ProcessingError: If results cannot be saved
    """
    try:
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Save descriptions
        descriptions_path = output_dir / "descriptions.json"
        with open(descriptions_path, 'w', encoding='utf-8') as f:
            json.dump({os.path.basename(k): v for k, v in descriptions.items()}, f, indent=2)
        
        # Save segmentation results
        results_path = output_dir / "segmentation_results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump({os.path.basename(k): v for k, v in segmentation_results.items()}, 
                     f, indent=2)
        
        # Copy selected frames to output directory
        frames_dir = output_dir / "selected_frames"
        frames_dir.mkdir(exist_ok=True)
        
        for frame in selected_frames:
            dest = frames_dir / frame.name
            # Use shutil.copy2 in a real implementation to preserve metadata
            # Here using cv2 to simplify dependencies
            img = cv2.imread(str(frame))
            cv2.imwrite(str(dest), img)
        
        logging.info(f"Results saved to {output_dir}")
        
    except Exception as e:
        raise ProcessingError(f"Error saving results: {str(e)}")