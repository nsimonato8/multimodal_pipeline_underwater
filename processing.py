from typing import List, Dict, Union, Tuple, Optional
from pathlib import Path
from image import Image 
from artifact import Artifact
import os
import PIL

def detect_images(images: List[Path]) -> List[Image]:
    global client
    result = client.run_workflow(
        workspace_name=os.getenv('ROBOFLOW_WORKSPACE_NAME',''),
        workflow_id=os.getenv('ROBOFLOW_DETECTION_WORKFLOW_ID',''),
        images=images,
        parameters={}
    )
	return list(map(lambda image_path, res: Image(path=image_path, image=PIL.open(image_path), object_detection=res), images, result))

def segment_images(artifacts: List[Artifact]) -> List[Artifacts]:
    """
    Maybe we can remove it.    
    """
    global client
    result = client.run_workflow(
        workspace_name=os.getenv('ROBOFLOW_WORKSPACE_NAME',''),
        workflow_id=os.getenv('ROBOFLOW_SEGMENTATION_WORKFLOW_ID',''),
        images=images,
        parameters={}
    )
	pass

def frame_selection(images: List[Image], prompt: str) -> List[Artifact]:
    """
        TODO: Define output type.
    """
    global client
    result = client.run_workflow(
        workspace_name=os.getenv('ROBOFLOW_WORKSPACE_NAME',''),
        workflow_id=os.getenv('ROBOFLOW_SELECTION_WORKFLOW_ID',''),
        images=images,
        parameters={
            prompt: prompt
        }
    )
	pass

def frame_description(artifact: Artifact, prompt: str) -> Artifact:
    """
        TODO: define input and complete implementation.
    """
    global client
    result = client.run_workflow(
        workspace_name=os.getenv('ROBOFLOW_WORKSPACE_NAME',''),
        workflow_id=os.getenv('ROBOFLOW_DESCRIPTION_WORKFLOW_ID',''),
        images=images,
        parameters={
            prompt: prompt
        }
    )
	pass

