from typing import List
from pathlib import Path
from image import Image 
from artifact import Artifact
import os
import PIL

def detect_and_segmentation_workflow(images: List[Path]) -> List[Image]:
    global client
    result = client.run_workflow(
        workspace_name=os.getenv('ROBOFLOW_WORKSPACE_NAME',''),
        workflow_id=os.getenv('ROBOFLOW_DETECTION_WORKFLOW_ID',''),
        images=images,
        parameters={}
    )
    

    return list(map(lambda image_path, res: Image(path=image_path, image=PIL.open(image_path), object_detection=res), images, result))


def frame_selection(images: List[Image], prompt: str) -> List[Artifact]:
    """
        TODO: Define output type.
    """
    global client
    
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

