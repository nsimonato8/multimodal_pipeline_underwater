from typing import List
from image import Image 
from artifact import Artifact
import os

def detect_and_segmentation_workflow(images: List[Image]) -> List[Image]:
    global client
    result = client.run_workflow(
        workspace_name=os.getenv('ROBOFLOW_WORKSPACE_NAME',''),
        workflow_id=os.getenv('ROBOFLOW_DETECTION_WORKFLOW_ID',''),
        images=list(map(lambda x: x.image, images)),
        parameters={}
    )
    

    return result # TODO: Add parsing logic for the workflow result


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

