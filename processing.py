from typing import List
from image import Image 
from artifact import Artifact
import os
from itertools import groupby

def detect_and_segmentation_workflow(images: List[Image], prompt: str) -> List[Image]:
    global client
    result = client.run_workflow(
        workspace_name=os.getenv('ROBOFLOW_WORKSPACE_NAME',''),
        workflow_id=os.getenv('ROBOFLOW_DETECTION_WORKFLOW_ID',''),
        images=list(map(lambda x: x.image, images)),
        parameters={
            'prompt': prompt # TODO: check if this is correct
        }
    )
    # TODO: Add parsing logic for the workflow result
    return result 


def frame_selection(images: List[Image]) -> List[Artifact]:
    # Step 1: Select the bounding box with maximum area and return its predicted label
    for image in images:
        if not hasattr(image, 'object_detection') or not image.object_detection:
            continue
        max_bbox = max(
            image.object_detection,
            key=lambda bbox: (bbox['xmax'] - bbox['xmin']) * (bbox['ymax'] - bbox['ymin']),
            default=''
        )
        image.artifact_label = max_bbox['label']

    # Step 2: Group the images by the labels selected in the previous step
    grouped_images = groupby(
        images, key=lambda img: getattr(img, "artifact_label", None)
    )

    # Step 3: Group together the images with the same label and return a list of artifacts
    artifacts = []
    for label, group in grouped_images:
        if label is not None:
            artifact = Artifact(label=label, images=list(group))
            artifacts.append(artifact)

    return artifacts


def frame_description(artifact: Artifact, prompt: str) -> Artifact:
    """
        TODO: define input and complete implementation.
    """
    global client

    result = client.run_workflow(
        workspace_name=os.getenv('ROBOFLOW_WORKSPACE_NAME',''),
        workflow_id=os.getenv('ROBOFLOW_DESCRIPTION_WORKFLOW_ID',''),
        images=artifact.images,
        parameters={
            'prompt': prompt
        }
    )

    return result
