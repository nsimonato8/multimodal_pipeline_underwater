from typing import List
from image import Image
from artifact import Artifact
import os
from itertools import groupby
import supervision as sv

def get_largest_bbox_label(predictions: Dict[str, Any]) -> Optional[str]:
    try:
        bboxes = np.array(predictions["bboxes"])
        labels = predictions["labels"]
       
        if len(bboxes) == 0:
            return None
       
        areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
        return labels[np.argmax(areas)]
       
    except (KeyError, IndexError):
        return None


def detect_and_segmentation_workflow(images: List[Image], prompt: str) -> List[Image]:
    global client
    results = client.run_workflow(
        workspace_name=os.getenv("ROBOFLOW_WORKSPACE_NAME", ""),
        workflow_id=os.getenv("ROBOFLOW_DETECTION_WORKFLOW_ID", ""),
        images=list(map(lambda x: x.image, images)),
        parameters={"detection_prompt": prompt}
        )
    
    return list(map(lambda res, img: Image.from_workflow_result(res, img), results, images))


def frame_selection(images: List[Image]) -> List[Artifact]:
    # Step 1: Select the bounding box with maximum area and return its predicted label
    for image in images:
        if not hasattr(image, "object_detection") or not image.object_detection:
            continue
        image.artifact_label = get_largest_bbox_label(image.object_detection)

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


def generate_frame_description(artifact: Artifact, prompt: str) -> Artifact:
    # This function adds the caption to the Artifact object.
    # TODO: define input and complete implementation.
    global client

    result = client.run_workflow(
        workspace_name=os.getenv("ROBOFLOW_WORKSPACE_NAME", ""),
        workflow_id=os.getenv("ROBOFLOW_DESCRIPTION_WORKFLOW_ID", ""),
        images=artifact.images,
        parameters={"prompt": prompt},
    )

    return result
