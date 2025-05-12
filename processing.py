from typing import List, Dict, Any, Optional
from image import Image
from artifact import Artifact
import os
from itertools import groupby
import tqdm
import base64
import numpy as np
from openai import OpenAI
from openai.types.responses.response import Response
import cv2


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


def detect_and_segmentation_workflow(
    client, images: List[Image], prompt: str
) -> List[Image]:

    results = []
    for image in tqdm.tqdm(images):
        results += client.run_workflow(
            workspace_name=os.getenv("ROBOFLOW_WORKSPACE_NAME", ""),
            workflow_id=os.getenv("ROBOFLOW_DETECTION_WORKFLOW_ID", ""),
            images={"input_image": image.image},
            parameters={"detection_prompt": prompt},
        )

    return list(
        map(
            lambda res, img: Image.from_workflow_result(res, img.image, img.original),
            results,
            images,
        )
    )


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
    return [
        Artifact(name=label, images=list(group))
        for label, group in grouped_images
        if label is not None
    ]


def generate_frame_description(
    client, artifacts: List[Artifact], prompt: str
) -> List[Artifact]:
    # This function adds the caption to the Artifact object.
    def parse_caption(result: Response) -> str:
        print(result)
        return response.output[0].content[0].text

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    result = []
    for artifact in tqdm.tqdm(artifacts):
        print(
            type(artifact.best_image.image)
        )  # TODO: Implement a logic for the selection of the best image among the pictures of the artifact.

        image = artifact.best_image.image

        _, buffer = cv2.imencode(".jpg", image)

        base64_image = base64.b64encode(buffer).decode("utf-8")

        response = client.responses.create(
            model="gpt-4.1-nano",
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {
                            "type": "input_image",
                            "image_url": f"data:image/jpg;base64,{base64_image}",
                        },
                    ],
                }
            ],
        )

        result.append(parse_caption(response))

    return result
