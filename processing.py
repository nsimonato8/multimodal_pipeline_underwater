from typing import List
from image import Image 
from artifact import Artifact
import os
from pathlib import Path
import subprocess


def water_splatting(water_splatting_repo: Path, dataset_folder: Path) -> None:

    config_path = water_splatting_repo / Path(
        "outputs/unnamed/water-splatting"
    )  # /your_timestamp/config.yml
    config_path = (
        config_path
        / max(os.path.listdir(config_path), key=os.path.getmtime)
        / "config.yml"
    )

    subprocess.run(
        f"cd {water_splatting_repo} && \
                     source activate water_splatting && \
                     ns-train water-splatting --vis viewer colmap sparse --colmap-path sparse/0 --data {dataset_folder} --images-path images && \
                     ns-render dataset --load-config {config_path} --data {water_splatting_repo}/images \
                     source deactivate",
        shell=True,
    )
    pass


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
