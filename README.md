# Multimodal Pipeline Underwater
[![Paper](https://img.shields.io/badge/CS-Paper-b31b1b?logo=arxiv&logoColor=red)](https://arxiv.org/abs/2406.03207) <br>
[![Paper](https://img.shields.io/badge/3DGS-Underwater-blue)]()[![Paper](https://img.shields.io/badge/version-0.1-yellow)]() <br>

 [Niccoló Simonato](), [Corradetti Daniele](https://ualg.academia.edu/DanieleCorradetti) & [José António Bettencourt]()  <br><br>
*Università degli studi di Udine, Dipartimento di Scienze Matematiche, Informatiche e Fisiche (DMIF)* <br>
*Elementar s.r.l., Divisione Ricerca e Sviluppo, Galleria Enzo Tortora 21, 10121 Torino, Italy*<br>
*STAP Reabilitação Estrutural, SA Rua General Ferreira Martins 8 - 9B,  Algés, 1495-137, Portugal*<br>
*Grupo de Fisica Matematica, Instituto Superior Tecnico, Av. Rovisco Pais, Lisboa, 1049-001, Portugal* <br>



## Overview

In this paper we present a comprehensive pipeline for the three-dimensional detection and reconstruction of archaeological artifacts in underwater environments.
Our approach takes advantage of Large Multimodal Models (LMMs), which allow the integration of historical, geographic, and contextual data into the acquired images, thus aiding in the identification and interpretation of objects of possible archaeological interest.
In addition to multi-modal integration, our pipeline also supports the use of a 3D visualization method (Gaussian Splatting) that has never before been applied to underwater archaeology, but which in many ways is a natural candidate to replace the technical problems of underwater photogrammetry. To demonstrate the effectiveness of the method, we concretized a version of this pipeline and applied it to the investigation of the wreck of the steamship SS Main (1892), located in Porto Pim Bay, Faial Island, Azores.

## Installation

### Installing WaterSplatting

Install the WaterSplatting implementation from [the original repository](https://github.com/water-splatting/water-splatting).

### Installing the pipeline dependencies

After cloning the repository, run `pip install .`.

## Use

### Preparation

The execution of the pipeline requires three `.txt` files, containing the prompts used by the LLMs involved in the pipeline:

* `Historical.txt`, which contains information about the historical context of the archeological site.
* `GeographicalEnvironment.txt`, which contains information about the geographical environment of the archeological site.
* `Artifacts.txt`, which contains directives for the models about the artifacts to look for.

### Environment variables

The pipeline uses two Roboflow's Workflows for the tasks of Prompted Object Detection, Image Segmentation and Visual Captioning. For this application to work, you'll need Roboflow's API key (you can get it even with a free account) and the IDs of the workflows.    
The easiest way to do this is to define a `.env` file like the following:
```
ROBOFLOW_API_URL='' # If you leave this empty, it will be assumed that this is a local deployment.  
ROBOFLOW_API_KEY='YOUR_API_KEY'
ROBOFLOW_WORKSPACE_NAME='tesimastertest' 
ROBOFLOW_DETECTION_WORKFLOW_ID='underwateranalysis'
ROBOFLOW_DESCRIPTION_WORKFLOW_ID='underwater-captioning'
``` 

It is advised to deploy the pipeline on a local environment, as the tests were performed is similar circumstances.   
Before starting the deployment, run:
```
pip install inference && inference server start
``` 

### Execution

If the processing is performed on a sequence of images, run:

```
pipeline --input_path /path/to/the/image/folder \
--prompt_dir /path/to/the/prompts/folder \
--output_dir /path/to/the/outputs/folder \
```

If the processing is performed on a video, run:


```
pipeline --input_path /path/to/the/video/file \
--prompt_dir /path/to/the/prompts/folder \
--output_dir /path/to/the/outputs/folder \
--is-video \
--sample-rate sample_rate
```


