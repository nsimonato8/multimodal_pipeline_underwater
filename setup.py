from setuptools import setup

setup(
    name="multimodal_pipeline_underwater",
    version="0.0.1",
    install_requires=[
        "click",
        "python-dotenv",
        "inference_sdk",
        "opencv-python",
        "deprecated",
        "numpy",
        "tqdm",
        "supervision",
        "openai"
    ],
    entry_points={
        "console_scripts": [
            "pipeline = pipeline:main",
        ]
    },
)
