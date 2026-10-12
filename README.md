
# Image Classification API

An image classification web application using PyTorch, ResNet50,
FastAPI, HTML, CSS, and JavaScript.

## Features

- Pretrained ResNet50 model
- 1,000 ImageNet categories
- JPEG and PNG image uploads
- Top-1 prediction and top-5 results
- Confidence bars and image preview
- Recent prediction history in browser localStorage
- Image validation and upload size limits
- Model information and health endpoints
- Accuracy evaluation and confusion matrix generation

## Requirements

- Python 3.10 or compatible version
- Internet connection for the initial pretrained weight download
- Sufficient RAM and disk space

## Installation

Create and activate a virtual environment:

Windows:

    python -m venv venv
    venv\Scripts\activate

Install dependencies:

    pip install -r requirements.txt

Generate the class list:

    python generate_classes.py

## Run the application

    uvicorn app.main:app --reload

Open http://127.0.0.1:8000

API documentation:
http://127.0.0.1:8000/docs

## API endpoints

GET /api/health

GET /api/classes

GET /api/model/info

POST /api/predict

The prediction endpoint accepts a multipart/form-data file
field named "file".

## Evaluation

Prepare labeled test images in evaluation/test_data.
Folder names must match labels in app/classes.json.

Run:

    python -m evaluation.evaluate

The evaluation script reports top-1 and top-5 accuracy and
creates evaluation/confusion_matrix.png.

## Tests

    pytest -v

## Model details

Architecture: ResNet50
Pretrained weights: ImageNet-1K V2
Number of classes: 1000

The pretrained model is not fine-tuned on a custom dataset.
Report only accuracy measured on your actual test dataset.

## Limitations

Predictions depend on the ImageNet categories. The confidence
score is a model probability, not a guarantee of correctness.
Inference latency depends on hardware and image preprocessing.
