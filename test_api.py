
from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_classes():
    response = client.get("/api/classes")
    assert response.status_code == 200
    assert response.json()["total_classes"] >= 10


def test_model_info():
    response = client.get("/api/model/info")
    assert response.status_code == 200
    assert response.json()["architecture"] == "ResNet50"


def test_unsupported_file():
    response = client.post(
        "/api/predict",
        files={"file": ("note.txt", b"hello", "text/plain")}
    )
    assert response.status_code == 415


def test_corrupted_image():
    response = client.post(
        "/api/predict",
        files={"file": ("broken.png", b"invalid image", "image/png")}
    )
    assert response.status_code == 400


def test_valid_image():
    buffer = BytesIO()
    Image.new("RGB", (100, 100), "red").save(buffer, format="PNG")

    response = client.post(
        "/api/predict",
        files={"file": ("red.png", buffer.getvalue(), "image/png")}
    )

    assert response.status_code == 200

    data = response.json()
    assert "predicted_class" in data
    assert 0 <= data["confidence"] <= 100
    assert len(data["top5"]) == 5
