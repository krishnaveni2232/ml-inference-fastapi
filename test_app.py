from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200


def test_predict():
    response = client.post(
        "/predict",
        json={"text": "This product is excellent"}
    )

    assert response.status_code == 200

    data = response.json()

    assert "text" in data
    assert "prediction" in data
    assert "probabilities" in data

    assert isinstance(data["prediction"], str)
    assert isinstance(data["probabilities"], dict)


def test_predict_empty_text():
    response = client.post(
        "/predict",
        json={"text": ""}
    )

    assert response.status_code == 200
