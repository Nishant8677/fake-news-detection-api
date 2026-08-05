from fastapi.testclient import TestClient

from inference.app import app

client = TestClient(app)

def test_health_endpoint():
    # The container HEALTHCHECK polls this route, so it must exist and must
    # report 503 rather than 200 when the weights were not mounted.
    response = client.get("/health")
    assert response.status_code in (200, 503)

    data = response.json()
    assert "model_loaded" in data
    assert "dataset_version" in data

    if data["model_loaded"]:
        assert response.status_code == 200
        assert data["status"] == "ok"
    else:
        assert response.status_code == 503
        assert data["status"] == "degraded"


def test_predict_endpoint():
    response = client.post("/predict", json={"text": "This is a test news statement."})
    assert response.status_code == 200
    
    data = response.json()
    if "error" in data:
        assert data["error"] == "Model is not loaded. Please train the model first."
    else:
        assert "prediction" in data
        assert "confidence" in data
        assert "dataset_version" in data
