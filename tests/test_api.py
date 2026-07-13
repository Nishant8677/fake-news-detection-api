from fastapi.testclient import TestClient
from inference.app import app

client = TestClient(app)

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
