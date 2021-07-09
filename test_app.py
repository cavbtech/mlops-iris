from fastapi.testclient import TestClient
from main import app


def test_ping():
    with TestClient(app) as client:
        response = client.get("/ping")
        assert response.status_code == 200
        assert response.json() == {"ping":"pong"}

def test_pred_virginica():
    payload = {
      "sepal_length": 3,
      "sepal_width": 5,
      "petal_length": 3.2,
      "petal_width": 4.4
    }
    with TestClient(app) as client:
        response = client.post('/predict_flower', json=payload)
        flower_result = response.json()['flower_class']
        assert response.status_code == 200
        print(f"response.json()={response.json()}")
        assert flower_result == "Iris Virginica"