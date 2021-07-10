## Create a new test program to test various test cases
from fastapi.testclient import TestClient
from main import app
import json

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
        print(f"response.json()={response.json()}")
        flower_result = response.json()['flower_class']
        assert response.status_code == 200
        assert flower_result == "Iris Virginica"

def test_pred_versiocolor():
    payload = {
      "sepal_length": 0.1,
      "sepal_width": 0.2,
      "petal_length": 0.2,
      "petal_width": 0.4
    }
    with TestClient(app) as client:
        response = client.post('/predict_flower', json=payload)
        print(f"response.json()={response.json()}")
        flower_result = response.json()['flower_class']
        assert response.status_code == 200
        assert flower_result == 'Iris Versicolour'


def test_pred_iris_setosa():
    payload = {
      "sepal_length": 5.1,
      "sepal_width": 3.5,
      "petal_length": 1.4,
      "petal_width": 0.2
    }
    with TestClient(app) as client:
        response = client.post('/predict_flower', json=payload)
        print(f"response.json()={response.json()}")
        flower_result = response.json()['flower_class']
        assert response.status_code == 200
        assert flower_result == 'Iris Setosa'


