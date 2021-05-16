from fastapi.testclient import TestClient
import requests
import pandas as pd
import numpy as np

from app import app

client = TestClient(app)


def test_predict():
    data = pd.read_csv("heart.csv")
    request_features = list(data.columns)
    request_data = [
        x.item() if isinstance(x, np.generic) else x for x in data.iloc[0].tolist()
    ]
    response = requests.get(
        "http://127.0.0.1:8000/predict/",
        json={"data": [request_data], "features": request_features},
    )
    assert response.status_code == 200
    assert response.json()[0] == {"id": '0', "label": 1.0}, f"Responce was: {response.json()[0]}"
