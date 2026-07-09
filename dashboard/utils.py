import requests

API_URL = "http://127.0.0.1:8000"


def predict(sk_id_curr: int):
    response = requests.post(
        f"{API_URL}/predict",
        json={"sk_id_curr": sk_id_curr},
    )
    return response.json()


def predict_batch(n_clients: int):
    response = requests.post(
        f"{API_URL}/predict_batch?n_clients={n_clients}",
    )
    return response.json()


def monitoring_stats():
    response = requests.get(
        f"{API_URL}/monitoring/stats",
    )
    return response.json()


def data_drift():
    response = requests.get(
        f"{API_URL}/monitoring/drift",
    )
    return response.json()