import time
import requests

URL = "http://127.0.0.1:7860"

def test_health():
    r = requests.get(f"{URL}/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is True

def test_batch_normal():
    r = requests.post(f"{URL}/predict_batch", json={"comments": ["I love this", "this is bad", "ok"]})
    assert r.status_code == 200
    assert len(r.json()["predictions"]) == 3

def test_batch_vide():
    r = requests.post(f"{URL}/predict_batch", json={"comments": []})
    assert r.status_code in (400, 422)

def test_donnees_invalides():
    r = requests.post(f"{URL}/predict_batch", json={"wrong": "data"})
    assert r.status_code == 422

def test_cas_limites():
    comments = ["ok", "😀😀😀", "x" * 5000, "j'adore this video très bien"]
    r = requests.post(f"{URL}/predict_batch", json={"comments": comments})
    assert r.status_code == 200
    assert len(r.json()["predictions"]) == 4

def test_temps_50_commentaires():
    comments = ["I really like this video"] * 50
    start = time.time()
    r = requests.post(f"{URL}/predict_batch", json={"comments": comments})
    ms = (time.time() - start) * 1000
    print(f"Temps pour 50 commentaires : {ms:.0f} ms")
    assert r.status_code == 200
    assert ms < 1000