from fastapi.testclient import TestClient

from conftest import load_service

main, utils = load_service("predictr")
trainr_main, trainr_utils = load_service("trainr")


def use_model_dir(monkeypatch, path):
    monkeypatch.setattr(utils, "MODEL_PATH", str(path / "wine_lr.pkl"))
    monkeypatch.setattr(utils, "clf", None)
    monkeypatch.setattr(trainr_utils, "MODEL_DIR", str(path))
    monkeypatch.setattr(trainr_utils, "MODEL_PATH", str(path / "wine_lr.pkl"))
    monkeypatch.setattr(trainr_utils, "FEEDBACK_PATH", str(path / "feedback.json"))


def test_ping():
    with TestClient(main.app) as client:
        assert client.get("/ping").json() == {"ping": "pong"}


def test_predict_without_model_returns_503(tmp_path, monkeypatch, sample):
    use_model_dir(monkeypatch, tmp_path)
    with TestClient(main.app) as client:
        assert client.post("/predict_wine", json=sample).status_code == 503
        assert client.post("/reload_model").status_code == 503


def test_predict_each_cultivar(tmp_path, monkeypatch):
    from sklearn.datasets import load_wine

    use_model_dir(monkeypatch, tmp_path)
    trainr_utils.init_model()
    data = load_wine()
    with TestClient(main.app) as client:
        assert client.post("/reload_model").status_code == 200
        # first sample of each class in the dataset
        for target, label in [(0, "Cultivar 1"), (1, "Cultivar 2"), (2, "Cultivar 3")]:
            row = data.data[list(data.target).index(target)]
            payload = dict(zip(utils_features(), row.tolist()))
            response = client.post("/predict_wine", json=payload)
            assert response.status_code == 200
            assert response.json() == {"wine_class": label}


def test_rejects_negative_values(sample):
    sample["alcohol"] = -1
    with TestClient(main.app) as client:
        assert client.post("/predict_wine", json=sample).status_code == 422


def utils_features():
    from common.wine import FEATURES
    return FEATURES
