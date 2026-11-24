import json

import requests
from fastapi.testclient import TestClient

from conftest import load_service

main, utils = load_service("trainr")


class FakeResponse:
    def raise_for_status(self):
        pass


def use_model_dir(monkeypatch, path):
    monkeypatch.setattr(utils, "MODEL_DIR", str(path))
    monkeypatch.setattr(utils, "MODEL_PATH", str(path / "wine_lr.pkl"))
    monkeypatch.setattr(utils, "FEEDBACK_PATH", str(path / "feedback.json"))


def test_startup_trains_baseline(tmp_path, monkeypatch):
    use_model_dir(monkeypatch, tmp_path)
    with TestClient(main.app) as client:
        assert client.get("/ping").status_code == 200
    assert (tmp_path / "wine_lr.pkl").exists()


def test_train_keeps_feedback_and_reloads(tmp_path, monkeypatch, labelled):
    use_model_dir(monkeypatch, tmp_path)
    calls = []
    monkeypatch.setattr(main.requests, "post", lambda url, **kw: calls.append(url) or FakeResponse())
    with TestClient(main.app) as client:
        for _ in range(2):
            response = client.post("/train", json=[labelled])
            assert response.status_code == 200
            assert response.json()["cv_accuracy"] > 0.9
    # feedback accumulates instead of replacing earlier data
    assert len(json.loads((tmp_path / "feedback.json").read_text())) == 2
    assert calls == [f"{main.PREDICTR_ENDPOINT}/reload_model"] * 2


def test_train_reports_predictr_failure(tmp_path, monkeypatch, labelled):
    use_model_dir(monkeypatch, tmp_path)

    def fail(url, **kw):
        raise requests.ConnectionError("down")

    monkeypatch.setattr(main.requests, "post", fail)
    with TestClient(main.app) as client:
        assert client.post("/train", json=[labelled]).status_code == 502


def test_rejects_unknown_class(labelled):
    labelled["wine_class"] = "Red wine"
    with TestClient(main.app) as client:
        assert client.post("/train", json=[labelled]).status_code == 422
