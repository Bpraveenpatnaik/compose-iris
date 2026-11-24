import requests
from fastapi.testclient import TestClient

from conftest import load_service

main, utils = load_service("processr")


class FakeResponse:
    def raise_for_status(self):
        pass


def test_ping():
    with TestClient(main.app) as client:
        assert client.get("/ping").json() == {"ping": "pong"}


def test_process_dedupes_and_forwards(monkeypatch, labelled):
    sent = {}

    def fake_post(url, json, timeout):
        sent["url"], sent["json"] = url, json
        return FakeResponse()

    monkeypatch.setattr(main.requests, "post", fake_post)
    with TestClient(main.app) as client:
        response = client.post("/process", json=[labelled, labelled])
    assert response.status_code == 200
    assert response.json()["samples_sent"] == 1
    assert sent["url"].endswith("/train")
    assert sent["json"] == [labelled]


def test_process_reports_trainr_failure(monkeypatch, labelled):
    def fail(*args, **kwargs):
        raise requests.ConnectionError("down")

    monkeypatch.setattr(main.requests, "post", fail)
    with TestClient(main.app) as client:
        assert client.post("/process", json=[labelled]).status_code == 502
