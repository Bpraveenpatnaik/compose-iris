import os

import requests
import uvicorn
from fastapi import FastAPI, HTTPException

from common.wine import WineSample
from utils import process_data

TRAINR_ENDPOINT = os.getenv("TRAINR_ENDPOINT", "http://localhost:7777")

app = FastAPI(title="processr", docs_url="/")


@app.get("/ping")
def ping():
    """Health check."""
    return {"ping": "pong"}


@app.post("/process", status_code=200)
def process(data: list[WineSample]):
    """Clean labelled samples and send them to trainr for training."""
    processed = process_data(data)
    try:
        response = requests.post(f"{TRAINR_ENDPOINT}/train", json=processed, timeout=30)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"trainr request failed: {exc}")
    return {"detail": "Processing successful", "samples_sent": len(processed)}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8888)
