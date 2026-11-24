import os
from contextlib import asynccontextmanager

import requests
import uvicorn
from fastapi import FastAPI, HTTPException

from common.wine import WineSample
from utils import init_model, train_model

PREDICTR_ENDPOINT = os.getenv("PREDICTR_ENDPOINT", "http://localhost:9999")


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_model()
    yield


app = FastAPI(title="trainr", docs_url="/", lifespan=lifespan)


@app.get("/ping")
def ping():
    """Health check."""
    return {"ping": "pong"}


@app.post("/train", status_code=200)
def train(data: list[WineSample]):
    """Retrain the model with new labelled samples, then tell predictr to
    load the new version."""
    accuracy = train_model(data)
    try:
        response = requests.post(f"{PREDICTR_ENDPOINT}/reload_model", timeout=30)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"predictr reload failed: {exc}")
    return {"detail": "Training successful", "cv_accuracy": round(accuracy, 3)}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=7777)
