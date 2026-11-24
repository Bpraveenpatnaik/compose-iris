from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import utils
from common.wine import WineClass, WineFeatures


@asynccontextmanager
async def lifespan(app: FastAPI):
    utils.load_model()
    yield


app = FastAPI(title="predictr", docs_url="/", lifespan=lifespan)


class QueryOut(BaseModel):
    wine_class: WineClass


@app.get("/ping")
def ping():
    """Health check."""
    return {"ping": "pong"}


@app.post("/predict_wine", response_model=QueryOut, status_code=200)
def predict_wine(query: WineFeatures):
    """Predict which cultivar a wine sample came from."""
    if utils.clf is None and not utils.load_model():
        raise HTTPException(status_code=503, detail="No trained model available yet")
    return {"wine_class": utils.predict(query)}


@app.post("/reload_model", status_code=200)
def reload_model():
    """Load the latest model saved by trainr."""
    if not utils.load_model():
        raise HTTPException(status_code=503, detail="No trained model available yet")
    return {"detail": "Model successfully loaded"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=9999)
