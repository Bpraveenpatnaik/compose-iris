import os
import pickle

from common.wine import CLASSES, WineFeatures

MODEL_DIR = os.getenv("MODEL_DIR", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "wine_lr.pkl")

clf = None


def load_model() -> bool:
    """Load the latest model from the shared volume. Returns False if trainr
    hasn't saved one yet."""
    global clf
    if not os.path.isfile(MODEL_PATH):
        return False
    with open(MODEL_PATH, "rb") as f:
        clf = pickle.load(f)
    return True


def predict(query: WineFeatures) -> str:
    prediction = clf.predict([query.to_row()])[0]
    return CLASSES[int(prediction)]
