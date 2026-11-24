import json
import os
import pickle

from sklearn.datasets import load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common.wine import CLASS_IDS, WineSample

MODEL_DIR = os.getenv("MODEL_DIR", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "wine_lr.pkl")
FEEDBACK_PATH = os.path.join(MODEL_DIR, "feedback.json")


def build_model():
    # Scaling matters here: proline is in the hundreds, hue is around 1.
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))


def load_feedback() -> list[dict]:
    if not os.path.isfile(FEEDBACK_PATH):
        return []
    with open(FEEDBACK_PATH) as f:
        return json.load(f)


def fit_and_save(feedback: list[dict]) -> float:
    """Train on the base dataset plus all feedback, save the model, and
    return its 5-fold cross-validated accuracy."""
    X, y = load_wine(return_X_y=True)
    X, y = X.tolist(), y.tolist()
    for record in feedback:
        sample = WineSample(**record)
        X.append(sample.to_row())
        y.append(CLASS_IDS[sample.wine_class])

    model = build_model()
    accuracy = cross_val_score(build_model(), X, y, cv=5).mean()
    model.fit(X, y)

    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"Model trained on {len(y)} samples, CV accuracy {accuracy:.3f}")
    return accuracy


def init_model():
    """On startup, train a baseline model if none has been saved yet."""
    if not os.path.isfile(MODEL_PATH):
        fit_and_save(load_feedback())


def train_model(data: list[WineSample]) -> float:
    """Add new samples to the stored feedback and retrain.

    The original version refit the model on only the new batch, which threw
    away everything learned before (and fails on a single-class batch).
    """
    feedback = load_feedback() + [d.model_dump() for d in data]
    accuracy = fit_and_save(feedback)
    with open(FEEDBACK_PATH, "w") as f:
        json.dump(feedback, f)
    return accuracy
