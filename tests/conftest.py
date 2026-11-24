import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SAMPLE = {
    "alcohol": 14.23, "malic_acid": 1.71, "ash": 2.43, "alcalinity_of_ash": 15.6,
    "magnesium": 127, "total_phenols": 2.8, "flavanoids": 3.06,
    "nonflavanoid_phenols": 0.28, "proanthocyanins": 2.29, "color_intensity": 5.64,
    "hue": 1.04, "od280_od315_of_diluted_wines": 3.92, "proline": 1065,
}


def load_service(name):
    """Import a service's main/utils modules.

    Every service has modules called main.py and utils.py, so drop any
    previously imported ones before importing this service's versions.
    """
    for mod in ("main", "utils"):
        sys.modules.pop(mod, None)
    sys.path.insert(0, str(ROOT / name))
    try:
        utils = importlib.import_module("utils")
        main = importlib.import_module("main")
    finally:
        sys.path.remove(str(ROOT / name))
    return main, utils


@pytest.fixture
def sample():
    return dict(SAMPLE)


@pytest.fixture
def labelled(sample):
    return {**sample, "wine_class": "Cultivar 1"}
