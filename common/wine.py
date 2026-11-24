"""Schemas and label mappings shared by all three services.

The UCI Wine dataset has 13 chemical measurements for wines grown in the same
region of Italy by three different cultivars (grape growers). The target is
the cultivar, not the type of wine.
"""
from typing import Literal

from pydantic import BaseModel, Field

# Order matches sklearn.datasets.load_wine().feature_names
FEATURES = [
    "alcohol",
    "malic_acid",
    "ash",
    "alcalinity_of_ash",
    "magnesium",
    "total_phenols",
    "flavanoids",
    "nonflavanoid_phenols",
    "proanthocyanins",
    "color_intensity",
    "hue",
    "od280_od315_of_diluted_wines",
    "proline",
]

CLASSES = {0: "Cultivar 1", 1: "Cultivar 2", 2: "Cultivar 3"}
CLASS_IDS = {name: idx for idx, name in CLASSES.items()}

WineClass = Literal["Cultivar 1", "Cultivar 2", "Cultivar 3"]


class WineFeatures(BaseModel):
    """The 13 measurements for one wine sample. All are non-negative."""

    alcohol: float = Field(ge=0)
    malic_acid: float = Field(ge=0)
    ash: float = Field(ge=0)
    alcalinity_of_ash: float = Field(ge=0)
    magnesium: float = Field(ge=0)
    total_phenols: float = Field(ge=0)
    flavanoids: float = Field(ge=0)
    nonflavanoid_phenols: float = Field(ge=0)
    proanthocyanins: float = Field(ge=0)
    color_intensity: float = Field(ge=0)
    hue: float = Field(ge=0)
    od280_od315_of_diluted_wines: float = Field(ge=0)
    proline: float = Field(ge=0)

    def to_row(self) -> list[float]:
        """Return the features as a list in the model's column order."""
        return [getattr(self, name) for name in FEATURES]


class WineSample(WineFeatures):
    """A labelled sample, used for training and feedback."""

    wine_class: WineClass
