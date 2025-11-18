import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate

from dataclasses import dataclass


@dataclass
class RegressionResults:
    score_mean: float
    score_std: float


def evaluate_classical() -> RegressionResults:
    df = pd.read_csv("data/housing.csv")
    model = LinearRegression()
    cv_results = cross_validate(model, df[["MedInc"]], df["MedHouseVal"])
