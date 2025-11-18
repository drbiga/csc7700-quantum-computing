import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split

from config import DATA_DIR


def download_all_datasets():
    df = pd.concat(
        datasets.fetch_california_housing(
            data_home=DATA_DIR, return_X_y=True, as_frame=True
        ),
        axis=1,
    )
    df.to_csv(f"{DATA_DIR}/housing.csv", index=False)

    df = pd.concat(datasets.load_breast_cancer(return_X_y=True, as_frame=True), axis=1)
    df.to_csv(f"{DATA_DIR}/cancer.csv", index=False)

    df = pd.concat(datasets.load_digits(return_X_y=True, as_frame=True), axis=1)
    df.to_csv(f"{DATA_DIR}/mnist.csv", index=False)


def load_housing_train_test() -> (
    tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
):
    """Loads the housing dataset already split into training and testing sets"""
    df = pd.read_csv("data/housing.csv")
    X_train, X_test, y_train, y_test = train_test_split(
        df[["MedInc"]], df["MedHouseVal"], test_size=0.2
    )
    return X_train, X_test, y_train, y_test
