import pandas as pd
from sklearn import datasets

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
