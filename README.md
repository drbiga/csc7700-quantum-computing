# CSC 7700 - Quantum Computing

This repository contains the code to run a simple Quantum Machine Learning (QML) project and evaluate the models'
performance against corresponding classical baselines.

This project is using Python 3.13

In order to use this repo, clone it and install the necessary packages.

```cmd
python3.13 -m venv .venv
```

```bash
pip install -r requirements.txt
```

After installing the necessary packages, download the data with the function `download_all_datasets` that is in the [data](./src/data.py) module.

The [config](./src/config.py) module contains a few definitions of directory names and other configurations. Change them as needed.

## Machine Learning Tasks

We evaluated three different tasks: 1) regression, 2) binary classification, and 3) multi-class classification. Each task has its own file, and each of those files contain definitions for both the classical model and the quantum model.