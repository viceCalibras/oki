# Cutting force prediction
This repository contains the machine learning framework for prediction of cutting forces in CNC machining out of the process parameters. 

The repository is structured as follows:

- Data Mining
    - Consists of `data_mining_deform2d.ipynb` and `data_mining_roughness.ipynb` notebooks.
    - Capable of loading raw simulation and experimental datasets, visualizing the data, performing feature engineering and finally, exporting the datasets ready for training.
    - Data mining functionalities are split for the two use cases: simulation and experimental (roughness). 
    - Preparation of simulation dataset consists of parsing .MSG files containing temperature data and merging them with tabular data.
    - Data is not part of the repository and should be provided independantly.
- Model and dataset
    - Consists of `dataset.py`(Pytorch dataset class definition) and `model.py` (contains 2 regression models).
    - Regression models architectures differ for simulation and experimental (roughness) use cases, and had been experimentally defined.
- Training and evaluation
    - Consists of `train_and_save.py` (performs dataset, model and parameter loading and definition, as well as the sole training functionality) and two evaluation notebooks: `evaluate_deform2d.ipynb` and `evaluate_roughness.ipynb`.
    - Model training had been parameterized, and it includes different functionalities, all described in the report.
    - Evaluation setup loads the testing dataset and the model and perfoms inference on the whole test data. The inference results are then used to compute evaluation metrics. Evaluation notebooks also include reverse transformation functionalities.

To use the repository, first ensure that a working version of Python3 is installed on the system (Python 3.10 reccomended). After that, do the basic setup, as so:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Next, crate two folder inside the repository `data` and `trained_models`. Inside, either copy-paste already prepared datasets or raw input, and trained models (if available).

Next, run the Data Mining notebooks and explore the data & dataset creation:

Notebook setup

```bash
python3 -m ipykernel install --user --name=cutting_force
jupyter notebook
```

To re-train the models (or train new ones), edit the paramters inside the `train_and_save.py` and run the training loop: `python3 train_and_save.py`. Utilize evaluation notebooks to visualize results.