# MLOps Zoomcamp

This repository contains all exercises, scripts, and resources for the [MLOps Zoomcamp](https://github.com/DataTalksClub/mlops-zoomcamp) course. You will find:

* **01\_Intro**: basic ML examples and introductory notebooks.
* **02\_Experiment\_Tracking**: end-to-end ML workflow with MLflow, including data preprocessing, training, hyperparameter tuning, model registry, and deployment exercises.

---

## Repository Structure

```
├── 01_Intro/
│   └── Data/                       # (optional) sample data and notebooks for introduction
├── 02_Experiment_Tracking/
│   ├── Data/                       # raw January–March 2023 taxi parquet files
│   ├── artifacts/                  # (optional) artifacts store root for MLflow server
│   ├── mlruns/                     # local MLflow file store (ignored)
│   ├── mlflow.db                  # SQLite backend store for MLflow (ignored)
│   └── homework/
│       ├── preprocess_data.py      # Q2: download & preprocess data
│       ├── train.py                # Q3: train with autolog
│       ├── hpo.py                  # Q5: hyperparameter tuning with Hyperopt
│       └── register_model.py       # Q6: select & register best model
├── .gitignore                      # ignore raw data, mlruns, mlflow.db
├── README.md                       # this file
└── requirements.txt                # Python dependencies
```

---

## Setup

1. **Clone the repo**

   ```bash
   git clone https://github.com/MyatKaung/mlops_zoomcamp.git
   cd mlops_zoomcamp
   ```

2. **Create a Python environment**

   ```bash
   conda create -n mlopszoomcamp python=3.10 -y
   conda activate mlopszoomcamp
   pip install -r requirements.txt
   ```

3. **Download raw data**

   * Grab the Green Taxi Trip Records (Parquet) for January, February, March 2023 from the official source.
   * Place the files under:

     * `01_Intro/Data/`
     * `02_Experiment_Tracking/Data/`

---

## Exercises

### Q2: Download & preprocess data

```bash
cd 02_Experiment_Tracking/homework
python preprocess_data.py --raw_data_path ../Data --dest_path ./output
```

* Outputs: `train.pkl`, `val.pkl`, and `dv.pkl` in `./output`.

### Q3: Train a model with autolog

```bash
python train.py --data_path ./output
```

* Logs metrics and parameters via MLflow autolog.

### Q4: Launch the MLflow tracking server

```bash
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./artifacts \
  --host 127.0.0.1 \
  --port 5000
```

### Q5: Hyperparameter tuning with Hyperopt

```bash
python hpo.py --data_path ./output --num_trials 15
```

* Logs each trial (params + RMSE) to the `random-forest-hyperopt` experiment.

### Q6: Promote the best model to the registry

```bash
python register_model.py --data_path ./output --top_n 5
```

* Evaluates top 5 runs on the test set, selects the best, and registers it as **RandomForestTaxiModel**.

---

## MLflow UI

Start the UI (or use the tracking server above) and navigate to:

```
http://127.0.0.1:5000
```

* **Experiments**: view runs for each step (preprocess, train, HPO, best-models).
* **Model Registry**: inspect and promote versions of **RandomForestTaxiModel**.

---

## .gitignore

We ignore heavy or auto-generated files:

```
/01_Intro/Data/
/02_Experiment_Tracking/Data/
/02_Experiment_Tracking/mlruns/
/02_Experiment_Tracking/mlflow.db
```

---

## License

Distributed under the MIT License. Feel free to fork and adapt for your own experiments.
