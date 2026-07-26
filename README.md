# Cyber-Physical Anomaly Detection for DER Systems

This repository documents experiments for detecting anomalous operating states in cyber-physical Distributed Energy Resource (DER) simulator data. The task is a binary classification problem:

| Target | Meaning |
| --- | --- |
| `0` | Good / normal operation |
| `1` | Bad / anomalous, non-compliant, or cyber-compromised operation |

This README adds an experiment-focused view of the methods, features, model families, thresholds, and submission results tried in the notebooks under `experiments/`.

## Project Context

The data comes from DERSec DER Simulator logs for three-phase photovoltaic systems. Each row represents a cyber-physical state snapshot containing SunSpec Modbus settings, electrical measurements, advanced DER control parameters, and out-of-band site measurements such as irradiance, grid voltages, and frequency.

The feature space contains 737 simulator-derived columns spanning:

- Cyber/interoperability settings and commands.
- Physical AC/DC measurements such as voltage, current, real power, reactive power, apparent power, and power factor.
- Control settings for IEEE 1547-2018 grid-support modes such as Volt-Var, Volt-Watt, Watt-Var, Frequency-Droop, power factor, reactive power, and maximum power controls.
- Site/context measurements such as irradiance, grid voltage phases, and grid frequency.

The anomaly classes represented by the target include:

- Bad commands outside expected IEEE 1547-2018 ranges.
- Invalid protocol enum or bitfield values.
- Missing SunSpec Modbus data.
- Data falsification through machine-in-the-middle manipulation of SunSpec communications.

## Evaluation Metric

All experiments optimize the F2 score:

```text
F2 = 5 * (precision * recall) / (4 * precision + recall)
```

F2 weights recall higher than precision, which is appropriate for critical infrastructure anomaly detection because missing a true anomaly is more costly than raising some false alarms.

## Repository Layout

```text
Cyber-Physical-Anomaly-Detection/
|-- README.md
|-- Dockerfile.lgbm
|-- run_docker_lgbm.sh
|-- run_docker_lgbm.ps1
|-- .dockerignore
|-- .gitignore
|-- src/
|   |-- __init__.py
|   |-- optuna_lgbm_runner.py
|-- experiments/
|   |-- detecting-anomalies-baseline.ipynb
|   |-- detecting-anomalies-optuna-xgb.ipynb
|   |-- detecting-anomalies-optuna-lgbm.ipynb
|   |-- detecting-anomalies-optuna-cb.ipynb
|   |-- domain-specific-feature-engineering.ipynb
|   |-- detecting-anomalies-lgbm-digit-decomposition.ipynb
|   |-- detecting-anomalies-lgbm-digit-decompostion.ipynb
|   |-- anomaly-detection-lgbm-oof-preds.ipynb
|   |-- anomaly-detection-dnn-approach.ipynb
|   |-- seed-averaging.ipynb
|   |-- training-on-full-dataset.ipynb
|-- submissions/
|   |-- submission_*.csv
|-- data/
|   |-- train.csv
|   |-- test.csv
|-- kaggle-working/
    |-- submission_LGBM.csv
```

`data/` and `kaggle-working/` are local-only folders ignored by Git. `data/` holds the Kaggle competition CSVs, while `kaggle-working/` receives Docker-generated outputs.

Note: `detecting-anomalies-lgbm-digit-decompostion.ipynb` appears to be an earlier misspelled duplicate/variant of the digit-decomposition experiment.

## Common Experiment Pipeline

Most notebooks follow the same structure:

1. Load data with `polars`.
2. Drop high-null or unusable columns and separate labels from features.
3. Run basic target/null exploratory analysis where needed.
4. Split training data into train/validation partitions.
5. Build preprocessing with `sklearn`:
   - `OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)` for categorical columns.
   - `ColumnTransformer` with passthrough for remaining numeric columns.
   - `Pipeline` wrappers for preprocessing plus model training.
6. Train a model and predict validation probabilities.
7. Tune the classification threshold directly for F2 instead of using the default `0.5` cutoff.
8. Predict test probabilities, apply the selected threshold, and write a submission CSV.

## Feature Engineering Tried

### Baseline Features

The baseline experiments use the provided simulator columns after null handling and categorical encoding. This established a strong reference before adding domain-specific transformations.

### Domain-Specific DER Features

The feature-engineering notebook adds three physics-inspired ratios:

| Feature | Formula | Motivation |
| --- | --- | --- |
| `power_factor` | `DERMeasureAC[0].W / DERMeasureAC[0].VA` | Captures real-power share of apparent power. |
| `power_per_amp` | `DERMeasureAC[0].W / DERMeasureAC[0].A` | Relates output power to current draw. |
| `power_utilization` | `DERMeasureAC[0].W / DERCapacity[0].WMax` | Measures real-power output relative to rated capacity. |

Small epsilons are added to denominators to avoid division by zero.

### Target Encoding

`domain-specific-feature-engineering.ipynb` includes a custom sklearn-compatible `TargetEncoder` transformer for selected categorical columns. The transformer maps category values to mean target rates learned from the training partition, then fills unseen values with the global target mean.

### Digit and Modulo Decomposition

The LGBM digit-decomposition experiments create synthetic features from the first numeric columns:

- Decimal digit features: `feature_digit_0` through `feature_digit_3`.
- Modulo features: `feature_mod10`, `feature_mod50`, and `feature_mod100`.

This was tested as a way to expose register-like patterns, encoded value ranges, or low-order numerical artifacts that tree models may exploit.

## Models and Methods

### XGBoost Baseline

`detecting-anomalies-baseline.ipynb` trains an initial `XGBClassifier` with encoded categorical columns and threshold tuning. This experiment produced the lowest submission score but served as the reference point for later tuning.

Notebook validation:

- Best threshold: `0.31`
- Validation F2: `0.87663`
- Submission score: `0.87855`

### Optuna-Tuned XGBoost

`detecting-anomalies-optuna-xgb.ipynb` tunes core XGBoost hyperparameters with validation F2 as the objective. The tuned setup uses histogram tree construction and GPU execution where available.

Best noted parameters include:

- `n_estimators=2382`
- `learning_rate=0.0369895`
- `max_depth=12`
- `min_child_weight=4`
- `gamma=2.22461`
- `subsample=0.61093`
- `colsample_bytree=0.94178`
- `reg_alpha=4.766e-06`
- `reg_lambda=1.019e-08`

Notebook validation:

- Best threshold: `0.12`
- Validation F2: `0.91017`
- Submission score: `0.91147`

### Optuna-Tuned LightGBM

`detecting-anomalies-optuna-lgbm.ipynb` applies the same threshold-aware tuning idea to `LGBMClassifier`.

Best noted parameters include:

- `n_estimators=3401`
- `learning_rate=0.0200096`
- `num_leaves=76`
- `max_depth=10`
- `min_child_samples=35`
- `subsample=0.99744`
- `colsample_bytree=0.61619`
- `reg_alpha=0.16956`
- `reg_lambda=0.16021`

Notebook validation:

- Best threshold: `0.13`
- Validation F2: `0.91019`
- Best related submission score: `0.91150`

### Optuna-Tuned CatBoost

`detecting-anomalies-optuna-cb.ipynb` tests `CatBoostClassifier` with GPU-oriented settings. The tuning search includes:

- `iterations`
- `depth`
- `learning_rate`
- `l2_leaf_reg`

Notebook validation:

- Best threshold: `0.16`
- Validation F2: `0.90989`
- Submission score: `0.91120`

### LGBM with Digit/Modulo Decomposition

The digit-decomposition notebooks add digit and modulo features before fitting tuned LightGBM models. One completed variant reports:

- Best threshold: `0.15`
- Validation F2: `0.91018`

The corresponding submission result in the folder is close to the best LGBM score, suggesting that digit decomposition was competitive but not clearly superior to tuned LGBM alone.

### OOF LightGBM Predictions

`anomaly-detection-lgbm-oof-preds.ipynb` trains a 5-fold `StratifiedKFold` LightGBM ensemble and stores out-of-fold probabilities. The method averages test probabilities across folds and searches thresholds using both per-fold validation predictions and global OOF predictions.

Observed fold results:

| Fold | F2 | Threshold |
| --- | ---: | ---: |
| 0 | 0.90960 | 0.1090 |
| 1 | 0.91005 | 0.2377 |
| 2 | 0.91022 | 0.1288 |
| 3 | 0.90989 | 0.1288 |
| 4 | 0.91013 | 0.1931 |

Summary:

- Mean fold F2: `0.90998`
- Global OOF threshold: `0.01421`
- Global OOF F2: `0.90943`
- Submission score: `0.91149`

### Deep Neural Network

`anomaly-detection-dnn-approach.ipynb` evaluates a TensorFlow/Keras dense neural-network approach after domain-specific feature engineering, one-hot/scaled preprocessing, and a custom TensorFlow F2 metric. The custom metric uses an internal threshold of `0.2`.

Notebook validation:

- Reported sklearn F2: `0.90896`
- Submission score: `0.89455`

This was weaker than the tree-based methods on the submitted test set.

### Seed Averaging

`seed-averaging.ipynb` trains the tuned XGBoost configuration across multiple random seeds and averages predicted probabilities. This was intended to reduce variance from a single seed.

Observed submission:

- XGBoost seed averaging: `0.91147`
- LGBM seed averaging with threshold `0.12`: `0.91149`

### Training on Full Dataset

`training-on-full-dataset.ipynb` retrains the tuned XGBoost model on the full available training set after selecting hyperparameters and threshold from earlier validation experiments.

Observed submission:

- Tuned XGBoost full dataset: `0.91146`

## Submission Results

Scores below are inferred from the filenames in `submissions/`.

| Submission file | Method | Score |
| --- | --- | ---: |
| `submission_LGBM - 0.91150.csv` | Tuned LightGBM | 0.91150 |
| `submission_LGBM - 0.91149.csv` | Tuned LightGBM variant | 0.91149 |
| `Submission_OOF_preds_thr(0.1) - 0.91149.csv` | 5-fold LGBM OOF predictions | 0.91149 |
| `submission_seed_avg_lgbm_thr(0.12) - 0.91149.csv` | LGBM seed averaging | 0.91149 |
| `submission_seed_averaging_xgb - 0.91147.csv` | XGBoost seed averaging | 0.91147 |
| `submission_optuna_xgb - 0.91147.csv` | Optuna-tuned XGBoost | 0.91147 |
| `submission_tuned_xgb_fulldataset - 0.91146.csv` | Tuned XGBoost trained on full dataset | 0.91146 |
| `submission_FE_targetEncoding_included - 0.91145.csv` | Domain features + target encoding | 0.91145 |
| `submission_domain_specific_FE - 0.91145.csv` | Domain-specific feature engineering | 0.91145 |
| `submission_CB - 0.91120.csv` | Optuna-tuned CatBoost | 0.91120 |
| `submission_DNN_thresh(0.1)- 0.89455.csv` | DNN approach | 0.89455 |
| `submission_baseline - 0.87855.csv` | Baseline XGBoost | 0.87855 |

The best recorded submission is `submission_LGBM - 0.91150.csv`, with several other LightGBM and XGBoost variants within `0.00005` of it.

## Key Findings

- Gradient-boosted tree models dominate this dataset.
- Threshold tuning is essential because the target metric is F2, not log loss or accuracy.
- LightGBM produced the best recorded submission score and the most consistently strong variants.
- XGBoost improved substantially after Optuna tuning, moving from the baseline `0.87855` submission to roughly `0.91147`.
- CatBoost was competitive but slightly below the best LightGBM/XGBoost submissions.
- The DNN achieved a decent validation F2 but generalized worse in the submitted result.
- Domain-inspired features were reasonable and interpretable, but the recorded submissions show only marginal gains or neutral impact compared with the strongest tuned tree baselines.
- OOF and seed averaging helped produce stable high-scoring submissions, though not a large jump over the best single tuned LightGBM.

## Reproducibility Notes

The notebooks expect the competition data files to be available in the runtime environment. They were not included in this repository snapshot, so exact reruns require placing the original `train.csv`, `test.csv`, and sample submission files where the notebooks expect them.

### Dockerized LightGBM Notebook

`experiments/detecting-anomalies-optuna-lgbm.ipynb` has been converted into a repeatable batch runner at `src/optuna_lgbm_runner.py`.

The recommended way to run this converted notebook is through the pinned Kaggle CPU Docker image. That path matches the original Kaggle notebook runtime more closely than a lightweight local Python environment.

#### 1. Prepare the data

If you have the Kaggle CLI configured, download the competition files into `data/`:

```bash
mkdir -p data
kaggle competitions download -c cyber-physical-anomaly-detection-for-der-systems -p data
```

On Windows PowerShell:

```powershell
New-Item -ItemType Directory -Force -Path data
kaggle competitions download -c cyber-physical-anomaly-detection-for-der-systems -p data
```

That should produce:

```text
data/train.csv
data/test.csv
data/sample_submission.csv
```

`sample_submission.csv` is optional for this runner; `train.csv` and `test.csv` are required.

#### 2. Build the wrapper image

Build the Kaggle-based CPU image:

```bash
docker build --platform linux/amd64 -f Dockerfile.lgbm -t der-optuna-lgbm .
```

This image does not manually install the notebook dependencies. Instead, it pins the same Kaggle Python base image used by the original notebook environment, which already includes packages such as `polars`, `scikit-learn`, `lightgbm`, and `optuna`.

On Apple Silicon, keep `--platform linux/amd64` for both `docker build` and `docker run`; the pinned Kaggle image is built for `linux/amd64`.

#### 3. Run the pinned workflow

Run the full LightGBM pipeline:

```bash
./run_docker_lgbm.sh
```

On Windows PowerShell:

```powershell
.\run_docker_lgbm.ps1
```

The script:

- mounts the repo root at `/workspace`
- mounts `data/` at `/kaggle/input/competitions/cyber-physical-anomaly-detection-for-der-systems`
- mounts `kaggle-working/` at `/kaggle/working`
- runs `python -m src.optuna_lgbm_runner` inside the container

For a quick smoke test, use fewer rows and fewer trees:

```bash
./run_docker_lgbm.sh --limit-rows 10000 --n-estimators 50
```

On Windows PowerShell:

```powershell
.\run_docker_lgbm.ps1 --limit-rows 10000 --n-estimators 50
```

#### 4. Collect the output

The submission is written to:

```text
kaggle-working/submission_LGBM.csv
```

Python packages used across the notebooks include:

- `polars`
- `pandas`
- `numpy`
- `scikit-learn`
- `xgboost`
- `lightgbm`
- `catboost`
- `optuna`
- `tensorflow`
- `matplotlib`

Because the experiments tune thresholds on validation probabilities, the exact split seed and preprocessing pipeline should be preserved when comparing validation F2 scores. Submission scores should be treated as the final comparable metric across notebooks.
