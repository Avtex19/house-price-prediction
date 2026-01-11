Final Data Science Project — California Housing Price Prediction
==============================================================

Overview
--------
This repository contains an end‑to‑end workflow for predicting median house values using the California Housing dataset (1990 census). The project demonstrates a clear and reproducible data science pipeline:
- Data loading and checks
- Feature engineering and preprocessing
- Exploratory Data Analysis (EDA)
- Model training, tuning, and comparison
- Results saving and basic model persistence

The work is organized into Jupyter notebooks, with processed datasets, saved models, and result artifacts tracked in version control for easy review.

Repository Structure
--------------------
- `data/`
  - `raw/`
    - `california_housing.csv` — original dataset (block‑group level)
  - `processed/`
    - `train.csv`, `test.csv` — splits produced by preprocessing
    - `preprocessing_config.json` — configuration used for preprocessing and feature engineering
- `notebooks/`
  - `01_data_exploration.ipynb` — initial data loading and sanity checks
  - `02_data_preprocessing.ipynb` — feature engineering and dataset splitting
  - `03_eda_visualization.ipynb` — EDA plots and insights
  - `04_machine_learning.ipynb` — model training, tuning, evaluation
- `models/`
  - `decision_tree_tuned.pkl`, `linear_regression.pkl` — persisted example models
- `results/`
  - `final_model_comparison.csv` — side‑by‑side metrics for trained models
  - `best_model_performance.png` — visual summary of top model performance
- `README.md` — this file
- `REPORT.md` — non‑technical project report
- `requirements.txt` — pinned dependencies for reproducibility

Dataset
-------
- Source: California Housing dataset (scikit‑learn; 1990 census)
- Rows: ~20k census block groups
- Target: `MedHouseVal` (median house value, in $100k)
- Core inputs (examples): `MedInc`, `HouseAge`, `AveRooms`, `AveBedrms`, `Population`, `AveOccup`, `Latitude`, `Longitude`
- Engineered features (see `data/processed/preprocessing_config.json`):
  - `BedroomsPerRoom = AveBedrms / AveRooms`
  - `RoomsPerPerson = AveRooms / AveOccup`
  - `RoomsMinusBedrooms = AveRooms − AveBedrms`

Quick Start
-----------
1) Create and activate a virtual environment (Python 3.11 recommended).

macOS/Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows (Powershell):
```bash
python -m venv .venv
.venv\\Scripts\\Activate.ps1
```

2) Install dependencies:
```bash
pip install -r requirements.txt
```

3) Open the notebooks in order:
```bash
ipython kernel install --user --name=final-ds
jupyter notebook notebooks/
```
Run:
- `01_data_exploration.ipynb`
- `02_data_preprocessing.ipynb`
- `03_eda_visualization.ipynb`
- `04_machine_learning.ipynb`

After execution, check:
- `results/final_model_comparison.csv`
- `results/best_model_performance.png`
- `models/` for saved models

Reproducibility and Configuration
---------------------------------
Preprocessing settings and features are captured in:
- `data/processed/preprocessing_config.json`
  - `target_column`: `MedHouseVal`
  - `feature_columns`: original plus engineered features
  - `outlier_removal`: IQR method with factor 3.0
  - `scaler`: `StandardScaler()`
  - `split`: 80/20 with `random_state` 42

Models and Results
------------------
- Trained models and parameters are explored in `04_machine_learning.ipynb`.
- Comparison metrics (e.g., R², RMSE, MAE) are summarized in:
  - `results/final_model_comparison.csv`
- A compact visual of the top model is provided in:
  - `results/best_model_performance.png`
- Example persisted models:
  - `models/linear_regression.pkl`
  - `models/decision_tree_tuned.pkl`

Minimal Inference Example
-------------------------
If you want to load a saved model and run a quick prediction in Python:

```python
import joblib
import numpy as np

# Example: load decision tree model
model = joblib.load("models/decision_tree_tuned.pkl")

# Example feature vector matching the training feature order
# Replace with real values and proper preprocessing if required.
example_features = np.array([[
    3.5,   # MedInc
    20.0,  # HouseAge
    5.0,   # AveRooms
    1.0,   # AveBedrms
    1500,  # Population
    3.0,   # AveOccup
    34.5,  # Latitude
    -119,  # Longitude
    0.2,   # BedroomsPerRoom
    1.7,   # RoomsPerPerson
    4.0    # RoomsMinusBedrooms
]])

pred = model.predict(example_features)[0]
print(f"Predicted median house value (in $100k units): {pred:.3f}")
```

Notes:
- For best results, replicate the same preprocessing used in the notebooks (scaling, feature engineering).
- The persisted models in this repository are intended for demonstration; retrain for production use.

Troubleshooting
---------------
- Kernel issues: Ensure the venv is active and the kernel is set to your environment (e.g., `final-ds`).
- Dependency mismatches: Use the pinned versions in `requirements.txt`. If using Apple Silicon, ensure you have a compatible Python build.
- Missing artifacts: Run notebooks in order to regenerate `results/` and `models/`.

Acknowledgments
---------------
- Scikit‑learn team for the California Housing dataset
- Project contributors: Avtandil Beradze, Gigi Jishkariani, Giorgi Losaberidze

