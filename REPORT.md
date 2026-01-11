# House Price Prediction — Project Report (Non‑Technical)

Contributors: Avtandil Beradze, Gigi Jishkariani, Giorgi Losaberidze

## Executive Summary
We built a system that estimates the price of a house using publicly available housing data from California. The goal is to help users quickly understand what drives house prices and to provide reasonable price estimates given area‑level characteristics. Our best model performed clearly better than a naïve “average price” baseline and explains a meaningful share of price variation. Exact scores for each model are saved in `results/final_model_comparison.csv`.

In plain language: higher local income is associated with higher prices, location matters, and room configuration gives extra signal (e.g., many bedrooms relative to rooms can indicate smaller rooms and lower prices).

## Dataset
- Source: California Housing dataset (from scikit‑learn; 1990 census).
- Rows: ~20k census block groups.
- Target (what we predict): median house value in the block group (in $100k).
- Main inputs (examples): median income, median house age, average rooms, average bedrooms, population, average household size, latitude, longitude.

## What We Did (Step‑by‑Step)
1) Data Loading and Quality Checks
   - Loaded the dataset, verified types, and checked for missing data (none significant).
2) Cleaning and Transformations
   - Standardized numerical features (for linear models).
   - Created simple, meaningful features:
     - BedroomsPerRoom = AveBedrms / AveRooms
     - RoomsPerPerson = AveRooms / AveOccup
     - RoomsMinusBedrooms = AveRooms − AveBedrms
3) Exploration (EDA)
   - Plotted price distribution, relationships with income, rooms, age, and geographic bands.
   - Observed clear positive relationship between income and prices; geographic variation by latitude.
4) Modeling
   - Baseline: DummyRegressor (predicts the mean) for a sanity check.
   - Linear Regression: simple, explainable baseline with scaling.
   - Ridge Regression: linear model with regularization; tuned a small alpha grid.
   - Decision Tree Regressor: captures non‑linear patterns; tuned depth and leaf size.
5) Evaluation
   - Train/test split (80/20) plus 5‑fold cross‑validation for robustness.
   - Metrics: R² (higher is better), RMSE and MAE (lower are better).
   - Results saved to CSV for transparency and reproducibility.

## Results (Where to Look)
- Main comparison: `results/final_model_comparison.csv`
- Best model performance summary: `results/best_model_performance.png`

What this means in practice:
- The tree and ridge models typically outperform the naïve baseline and simple linear model.
- Local median income is consistently the strongest single predictor of prices.
- Our engineered features add helpful nuance about room composition and occupancy.

## How to Use the Estimates
Provide the input fields (income, rooms, bedrooms, age, etc.), and the model outputs an estimated median price for that area. These estimates are most reliable for California‑like areas and for inputs similar to those seen in training (1990 census characteristics).

## Limitations and Assumptions
- Geography: Latitude/longitude are used at a coarse resolution; no detailed neighborhood effects.
- Time: Data reflect the 1990 housing market; results illustrate methodology rather than current market pricing.
- Scope: Estimates reflect median values at a block‑group level, not individual property valuations.

## Reproducibility
1) Follow the README to create a Python 3.11 virtual environment and install requirements.
2) Run notebooks in order:
   - `notebooks/01_data_exploration.ipynb`
   - `notebooks/02_data_preprocessing.ipynb`
   - `notebooks/03_eda_visualization.ipynb`
   - `notebooks/04_machine_learning.ipynb`
3) Check the `results/` and `models/` folders for outputs and saved models.

## Plain‑Language Glossary
- R²: How much of the price variation is explained by the model (closer to 1 is better).
- RMSE/MAE: Average size of prediction errors (lower is better).
- Regularization (Ridge): A method to prevent overfitting by shrinking coefficients.
- Permutation importance: Measures how much a model’s accuracy drops when a feature’s values are shuffled; larger drop = more important feature.

## Conclusion
We built a clear, reproducible workflow that cleans data, explores key patterns, and compares multiple models. The best model provides meaningful price estimates and insights into what drives prices. While the data are historical and coarse, the pipeline demonstrates solid predictive modeling practices suitable for a university final project.


