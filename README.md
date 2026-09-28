# Student Mental Health Score Prediction

An end-to-end machine learning project that predicts a numeric `Mental_Health_Score` from student social media usage, study habits, sleep, physical activity, and stress level. The project covers exploratory analysis, data cleaning, feature preparation, model comparison, tuning, and export of a reusable prediction pipeline.

> **Scope:** This is a dataset-based regression exercise. Its predictions are not a clinical assessment or a measure of any individual's mental health.

## Project at a glance

| Item | Details |
| --- | --- |
| Task | Supervised regression |
| Dataset | `Student Social Media And Mental Health Impact.csv` |
| Data size | 5,000 rows and 13 original columns; two duplicate rows removed |
| Target | `Mental_Health_Score` |
| Evaluation | 70% training / 30% test split, `random_state=42` |
| Best test result | Default Random Forest: R² **0.878**, MAE **0.347**, RMSE **0.464** |
| Saved artifact | `Mental_Health_Model.pkl` (preprocessing and Random Forest together) |

## Dataset and approach

The original data includes age, gender, country, academic level, most used platform, purpose of use, average daily usage hours, daily unlocks, study hours, physical activity hours, sleep hours, stress level, and the target score. The notebook explores distributions, correlations, stress levels, usage, sleep, and platform counts.

The workflow:

1. Check missing values, duplicates, ranges, and numeric outliers. The dataset has no missing values; remove two duplicate rows and clip negative physical activity hours to zero.
2. Group less frequent countries into `Other` to reduce category count. Create `Grouped_country` for modeling.
3. Split features and target into training and test sets (70/30).
4. Use a scikit-learn `ColumnTransformer` within a `Pipeline`: apply `log1p` and scaling to `Study_Hours`, scale the other numeric features, ordinal encode `Stress_Level`, and one-hot encode the nominal features. Unknown nominal categories are ignored during transformation.
5. Train a Linear Regression baseline and a default Random Forest. Tune a separate Random Forest with `RandomizedSearchCV` (15 sampled configurations, five-fold cross-validation).
6. Compare models on the held-out test set and save the default Random Forest pipeline with `joblib`.

The model uses `Grouped_country` instead of raw `Country`. Country grouping and clipping are performed in notebook code **before** the saved pipeline, so any future prediction interface must repeat these steps on incoming records.

## Results

The following numbers come from the notebook's executed output on the held-out test set. Lower MAE and RMSE are better; higher R² is better.

| Model | Test R² | Test MAE | Test RMSE | Training R² |
| --- | ---: | ---: | ---: | ---: |
| Linear Regression | 0.740 | 0.536 | 0.676 | 0.724 |
| Random Forest (default) | **0.878** | **0.347** | **0.464** | 0.981 |
| Random Forest (tuned) | 0.865 | 0.369 | 0.487 | 0.955 |

The default Random Forest achieved the best held-out scores and is the model saved in `Mental_Health_Model.pkl`. Tuning did not improve its test performance. Its higher training R² also suggests some overfitting, so the test results are the more useful summary.

## Files

```text
.
├── README.md
├── ML_Project.ipynb
├── Student Social Media And Mental Health Impact.csv
└── Mental_Health_Model.pkl
```

## Run the notebook

1. Download the four files above into the same folder.
2. Create a Python environment and install the required libraries:

   ```bash
   python -m pip install numpy pandas matplotlib seaborn scikit-learn joblib notebook
   ```

3. Start Jupyter and run the notebook cells in order:

   ```bash
   jupyter notebook ML_Project.ipynb
   ```

The notebook reads the CSV by its exact filename and writes `Mental_Health_Model.pkl` to the current working directory. To load the exported pipeline, use `joblib.load("Mental_Health_Model.pkl")` only with a trusted file and a compatible scikit-learn environment. Re-running the notebook is the most reliable way to produce an artifact for your installed version.

## Current limitations and next steps

- The notebook's country grouping and physical activity clipping happen outside the exported pipeline; include them in a single reusable preprocessing component before serving predictions.
- The source and collection method of the dataset are not documented here; verify provenance and representativeness before interpreting results beyond this dataset.
- No web app or deployed prediction endpoint is included. A future interface can accept inputs, apply the same transformations, and call the saved pipeline.

## Tech stack

Python · pandas · NumPy · seaborn · Matplotlib · scikit-learn · joblib · Jupyter Notebook
