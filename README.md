# Green Smart Campus: Energy Saved Predictor

Professional Training in AI 1 - Project 1.
A regression model that predicts **`Total_Energy_Saved_kWh`** from campus sensor readings, deployed as a Streamlit app.

**Live app:** `<<ADD YOUR STREAMLIT COMMUNITY CLOUD LINK HERE AFTER DEPLOYING>>`

## Results (test set, 535 rows never used for any decision)

| Metric | Value |
|---|---|
| R2 | 0.913 |
| MAE | 2.86 kWh |
| RMSE | 3.48 kWh |
| 5-fold cross-validation R2 (training set) | 0.917 |

Model: Ridge regression in a pipeline (scaling + one-hot encoding + `SelectKBest`), tuned with grid search over the number of kept features and the Ridge penalty.
The best model keeps a single feature, `Solar_Generation_kWh`.

## What we found

* The target is **70% to 100% of `Solar_Generation_kWh`** in every row, so one feature explains almost everything. The other 14 candidate features (water, waste, occupancy, weather, energy consumption, building, day type) carry no usable information.
* `Solar_Generation_kWh` is exactly `0.05 x (Beam + Diffuse irradiance)`, so those three columns are redundant (VIF in the millions). Only solar generation is kept.
* The test R2 (0.913) is close to the best possible score (about 0.918), because the remaining error is a random 70-100% factor that no column explains.
* The data looks simulated (all features uniform and independent), so the results should not be assumed to carry over to a real campus.

## Repository structure

```
app.py                              Streamlit app (Stage 5)
models/                             Pickle files: preprocessor.pkl, selector.pkl, model.pkl, feature_info.pkl
green_campus_project.ipynb          Notebook with stages 1 to 4 (already executed, with outputs)
green_campus_project.py             The same code as a script ("# %%" cells, for Spyder)
data/green_smart_campus_dataset.csv The dataset
figures/                            Plots saved by the notebook
CRCTargetEncoder.py                 Custom CRC32 target encoder (only used in the encoding comparison in the notebook)
test_app.py                         Optional automated check of the app
requirements.txt                    Packages needed by the app
requirements-notebook.txt           Extra packages for re-running the notebook
```

## Run the app locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Re-run stages 1 to 4

```bash
pip install -r requirements-notebook.txt
jupyter notebook green_campus_project.ipynb
```

In Spyder, open `green_campus_project.py`, set the working directory to this folder, and run it with F5 (or cell by cell with Ctrl+Enter).
Running it again regenerates the plots and the files in `models/`.

## Deploy on Streamlit Community Cloud

1. Push this folder to a **public** GitHub repository.
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **Create app**, choose the repository, the branch (`main`) and the main file `app.py`.
4. Open **Advanced settings** and select **Python 3.12** (the version used to create the pickle files).
5. Click **Deploy** and wait for the build. Copy the public link into this README.

## Test the live app

Use "Enter it directly" and click **Predict energy saved**:

| Solar generation (kWh) | Expected prediction | Expected range |
|---|---|---|
| 15.0 | 12.84 kWh | 10.83 to 14.88 kWh |
| 38.6 | 32.79 kWh | 27.65 to 37.99 kWh |
| 60.0 | 50.88 kWh | 42.91 to 58.95 kWh |

Use "Calculate it from sunlight": beam irradiance 800 and diffuse irradiance 300 give solar generation 55.0 kWh and a prediction of 46.65 kWh (range 39.34 to 54.06 kWh).
A value outside the training range (for example 150 kWh) shows a warning.

## Notes

* The pickle files were created with scikit-learn 1.9.1, pandas 3.0.6, numpy 2.5.4 and Python 3.12. `requirements.txt` pins the same versions, because pickle files can fail to load with a different scikit-learn version.
  If you see an unpickling error after changing versions, re-run the notebook in your own environment (it recreates `models/`) and update `requirements.txt` with your versions.
**Dataset Source:** MIT Office of Sustainability — Sustainability DataPool (https://datapool.mit.edu/), distributed as the Green Smart Campus Sustainability & Energy Dataset (`green_smart_campus_dataset.csv`).
