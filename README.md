# 🌴 California Housing Price Prediction

A regression project on the **California Housing** dataset (1990 U.S. census, 20,640 districts). It predicts a
district's median house value from income, housing and location features. Includes exploratory data analysis, a
cross-validated comparison of four regression models, and an interactive **Streamlit** demo.

**🔗 Live demo:** https://california-housing-price-prediction-rywery2h5dmajnv5xumnal.streamlit.app/

## 📊 Dataset

Loaded with `sklearn.datasets.fetch_california_housing`, and saved as `california_housing.csv` so the app doesn't
need to download anything.

| Feature | Meaning |
|---|---|
| `MedInc` | median household income (in $10,000s) |
| `HouseAge` | median house age (capped at 52) |
| `AveRooms`, `AveBedrms` | average rooms / bedrooms per household |
| `Population`, `AveOccup` | district population, average household size |
| `Latitude`, `Longitude` | district location |
| **`MedHouseVal`** (target) | median house value in $100,000s, **capped at 5.0 ($500,000)** |

## 🚀 Features

- **EDA notebook:** distributions, the income-vs-value relationship, correlations, outliers and a location map.
- **`california_housing.py`:**
  - data loading;
  - a 5-fold cross-validated comparison of the models listed for this project;
  - training and saving the best model.
- **Streamlit app:**
  - predict a district's median value from its features;
  - quick locations for six major cities;
  - input validation and a range warning;
  - the median value of real districts nearby;
  - a "what if income changed" chart;
  - a statewide price map, distribution and correlations;
  - model comparison and feature importance.

## 📈 Results

5-fold cross-validation, `KFold(shuffle=True, random_state=42)`:

| Model | R² | RMSE | MAE |
|---|---|---|---|
| **Gradient Boosting** | **0.844 ± 0.006** | **$45,500** | **$29,900** |
| Random Forest | 0.812 ± 0.008 | $50,100 | $32,500 |
| Decision Tree | 0.709 ± 0.011 | $62,200 | $41,100 |
| Linear Regression | 0.601 ± 0.017 | $72,800 | $53,200 |

The app uses the best model, Gradient Boosting, saved in `models/model.joblib`. If that file can't be loaded (for
example, after a scikit-learn upgrade), the app retrains it automatically in about 25 seconds.

## 🛠️ Tech Stack

Python · pandas · NumPy · scikit-learn · Matplotlib · Seaborn · Altair · Streamlit · Jupyter

## 📂 Project Structure

```
California-Housing-Price-Prediction/
├── clofornia houses.ipynb      # EDA notebook
├── california_housing.csv      # Dataset (from fetch_california_housing)
├── california_housing.py       # Data loading, model comparison, training
├── models/
│   ├── metrics.json            # Cross-validation results
│   └── model.joblib            # Trained Gradient Boosting model
├── app.py                      # Streamlit demo
├── requirements.txt
└── .streamlit/config.toml
```

## 📦 Installation

```bash
git clone https://github.com/amirafarhan/California-Housing-Price-Prediction.git
cd California-Housing-Price-Prediction
python -m venv venv
# Windows: venv\Scripts\activate    Linux/Mac: source venv/bin/activate
pip install -r requirements.txt
```

## ▶️ Run

Live demo: https://california-housing-price-prediction-rywery2h5dmajnv5xumnal.streamlit.app/

Run the Streamlit demo locally (opens http://localhost:8501):

```bash
streamlit run app.py
```

Re-run the model comparison and retrain the saved model (about 2 minutes):

```bash
python california_housing.py
```

Notebook: run `pip install jupyter matplotlib seaborn`, then `jupyter notebook`.

## 👩‍💻 Author

Amira Farhan — Machine Learning & Computer Vision Engineer
