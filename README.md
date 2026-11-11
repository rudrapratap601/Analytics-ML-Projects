# Analytics-ML-Projects

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=for-the-badge&logo=python&logoColor=white)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-4c72b0?style=for-the-badge&logo=python&logoColor=white)](https://seaborn.pydata.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)

A structured collection of hands-on **Data Science and Machine Learning projects, statistical experiments, and predictive modeling implementations** built by **Rudrapratap Sarma** as part of my journey toward becoming an **AI/ML Engineer**.

This repository emphasizes core ML engineering disciplines: rigorous Exploratory Data Analysis (EDA), multicollinearity mitigation, domain-specific feature engineering, leakage-free preprocessing, algorithmic benchmarking, residual diagnostics, and production-grade project structuring.

---

## 📌 Repository Architecture

```text
Analytics-ML-Projects/
├── YBI_Foundation_Final_Project/      # Fish Weight Prediction (Completed)
│   ├── Fish.csv                       # Morphometric fish dataset (159 records)
│   ├── Fish_Weight_Prediction.ipynb   # Full EDA, OHE, Linear Regression & Random Forest
│   └── README.md                      # Detailed project documentation
│
├── house_price_prediction/            # California Housing Price Prediction (Completed)
│   ├── house_price_prediction_model.ipynb # Multicollinearity fix, 14 engineered features, regression & diagnostics
│   └── README.md                      # Detailed project documentation
│
├── rossmann-sales-forecasting/        # Retail Store Sales Forecasting (In Progress)
│   ├── data/                          # raw/ and processed/ data stores
│   ├── notebooks/                     # Time-series EDA and prototype experiments
│   ├── reports/figures/               # Visualizations and diagnostic plots
│   ├── saved_models/                  # Serialized pipelines and model weights
│   ├── src/                           # Modular production code (data, features, models, utils)
│   ├── .gitignore
│   └── README.md                      # Architecture and roadmap overview
│
├── .gitignore                         # Workspace ignore rules
└── README.md                          # Repository overview & project index
```

---

## 🚀 Projects Overview

| Project | Problem Domain | Model(s) | Primary Metrics | Key Engineering Techniques | Status |
| :--- | :--- | :--- | :--- | :--- | :---: |
| **[Fish Weight Prediction](./YBI_Foundation_Final_Project)** | Biometrics / Regression | • Linear Regression<br>• **Random Forest Regressor** | **$R^2 = 0.969$**<br>$\text{MAE} = 44.49\text{g}$<br>$\text{RMSE} = 65.90\text{g}$ | • One-Hot Encoding<br>• Outlier retention rationale<br>• Feature importance analysis (Width = 71.4%) | ✅ Completed |
| **[California House Price Prediction](./house_price_prediction)** | Real Estate / Regression | • **Multiple Linear Regression** | **$R^2 = 0.6119$**<br>$\text{Adjusted } R^2 = 0.6111$<br>$\text{MAE} = 0.5265$<br>$\text{RMSE} = 0.7137$ | • Multicollinearity removal ($r > 0.84$ dropped)<br>• Polynomial & Interaction features<br>• Log transformations & scaling<br>• Residual error diagnostics | ✅ Completed |
| **[Rossmann Store Sales Forecasting](./rossmann-sales-forecasting)** | Retail Time-Series / Forecasting | • Linear Regression<br>• Ridge / Lasso / ElasticNet *(Planned)* | Target Metric: **RMSPE** | • Modular package architecture (`src/`)<br>• Calendar & promotional lag features<br>• Rolling window aggregations | 🚧 In Progress |

---

## 🔬 In-Depth Project Summaries

### 1. 🐟 Fish Weight Prediction
* **Directory:** [`./YBI_Foundation_Final_Project`](./YBI_Foundation_Final_Project)
* **Goal:** Estimate fish weight based on vertical, diagonal, and cross length measurements, height, width, and species classification.
* **Dataset:** 159 observations across 7 distinct freshwater species (Bream, Roach, Whitefish, Parkki, Perch, Pike, Smelt).
* **Key Insights & Methodology:**
  * **Exploratory Analysis:** Identified strong positive correlations ($r > 0.90$) between all length dimensions and fish weight, with right-skewed weight distributions driven by heavier species like Pike and Bream.
  * **Biological Outlier Strategy:** Extreme measurements were retained rather than pruned, recognizing them as natural biological diversity rather than measurement corruption.
  * **Feature Preprocessing:** Applied one-hot encoding on `Species` (`drop_first=True`) and standardized continuous numerical measurements.
  * **Comparative Performance:**

| Model | MAE (g) | RMSE (g) | $R^2$ Score | Outcome |
| :--- | :---: | :---: | :---: | :--- |
| Linear Regression | 65.30 | 83.71 | 0.951 | Reliable linear baseline |
| **Random Forest Regressor** | **44.49** | **65.90** | **0.969** | **Selected Model (Best generalization)** |

  * **Key Takeaway:** Random Forest feature importances revealed **`Width` (71.38%)** as the single most critical mass predictor, validating biological intuition that fish volume and cross-sectional density scale cubically rather than purely linearly with length.

---

### 2. 🏡 California House Price Prediction
* **Directory:** [`./house_price_prediction`](./house_price_prediction)
* **Goal:** Predict median block group home values in California using socio-economic, architectural, and geographic census attributes.
* **Dataset:** Scikit-Learn California Housing dataset (20,640 records, 8 baseline features).
* **Key Insights & Methodology:**
  * **Multicollinearity Removal:** Discovered severe collinearity between `AveBedrms` and `AveRooms` ($r = 0.848$) and `Latitude` and `Longitude` ($r = -0.925$). Removed `AveBedrms` and `Longitude` to protect linear model coefficient stability.
  * **Domain Feature Engineering:** Engineered 7 specialized interaction and polynomial terms:
    * `RoomsPerPerson`: Household living density ($\frac{\text{AveRooms}}{\text{AveOccup}}$)
    * `PopulationDensity`: Census block density proxy ($\frac{\text{Population}}{\text{AveOccup}}$)
    * `IncomeAge`: Interaction of neighborhood maturity and earning power (`MedInc` $\times$ `HouseAge`)
    * `IncomeLatitude`: Geographic affluence indicator (`MedInc` $\times$ `Latitude`)
    * `MedInc_squared` & `MedInc_cubed`: Higher-order polynomial features capturing non-linear price elasticity
    * `Log_Population` & `Log_AveOccup`: $\log(1 + x)$ compression of severe right-tail distribution skew
  * **Data Splitting & Scaling:** 70/30 train-test split (`random_state=42`) with `StandardScaler` fitted strictly on training data to prevent data leakage.
  * **Performance & Diagnostics:**
    * **$R^2$ Score:** `0.6119` (Adjusted $R^2$: `0.6111`) across 14 engineered features.
    * **MAE:** `0.5265` (~$52,650 average absolute deviation).
    * **RMSE:** `0.7137` (~$71,370 standard error).
    * **Residual Diagnostics:** Mean residual of `0.0042` with symmetric, zero-centered Gaussian distribution, demonstrating an unbiased linear model.

---

### 3. 📈 Rossmann Store Sales Forecasting
* **Directory:** [`./rossmann-sales-forecasting`](./rossmann-sales-forecasting)
* **Goal:** Forecast daily store sales across 1,115+ Rossmann drug stores in Germany, accounting for promotions, local competition, school/state holidays, and store formats.
* **Architecture & Standards:**
  * Built using a **production-oriented modular ML architecture**:
    * `src/data/`: Ingestion and schema validation routines
    * `src/features/`: Temporal feature extraction, rolling averages, and lag generation
    * `src/models/`: Baseline Linear Regression, Ridge, Lasso, and ElasticNet estimators
    * `src/utils/`: Custom evaluation metrics (RMSPE) and artifact serialization
  * **Current Status:** Foundation, directory scaffold, `.gitignore`, and module boundaries established; exploratory analysis and modeling pipeline under active development.

---

## 🧠 End-to-End Machine Learning Workflow

Across all projects, experiments follow a rigorous, reproducible engineering pipeline:

```text
┌─────────────────────────────────┐
│       1. Data Ingestion         │  Direct APIs, CSVs, and Scikit-Learn loaders
└────────────────┬────────────────┘
                 │
┌────────────────▼────────────────┐
│   2. Exploratory Data Analysis   │  Distributions, skewness, outliers, & pair correlations
└────────────────┬────────────────┘
                 │
┌────────────────▼────────────────┐
│ 3. Data Cleaning & Sanitization │  Missing value handling, deduplication, & multicollinearity mitigation
└────────────────┬────────────────┘
                 │
┌────────────────▼────────────────┐
│     4. Feature Engineering      │  Domain ratios, polynomial terms, log transforms, & encoding
└────────────────┬────────────────┘
                 │
┌────────────────▼────────────────┐
│    5. Leak-Free Preprocessing   │  Train/Test Split (70/30 or 80/20) → Scalers fit on train only
└────────────────┬────────────────┘
                 │
┌────────────────▼────────────────┐
│   6. Model Training & Tuning    │  Linear models, regularized regression, & tree ensembles
└────────────────┬────────────────┘
                 │
┌────────────────▼────────────────┐
│ 7. Evaluation & Residual Checks │  MAE, MSE, RMSE, R², Adjusted R², homoscedasticity & error plots
└─────────────────────────────────┘
```

---

## 🛠️ Tech Stack & Libraries

* **Language:** Python 3.9+
* **Data Manipulation:** `pandas`, `numpy`
* **Machine Learning & Modeling:** `scikit-learn` (`LinearRegression`, `RandomForestRegressor`, `StandardScaler`, `train_test_split`)
* **Visualization:** `matplotlib`, `seaborn`
* **Development Environment:** JupyterLab / Jupyter Notebook, Anaconda, VS Code

---

## ⚡ Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/rudrapratapsarma/Analytics-ML-Projects.git
cd Analytics-ML-Projects
```

### 2. Set Up a Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install numpy pandas scikit-learn matplotlib seaborn jupyter
```

### 4. Launch JupyterLab / Notebooks
```bash
jupyter lab
```
Navigate to any project directory (e.g., `YBI_Foundation_Final_Project/` or `house_price_prediction/`) and run the notebooks step-by-step.

---

## 📈 Roadmap

- [x] **Fish Weight Prediction** — Complete EDA, One-Hot Encoding, Linear Regression & Random Forest evaluation.
- [x] **California House Price Prediction** — Multicollinearity elimination, 14 engineered features, Multiple Linear Regression, and residual analysis.
- [ ] **Rossmann Store Sales Forecasting** — Feature pipeline, temporal lags, regularized linear models (Ridge/Lasso/ElasticNet), and RMSPE optimization.
- [ ] **Hyperparameter Optimization** — Implement `GridSearchCV` / `RandomizedSearchCV` across ensemble models.
- [ ] **Advanced Gradient Boosting** — Implement XGBoost, LightGBM, and CatBoost benchmarks.
- [ ] **Model Serving & Apps** — Build interactive Streamlit / FastAPI demo interfaces for inference.

---

## 👨‍💻 Author

**Rudrapratap Sarma**  
*Aspiring AI/ML Engineer*

* Continuous learner committed to clean code, reproducible data science, and practical machine learning solutions.
* Feel free to explore the notebooks, open issues, or suggest improvements!
