# 🏡 California House Price Prediction — Machine Learning Project

## 📌 Overview
This project predicts median house values for California districts based on demographic, economic, and geographic features from the 1990 U.S. Census. Using Python, Scikit-Learn, and statistical regression techniques, this project demonstrates end-to-end data preprocessing, multicollinearity mitigation, domain-specific feature engineering, polynomial feature expansion, model training, and residual analysis.

---

## 📂 Dataset Description
The dataset is loaded directly via Scikit-Learn's `fetch_california_housing()`, containing **20,640 samples** across California block groups:

| Feature | Description |
| :--- | :--- |
| **MedInc** | Median income in block group (in tens of thousands USD) |
| **HouseAge** | Median house age in block group (years) |
| **AveRooms** | Average number of rooms per household |
| **AveBedrms** | Average number of bedrooms per household |
| **Population** | Block group population |
| **AveOccup** | Average number of household members |
| **Latitude** | Block group latitude |
| **Longitude** | Block group longitude |
| **Price** *(Target)* | Median house value for California districts (hundreds of thousands USD, \$100,000s) |

---

## 🔍 Exploratory Data Analysis & Preprocessing

### 1. Data Cleaning
- **Missing Values:** Zero missing values across all features.
- **Duplicates:** Zero duplicate entries detected.

### 2. Multicollinearity Mitigation
Correlation analysis revealed strong collinearity:
- `AveBedrms` and `AveRooms` correlation: **0.8476**
- `Latitude` and `Longitude` correlation: **-0.9247**

To prevent instability and inflated variance in linear regression coefficients, **`AveBedrms`** and **`Longitude`** were removed.

### 3. Feature Engineering
Seven domain-informed features and non-linear terms were created:
1. **`RoomsPerPerson`**: $\frac{\text{AveRooms}}{\text{AveOccup}}$ (housing space and quality index)
2. **`PopulationDensity`**: $\frac{\text{Population}}{\text{AveOccup}}$ (density proxy per block)
3. **`IncomeAge`**: $\text{MedInc} \times \text{HouseAge}$ (interaction of neighborhood maturity and income)
4. **`IncomeLatitude`**: $\text{MedInc} \times \text{Latitude}$ (geographic wealth distribution interaction)
5. **`MedInc_squared`** & **`MedInc_cubed`**: Polynomial terms ($\text{MedInc}^2, \text{MedInc}^3$) capturing non-linear elasticity between income and property value.
6. **`Log_Population`** & **`Log_AveOccup`**: $\log(1 + x)$ transforms to compress extreme right-skewed tail distributions.

### 4. Train-Test Split & Scaling
- **Split:** 70% Training (14,448 samples), 30% Testing (6,192 samples) with `random_state=42`.
- **Scaling:** `StandardScaler` fitted strictly on the training set and applied to the test set to eliminate data leakage.

---

## 🤖 Model & Performance

A Multiple Linear Regression model was fitted on the 14 scaled features.

### Performance Metrics on Test Set

| Metric | Score | Interpretation |
| :--- | :--- | :--- |
| **Mean Absolute Error (MAE)** | **0.5265** | Average prediction error is ~\$52,650 |
| **Mean Squared Error (MSE)** | **0.5094** | Low overall squared penalty |
| **Root Mean Squared Error (RMSE)** | **0.7137** | Typical error standard deviation is ~\$71,370 |
| **$R^2$ Score** | **0.6119** | Explains **61.19%** of variance in house prices |
| **Adjusted $R^2$ Score** | **0.6111** | High variance explained accounting for feature count |

---

## 🌟 Feature Importance & Coefficients

The top drivers sorted by absolute coefficient magnitude:

| Feature | Coefficient | Impact |
| :--- | :--- | :--- |
| **MedInc_squared** | `+1.9139` | Strongest positive non-linear income driver |
| **MedInc_cubed** | `-1.2559` | Stabilizes extreme income growth effects |
| **MedInc** | `-0.8962` | Linear component of income model |
| **IncomeLatitude** | `+0.8636` | Captures premium housing locations in higher-income latitudes |
| **AveRooms** | `-0.2571` | Adjusts baseline room counts |
| **RoomsPerPerson** | `+0.2479` | Positive contribution from spacious living conditions |
| **Latitude** | `-0.2246` | Regional baseline price adjustment |
| **PopulationDensity** | `+0.1868` | Positive price pressure in dense urban clusters |
| **HouseAge** | `+0.1718` | Positive value appreciation in mature neighborhoods |

---

## 📊 Residual Diagnostics
- **Mean of Residuals:** `0.0042` (near-zero mean confirms unbiased predictions).
- **Residual Distribution:** Approximately Gaussian bell curve centered at zero with minimal systematic drift.
- **Residual Scatter:** Homoscedasticity holds across median value ranges, with predictable boundary effects near the dataset ceiling (\$5.00).

---

## 🔮 Next Steps
- Implement regularized regression models (**Ridge**, **Lasso**, **ElasticNet**) to control polynomial multicollinearity.
- Evaluate non-linear ensemble models (**Random Forest**, **XGBoost**, **LightGBM**).
- Incorporate spatial clustering (e.g., K-Means on coordinates) as categorical location features.
