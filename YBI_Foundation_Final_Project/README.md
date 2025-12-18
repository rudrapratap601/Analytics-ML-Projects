# 🐟 Fish Weight Prediction — Machine Learning Project

## 📌 Overview
This project predicts the **weight of a fish** based on its species and body measurements such as length, height, and width.  
The dataset contains **159 samples** from different fish species, and regression models were used to estimate fish weight accurately.

---

## 📂 Dataset Description
The dataset includes the following features:

- **Species** (Bream, Roach, Pike, Perch, etc.)
- **Weight** (Target variable)
- **Length1, Length2, Length3** (Multiple length measurements)
- **Height**
- **Width**

✔ No missing values  
✔ Clean and ready for modeling

---

## 🔍 Exploratory Data Analysis (EDA)

Key insights from EDA:

- Length measurements have **very strong correlations** with weight (≈0.90+).
- Width also shows strong correlation and becomes the **most important feature** during model learning.
- Weight distribution is **right-skewed** due to naturally heavy fish like Pike and Bream.
- Species significantly affects weight, shown clearly in boxplots.
- Outliers were kept because they represent **real biological variation**.

---

## 🧹 Preprocessing Steps
- One-Hot Encoding applied to the **Species** column  
- Removed the **Category** column (not useful for prediction)  
- Train–test split at **80–20 ratio**  
- Feature scaling applied for **Linear Regression**

---

## 🤖 Models Used
- **Linear Regression** (baseline model)
- **Random Forest Regressor** (final selected model)

---

## 📈 Model Performance

| Model                   | MAE     | RMSE    | R²     |
|------------------------|---------|---------|--------|
| Linear Regression       | 65.30   | 83.71   | 0.951  |
| **Random Forest Regressor** | **44.49** | **65.90** | **0.969** |

---

## 🌟 Feature Importance

Top predictors identified by the Random Forest model:

- **Width (0.7138)** – Most important
- **Length3**
- **Length1**
- **Length2**
- Height → smaller contribution  
- Species dummy variables → smallest but meaningful contributions

---

## 📌 Project Conclusion
This project successfully built a regression model to predict fish weight using physical measurements and species information.  
EDA revealed strong linear relationships between the length features and weight, while **Width** emerged as a crucial indicator of body mass.

After testing multiple models, the **Random Forest Regressor** performed the best with:

- **MAE:** 44.49  
- **RMSE:** 65.90  
- **R² Score:** 0.969  

The model identifies Width as the most important predictor, followed by the three length measurements. This aligns with biological intuition: wider fish generally have more body mass even if lengths remain similar.

The final model demonstrates **strong predictive power**, making it suitable for real applications such as **fish farming, pricing, and biological studies**.

---

## 🔮 Next Steps
- Apply **hyperparameter tuning** (GridSearchCV)
- Try boosting models like **XGBoost** or **CatBoost**
- Deploy the model using **Streamlit** or **FastAPI**
- Build a **REST API** for fish weight prediction

---

