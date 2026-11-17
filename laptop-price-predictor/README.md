# Laptop Price Predictor

An end-to-end Machine Learning project to predict current laptop prices using Linear Regression and regularized variants (Ridge/Lasso).

## Project Structure

```
laptop-price-predictor/
├── data/
│   ├── raw/               # Raw scraped data from Flipkart
│   └── processed/         # Cleaned, encoded, and preprocessed datasets
├── notebooks/             # Jupyter notebooks for EDA and experimentation
├── src/                   # Modular source code
│   ├── scraper/           # Web scraping scripts
│   ├── data/              # Cleaning and feature engineering pipelines
│   ├── models/            # Model training, evaluation, and tuning
│   └── utils/             # Helper functions and plotting utilities
├── models/                # Saved trained model artifacts (.pkl / .joblib)
├── reports/               # Evaluation metrics, residual plots, figures
│   └── figures/
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
```
