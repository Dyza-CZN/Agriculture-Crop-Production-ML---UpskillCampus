# Prediction of Agriculture Crop Production in India

## Internship Project

This project develops a simple machine-learning approach for one-year-ahead
prediction of agricultural crop production using the supplied internship dataset.

## Dataset

The selected dataset contains 55 crop categories and production, area, and yield
values for 2006-07 through 2010-11.

The dataset is supplied separately and is not redistributed in this repository.

## Method

The project uses:
- Previous-year production
- Previous-year cultivated area
- Previous-year yield
- Crop category

The target is production in the following year.

A chronological split is used:
- Training: targets from 2007-08 through 2009-10
- Testing: target year 2010-11

This avoids randomly mixing observations from different years.

## Models

The following regression models are compared:
1. Linear Regression
2. Ridge Regression
3. Random Forest Regression
4. Gradient Boosting Regression

## Evaluation

Metrics:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R-squared (R²)

The current run identifies Linear Regression as the best-performing model on the
held-out 2010-11 data, based on RMSE.

## Files

- `agriculture_crop_production_prediction.py` — complete implementation
- `model_results.csv` — model performance
- `prepared_forecasting_data.csv` — prepared modeling data
- `model_comparison.png` — RMSE comparison
- `actual_vs_predicted.png` — actual vs predicted values
