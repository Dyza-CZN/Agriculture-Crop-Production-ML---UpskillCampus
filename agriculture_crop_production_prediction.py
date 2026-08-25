"""
Agriculture Crop Production Prediction
Internship Project — UCT / upSkill Campus

Dataset:
    datafile (2).csv

Approach:
    One-year-ahead production forecasting using previous-year
    production, area, yield, and crop category.

The final year (2010-11) is held out as a chronological test set
to avoid random train/test leakage across time.
"""

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_FILE = "datafile (2).csv"

df = pd.read_csv(DATA_FILE)
df.columns = [c.strip() for c in df.columns]
df["Crop"] = df["Crop"].astype(str).str.strip()

for c in df.columns[1:]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

years = ["2006-07", "2007-08", "2008-09", "2009-10", "2010-11"]

rows = []
for _, row in df.iterrows():
    for i in range(1, len(years)):
        previous_year = years[i - 1]
        target_year = years[i]
        rows.append({
            "Crop": row["Crop"],
            "Previous_Production": row[f"Production {previous_year}"],
            "Previous_Area": row[f"Area {previous_year}"],
            "Previous_Yield": row[f"Yield {previous_year}"],
            "Target_Year": target_year,
            "Target_Production": row[f"Production {target_year}"]
        })

data = pd.DataFrame(rows)

train = data[data["Target_Year"] != "2010-11"]
test = data[data["Target_Year"] == "2010-11"]

features = ["Crop", "Previous_Production", "Previous_Area", "Previous_Yield"]

preprocessor = ColumnTransformer([
    ("crop", OneHotEncoder(handle_unknown="ignore"), ["Crop"]),
    ("numeric", StandardScaler(),
     ["Previous_Production", "Previous_Area", "Previous_Yield"])
])

models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Random Forest": RandomForestRegressor(
        n_estimators=300, random_state=42, max_depth=6, min_samples_leaf=2
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        random_state=42, n_estimators=100, max_depth=2, learning_rate=0.05
    )
}

for name, model in models.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    pipeline.fit(train[features], train["Target_Production"])
    predictions = pipeline.predict(test[features])

    mae = mean_absolute_error(test["Target_Production"], predictions)
    rmse = np.sqrt(mean_squared_error(test["Target_Production"], predictions))
    r2 = r2_score(test["Target_Production"], predictions)

    print(f"\n{name}")
    print(f"MAE : {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2  : {r2:.4f}")
