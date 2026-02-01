# baseline_random_forest.py

import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import joblib

from pipeline.pipeline import MarketDataPipeline

pipeline = MarketDataPipeline(
    iterations=15,
    logging_level=logging.INFO
)

labeled_df, splits = pipeline.run()

rf_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("rf", RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42
    ))
])

all_metrics = []

for i, (train, test) in enumerate(splits):
    train = train.dropna(subset=['label'])
    test = test.dropna(subset=['label'])
    if test.empty or train.empty:
        print(f"Skipping split {i+1} because train or test is empty")
        continue
    X_train, y_train = train.drop(columns=['label']), train['label']
    X_test, y_test = test.drop(columns=['label']), test['label']

    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)

    rf_pipeline.fit(X_train, y_train)
    y_pred = rf_pipeline.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    all_metrics.append((mse, r2))

    print(f"Split {i+1} | MSE: {mse:.4f} | R²: {r2:.4f}")

mean_mse = sum(m[0] for m in all_metrics) / len(all_metrics)
mean_r2 = sum(m[1] for m in all_metrics) / len(all_metrics)

print(f"\nOverall metrics across all splits: MSE={mean_mse:.4f}, R²={mean_r2:.4f}")
labeled_df = labeled_df.dropna(subset=['label'])

X_final, y_final = labeled_df.drop(columns=['label']), labeled_df['label']
rf_pipeline.fit(X_final, y_final)
print("Final Random Forest model trained on all available data.")

joblib.dump(rf_pipeline, "./random_forest_signal.pkl")
print("Final model saved to 'models/random_forest_signal.pkl'.")
