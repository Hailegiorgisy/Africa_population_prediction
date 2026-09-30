import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from synthetic_generator import generate_mock_demographic_data
from models import DemographicEnsembleForecaster

def run_reproducible_pipeline():
    parquet_path = "data/synthetic_population.parquet"
    print("Step 1: Generating/Verifying synthetic demographic tensors...")
    df = generate_mock_demographic_data(output_path=parquet_path)

    print("Step 2: Training Ensemble Demographic Regressors...")
    forecaster = DemographicEnsembleForecaster()
    X, y, df_clean = forecaster.prepare_features(df)
    
    # Train / Test split by year: train <= 2018, test > 2018
    train_mask = df_clean["year"] <= 2018
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[~train_mask], y[~train_mask]

    forecaster.train(X_train, y_train)
    y_pred = forecaster.predict(X_test)
    
    metrics = forecaster.evaluate(y_test, y_pred)
    print("Step 3: Verification Metrics:")
    for k, v in metrics.items():
        print(f"  - {k}: {v}")

    assert metrics["Percentage_Bias"] < 5.0, "Bias exceeded allowable threshold."
    print("Reproducibility pipeline completed successfully.")

if __name__ == "__main__":
    run_reproducible_pipeline()
