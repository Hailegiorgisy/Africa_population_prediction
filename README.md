# Africa Population Prediction: Ensemble Modeling for Demographic Forecasting

This repository provides demographic forecasting pipelines for African nations, bridging longitudinal data from the United Nations World Population Prospects (UN WPP) and World Health Organization (WHO) with machine learning architectures.

## Modeling Methodology
- **Time-Series Baseline & Lags**: Captures temporal auto-regressive momentum and fertility transition.
- **Ensemble Regressors**: Random Forest and Gradient Boosting architectures to model multi-variate non-linear interactions across fertility, mortality, and urbanization indicators.
- **Evaluation Criteria**: Root Mean Square Error (RMSE), Mean Absolute Error (MAE), and Percentage Bias.

## Offline CI Data Pipeline
To ensure continuous integration and unit testing run reliably without external API dependencies:
- `src/synthetic_generator.py`: Generates standardized demographic tensors matching UN WPP empirical distributions.
- Local cache format: Highly optimized `.parquet` column storage.

## Reproducibility
Execute the reproducible pipeline:
```bash
python reproducibility/run_pipeline.py
```
