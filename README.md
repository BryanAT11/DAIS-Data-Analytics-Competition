# Nursing Home Staffing Predictor Evaluation Results

- Data Source: Phase 1 Training Dataset.xlsx
- Notebook used: nh_staffing_notebook_ver2

## Evaluation Metrics

The model performance is evaluated using four key metrics: MAE, MAPE, SMAPE, MIS

## Model Used: XGBoost and quantile regression for CI computation

## Validation: Time-based Validation on 20% training data

### Results

#### CNA Staffing Predictions

| Metric | Training Set | Validation Set |
| ------ | ------------ | -------------- |
| MAE    | 13.523       | 13.343         |
| MAPE   | 21.539%      | 18.247%        |
| SMAPE  | 13.456%      | 12.972%        |
| Coverage | 47.7%      | 45.8%          |
| Interval Width | 82.945 | 81.765       |
| MIS    | 440.277     | 436.794         |

#### LPN Staffing Evaluation

| Metric    | Training Set | Validation Set |
| --------- | ------------ | -------------- |
| MAE       | 11.432       | 12.184         |
| MAPE      | 53.096%      | 58.715%        |
| SMAPE     | 30.499%      | 32.804%        |
| Coverage  | 37.9%        | 36.2%          |
| Interval Width | 32.332  | 31.371         |
| MIS       | 372.081      | 377.059        |

#### RN Staffing Evaluation

| Metric    | Training Set | Validation Set |
| --------- | ------------ | -------------- |
| MAE       | 7.244        | 7.586          |
| MAPE      | 32.995%      | 34.017%        |
| SMAPE     | 33.429%      | 32.607%        |
| Coverage  | 48.4%        | 47.7%          |
| Interval Width | 28.361  | 29.128         |
| MIS       | 179.004      | 189.059        |

## Validation: Test data validation

### Full Results
Used Phase 1 Predictions Output_high.xlxs
| Metric | Entire File  |
|--------|--------------|
| MAE    | 78.43        |
| MAPE   | 208.277%      |
| SMAPE  | 87.06%       |
| MIS    | 707.80      |

## Key Findings

- CNA have much higher values than LPN and RN, and trend is also significantly different
- Based on correlation matrix, CNA and LPN have 0.77 correlation while CNA and RN 0.65 correlated
