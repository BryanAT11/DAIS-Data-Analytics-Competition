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
| MAE    | 44.392       | 45.279         |
| MAPE   | 20.228%      | 20.005%        |
| SMAPE  | 23.651%      | 23.953%        |
| MIS    | 146.848      | 151.804        |

#### LPN Staffing Evaluation

| Metric    | Training Set | Validation Set |
| --------- | ------------ | -------------- |
| **MAE**   | 24.993       | 24.117         |
| **MAPE**  | 35.515%      | 36.972%        |
| **SMAPE** | 45.016%      | 46.305%        |
| **MIS**   | 55.057       | 56.861         |

#### RN Staffing Evaluation

| Metric    | Training Set | Validation Set |
| --------- | ------------ | -------------- |
| **MAE**   | 13.582       | 14.322         |
| **MAPE**  | 33.259%      | 35.808%        |
| **SMAPE** | 45.125%      | 46.022%        |
| **MIS**   | 37.994       | 37.890         |

## Validation: Test data validation

### Result

## Key Findings

- CNA have much higher values than LPN and RN, and trend is also significantly different
- Based on correlation matrix, CNA and LPN have 0.77 correlation while CNA and RN 0.65 correlated
