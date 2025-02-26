# Nursing Home Staffing Predictor Evaluation Results

- Data Source: Phase 1 Training Dataset.xlsx

## Evaluation Metrics

The model performance is evaluated using four key metrics:

1. **MAE (Mean Absolute Error)**

   - Measures average absolute difference between predicted and actual values
   - Lower values indicate better performance
   - Unit: Number of staff members

2. **MAPE (Mean Absolute Percentage Error)**

   - Percentage error relative to actual values
   - Uses threshold of 1.0 to avoid division by small numbers
   - Lower values indicate better performance
   - Unit: Percentage

3. **MIS (Mean Interval Score)**
   - Evaluates prediction intervals (95% confidence level)
   - Penalizes predictions outside confidence interval
   - Uses median for robustness
   - Lower values indicate better performance
   - Unit: Composite score

## Model Used: XGBoost and quantile regression for CI computation

## Validation: Time-based Validation on 20% training data

## Alternative Validation to consider: Group K-Fold Cross-Validation (Leave-One-NH-Out per Group)

- Train on 4 NHs per group.
- Validate on the remaining 1 NH per group.
- Rotate until every NH has been validated once.

## Results

### CNA Staffing Predictions

| Metric | Training Set | Validation Set |
| ------ | ------------ | -------------- |
| MAE    | 86.033       | 88.073         |
| MAPE   | 42.529%      | 42.697%        |
| SMAPE  | 59.185%      | 60.218%        |
| MIS    | 198.693      | 263.979        |

### LPN Staffing Evaluation

| Metric    | Training Set | Validation Set |
| --------- | ------------ | -------------- |
| **MAE**   | 24.993       | 24.117         |
| **MAPE**  | 35.515%      | 36.972%        |
| **SMAPE** | 45.016%      | 46.305%        |
| **MIS**   | 55.057       | 56.861         |

### RN Staffing Evaluation

| Metric    | Training Set | Validation Set |
| --------- | ------------ | -------------- |
| **MAE**   | 13.582       | 14.322         |
| **MAPE**  | 33.259%      | 35.808%        |
| **SMAPE** | 45.125%      | 46.022%        |
| **MIS**   | 37.994       | 37.890         |

## Key Findings

- CNA have much higher values than LPN and RN, and trend is also significantly different
- Based on correlation matrix, CNA and LPN have 0.77 correlation while CNA and RN 0.65 correlated
