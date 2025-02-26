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

| Metric | Training Set | Test Set |
| ------ | ------------ | -------- |
| MAE    | 86.033       | 88.073   |
| MAPE   | 42.529%      | 42.697%  |
| SMAPE  | 59.185%      | 60.218%  |
| MIS    | 198.693      | 263.979  |

### LPN Staffing Predictions

| Metric | Training Set | Test Set |
| ------ | ------------ | -------- |
| MAE    | 33.272       | 32.261   |
| MAPE   | 47.894%      | 49.316%  |
| SMAPE  | 67.757%      | 68.615%  |
| MIS    | 179.354      | 164.064  |

### RN Staffing Predictions

| Metric | Training Set | Test Set |
| ------ | ------------ | -------- |
| MAE    | 17.803       | 18.458   |
| MAPE   | 41.438%      | 42.786%  |
| SMAPE  | 58.453%      | 58.308%  |
| MIS    | 35.460       | 36.461   |

## Key Findings

- CNA have much higher values than LPN and RN, and trend is also significantly different
- Based on correlation matrix, CNA and LPN have 0.77 correlation while CNA and RN 0.65 correlated
