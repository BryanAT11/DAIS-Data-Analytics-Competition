# Nursing Home Staffing Predictor Evaluation Results

## Data Split Configuration
- Training Set: 80% of the data
- Test Set: 20% of the data
- Split Method: Temporal split (maintaining time series order)
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

3. **SMAPE (Symmetric Mean Absolute Percentage Error)**
   - Symmetric version of MAPE that treats over/under predictions equally
   - Formula: 100 * mean(2 * |predicted - actual| / (|actual| + |predicted|))
   - Lower values indicate better performance
   - Unit: Percentage

4. **MIS (Mean Interval Score)**
   - Evaluates prediction intervals (95% confidence level)
   - Penalizes predictions outside confidence interval
   - Uses median for robustness
   - Lower values indicate better performance
   - Unit: Composite score

## Results

### CNA Staffing Predictions
| Metric | Training Set | Test Set |
|--------|--------------|----------|
| MAE    | 30.47        | 30.92    |
| MAPE   | 25.97%       | 24.66%   |
| SMAPE  | 21.88%       | 21.86%   |
| MIS    | 91.59        | 90.61   |

### LPN Staffing Predictions
| Metric | Training Set | Test Set |
|--------|--------------|----------|
| MAE    | 17.90        | 17.57    |
| MAPE   | 49.91%       | 52.40%   |
| SMAPE  | 37.51%       | 38.14%   |
| MIS    | 61.68        | 51.73    |

### RN Staffing Predictions
| Metric | Training Set | Test Set |
|--------|--------------|----------|
| MAE    | 15.38        | 15.81    |
| MAPE   | 66.83%       | 68.91%   |
| SMAPE  | 53.56%       | 52.10%   |
| MIS    | 52.41        | 53.30    |