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
| MAE    | 30.78       | 31.03    |
| MAPE   | 26.53%      | 25.03%   |
| SMAPE  | 22.12%      | 21.92%   |
| MIS    | 764.03      | 776.85   |

### LPN Staffing Predictions
| Metric | Training Set | Test Set |
|--------|--------------|----------|
| MAE    | 18.01       | 17.58    |
| MAPE   | 52.49%      | 55.57%   |
| SMAPE  | 37.65%      | 38.17%   |
| MIS    | 323.65      | 288.59   |

### RN Staffing Predictions
| Metric | Training Set | Test Set |
|--------|--------------|----------|
| MAE    | 15.65       | 15.86    |
| MAPE   | 69.42%      | 71.54%   |
| SMAPE  | 53.85%      | 51.96%   |
| MIS    | 175.31      | 158.87   |

## Key Findings

1. **Model Stability**
   - Similar metrics between training and test sets indicate good generalization
   - No significant overfitting observed

2. **Staff Type Performance**
   - CNA predictions show highest absolute error but lowest percentage error
   - RN predictions show lowest absolute error but highest percentage error
   - LPN predictions fall between CNA and RN for most metrics

3. **Prediction Intervals**
   - MIS values decrease from CNA to RN staffing
   - Suggests more precise confidence intervals for RN staffing predictions

## Running the Evaluation

To reproduce these results:

1. Ensure you have the required Python packages installed
2. Place the "Phase 1 Training Dataset.xlsx" in the same directory as the script
3. Run the evaluation script:
   ```bash
   python nh_staffing_predictor.py
   ```

The script will automatically:
- Load and split the data (80/20)
- Train the models
- Generate predictions
- Calculate and display all evaluation metrics
