import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict
import warnings
warnings.filterwarnings('ignore')

class PredictionEvaluator:
    def __init__(self, alpha=0.05):
        self.alpha = alpha
        
    def calculate_mae(self, y_true, y_pred):
        """Calculate Mean Absolute Error"""
        return np.mean(np.abs(y_true - y_pred))
    
    def calculate_mape(self, y_true, y_pred):
        """Calculate Mean Absolute Percentage Error"""
        # Filter out zero values to avoid division by zero
        mask = y_true != 0
        if not np.any(mask):
            return np.nan
        return 100 * np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))
    
    def calculate_mis(self, y_true, y_pred, lower_bound, upper_bound):
        """Calculate Mean Interval Score"""
        alpha = self.alpha
        n = len(y_true)
        
        # Initialize components of MIS
        coverage_penalty = np.zeros(n)
        width_penalty = np.zeros(n)
        
        # Calculate penalties
        for i in range(n):
            interval_width = upper_bound[i] - lower_bound[i]
            
            if y_true[i] < lower_bound[i]:
                coverage_penalty[i] = 2/alpha * (lower_bound[i] - y_true[i])
            elif y_true[i] > upper_bound[i]:
                coverage_penalty[i] = 2/alpha * (y_true[i] - upper_bound[i])
                
            width_penalty[i] = interval_width
            
        mis = np.mean(width_penalty + coverage_penalty)
        return mis
    
    def evaluate_predictions(self, y_true, y_pred, lower_bound, upper_bound):
        """Evaluate predictions using all metrics"""
        metrics = {
            'MAE': self.calculate_mae(y_true, y_pred),
            'MAPE': self.calculate_mape(y_true, y_pred),
            'MIS': self.calculate_mis(y_true, y_pred, lower_bound, upper_bound)
        }
        return metrics

class NHStaffingPredictor:
    def __init__(self):
        self.training_data = None
        self.groups = []
        self.evaluator = PredictionEvaluator()
        self.models = {}
        
    def load_training_data(self, file_path):
        """Load training data from Excel file with multiple sheets"""
        xl = pd.ExcelFile(file_path)
        self.training_data = {}
        self.groups = xl.sheet_names
        
        for sheet in self.groups:
            df = pd.read_excel(file_path, sheet_name=sheet)
            
            # Skip the first row which contains staff type labels
            df = df.iloc[1:]
            
            # Convert the date column to datetime
            df['Date'] = pd.to_datetime(df.iloc[:, 0])
            
            # Sort by date to ensure temporal order
            df = df.sort_values('Date')
            
            # Calculate split point (80% train, 20% test)
            split_idx = int(len(df) * 0.8)
            
            # Process data for each NH
            train_data = []
            test_data = []
            
            for i in range(5):  # 5 NHs per group
                start_col = i * 4  # Each NH has 4 columns (NH No., CNA, LPN, RN)
                
                if start_col + 3 >= len(df.columns):
                    break
                    
                # Extract data for each staff type
                cna_data = pd.to_numeric(df.iloc[:, start_col + 1], errors='coerce')
                lpn_data = pd.to_numeric(df.iloc[:, start_col + 2], errors='coerce')
                rn_data = pd.to_numeric(df.iloc[:, start_col + 3], errors='coerce')
                
                # Create features
                dates = df['Date']
                features = pd.DataFrame({
                    'dayofweek': dates.dt.dayofweek,
                    'month': dates.dt.month,
                    'day': dates.dt.day,
                    'is_weekend': dates.dt.dayofweek.isin([5, 6]).astype(int)
                })
                
                # Split into train and test
                train_staff_data = {
                    'NH_No': i + 1,
                    'CNA': cna_data[:split_idx].dropna().tolist(),
                    'LPN': lpn_data[:split_idx].dropna().tolist(),
                    'RN': rn_data[:split_idx].dropna().tolist(),
                    'features': features[:split_idx]
                }
                
                test_staff_data = {
                    'NH_No': i + 1,
                    'CNA': cna_data[split_idx:].dropna().tolist(),
                    'LPN': lpn_data[split_idx:].dropna().tolist(),
                    'RN': rn_data[split_idx:].dropna().tolist(),
                    'features': features[split_idx:]
                }
                
                train_data.append(train_staff_data)
                test_data.append(test_staff_data)
            
            self.training_data[sheet] = {
                'train': train_data,
                'test': test_data
            }
    
    def train_models(self, group_data):
        """Train Random Forest models for each staff type"""
        models = {}
        predictions = {}
        
        for staff_type in ['CNA', 'LPN', 'RN']:
            # Combine data from all NHs in the group
            X = pd.concat([nh['features'] for nh in group_data])
            y = []
            for nh in group_data:
                y.extend(nh[staff_type])
            y = np.array(y)
            
            # Train Random Forest model
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)
            
            # Get predictions and prediction intervals
            y_pred = model.predict(X)
            
            # Calculate prediction intervals using quantile regression
            lower_quantile = []
            upper_quantile = []
            
            # Use out-of-bag predictions for uncertainty estimation
            for estimator in model.estimators_:
                pred = estimator.predict(X)
                lower_quantile.append(pred)
                upper_quantile.append(pred)
            
            lower_bound = np.percentile(lower_quantile, 2.5, axis=0)
            upper_bound = np.percentile(upper_quantile, 97.5, axis=0)
            
            models[staff_type] = model
            predictions[staff_type] = {
                'point_prediction': np.mean(y_pred),
                'lower_bound': max(0, np.mean(lower_bound)),
                'upper_bound': np.mean(upper_bound)
            }
        
        return models, predictions
    
    def generate_predictions(self, group_data):
        """Generate predictions for a single NH from the group"""
        # Train models for this group
        _, predictions = self.train_models(group_data)
        return predictions
    
    def create_prediction_template(self, output_file):
        """Create predictions for all groups and save to template"""
        # Create date range from 4/1/2024 to 6/30/2024
        start_date = datetime(2024, 4, 1)
        end_date = datetime(2024, 6, 30)
        dates = []
        current_date = start_date
        while current_date <= end_date:
            dates.append(current_date)
            current_date += timedelta(days=1)
        
        # Create features for prediction dates
        pred_features = pd.DataFrame({
            'dayofweek': [d.weekday() for d in dates],
            'month': [d.month for d in dates],
            'day': [d.day for d in dates],
            'is_weekend': [1 if d.weekday() in [5, 6] else 0 for d in dates]
        })
        
        # Create Excel writer
        writer = pd.ExcelWriter(output_file, engine='openpyxl')
        
        # Create predictions for each group
        for group in self.groups:
            # Train models and get initial predictions
            models, base_predictions = self.train_models(self.training_data[group]['train'])
            
            # Create data for the sheet
            data = []
            for i, date in enumerate(dates):
                # Get predictions for this date
                row_predictions = {}
                for staff_type in ['CNA', 'LPN', 'RN']:
                    model = models[staff_type]
                    pred = model.predict(pred_features.iloc[i:i+1])[0]
                    
                    # Get prediction intervals
                    preds = []
                    for estimator in model.estimators_:
                        preds.append(estimator.predict(pred_features.iloc[i:i+1])[0])
                    
                    lower = max(0, np.percentile(preds, 2.5))
                    upper = np.percentile(preds, 97.5)
                    
                    row_predictions[f'{staff_type} Point Prediction'] = round(pred, 2)
                    row_predictions[f'{staff_type} Lower Bound'] = round(lower, 2)
                    row_predictions[f'{staff_type} Upper Bound'] = round(upper, 2)
                
                row = {'Date': date.strftime('%-m/%-d/%y')}
                row.update(row_predictions)
                data.append(row)
            
            # Create DataFrame and save to sheet
            df = pd.DataFrame(data)
            df.to_excel(writer, sheet_name=group, index=False)
            
        writer.close()
        
    def evaluate_group_predictions(self, group, actual_data):
        """Evaluate predictions for a specific group against actual data"""
        predictions = self.generate_predictions(self.training_data[group]['train'])
        metrics = {}
        
        for staff_type in ['CNA', 'LPN', 'RN']:
            y_true = np.array(actual_data[staff_type])
            y_pred = np.full_like(y_true, predictions[staff_type]['point_prediction'])
            lower_bound = np.full_like(y_true, predictions[staff_type]['lower_bound'])
            upper_bound = np.full_like(y_true, predictions[staff_type]['upper_bound'])
            
            metrics[staff_type] = self.evaluator.evaluate_predictions(
                y_true, y_pred, lower_bound, upper_bound
            )
            
        return metrics

def main():
    predictor = NHStaffingPredictor()
    
    # Load training data
    predictor.load_training_data('Phase 1 Training Dataset.xlsx')
    
    # Generate predictions and save to template
    predictor.create_prediction_template('Phase 1 Predictions_Output.xlsx')
    
    # Evaluate predictions using train-test split
    print("\nEvaluation Metrics using Train-Test Split:")
    print("=========================================")
    
    train_metrics = {
        'CNA': {'MAE': [], 'MAPE': [], 'MIS': []},
        'LPN': {'MAE': [], 'MAPE': [], 'MIS': []},
        'RN': {'MAE': [], 'MAPE': [], 'MIS': []}
    }
    
    test_metrics = {
        'CNA': {'MAE': [], 'MAPE': [], 'MIS': []},
        'LPN': {'MAE': [], 'MAPE': [], 'MIS': []},
        'RN': {'MAE': [], 'MAPE': [], 'MIS': []}
    }
    
    for group in predictor.groups:
        # Train models on training data
        train_predictions = predictor.generate_predictions(predictor.training_data[group]['train'])
        
        # Get predictions for test data
        test_predictions = predictor.generate_predictions(predictor.training_data[group]['test'])
        
        # Evaluate on training data
        for nh in predictor.training_data[group]['train']:
            for staff_type in ['CNA', 'LPN', 'RN']:
                y_true = np.array(nh[staff_type])
                y_pred = np.full_like(y_true, train_predictions[staff_type]['point_prediction'])
                lower_bound = np.full_like(y_true, train_predictions[staff_type]['lower_bound'])
                upper_bound = np.full_like(y_true, train_predictions[staff_type]['upper_bound'])
                
                metrics = predictor.evaluator.evaluate_predictions(
                    y_true, y_pred, lower_bound, upper_bound
                )
                
                for metric_name, value in metrics.items():
                    train_metrics[staff_type][metric_name].append(value)
        
        # Evaluate on test data
        for nh in predictor.training_data[group]['test']:
            for staff_type in ['CNA', 'LPN', 'RN']:
                y_true = np.array(nh[staff_type])
                y_pred = np.full_like(y_true, test_predictions[staff_type]['point_prediction'])
                lower_bound = np.full_like(y_true, test_predictions[staff_type]['lower_bound'])
                upper_bound = np.full_like(y_true, test_predictions[staff_type]['upper_bound'])
                
                metrics = predictor.evaluator.evaluate_predictions(
                    y_true, y_pred, lower_bound, upper_bound
                )
                
                for metric_name, value in metrics.items():
                    test_metrics[staff_type][metric_name].append(value)
    
    # Print training metrics
    print("\nTraining Set Metrics:")
    print("--------------------")
    for staff_type in ['CNA', 'LPN', 'RN']:
        print(f"\n{staff_type} Metrics:")
        print("-" * 20)
        for metric_name in ['MAE', 'MAPE', 'MIS']:
            values = train_metrics[staff_type][metric_name]
            avg_value = np.mean(values)
            print(f"{metric_name}: {avg_value:.2f}")
    
    # Print test metrics
    print("\nTest Set Metrics:")
    print("----------------")
    for staff_type in ['CNA', 'LPN', 'RN']:
        print(f"\n{staff_type} Metrics:")
        print("-" * 20)
        for metric_name in ['MAE', 'MAPE', 'MIS']:
            values = test_metrics[staff_type][metric_name]
            avg_value = np.mean(values)
            print(f"{metric_name}: {avg_value:.2f}")
    
if __name__ == "__main__":
    main()
