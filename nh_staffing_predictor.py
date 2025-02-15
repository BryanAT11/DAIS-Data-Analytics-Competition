import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.model_selection import TimeSeriesSplit, KFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
warnings.filterwarnings('ignore')

class PredictionEvaluator:
    def __init__(self, alpha=0.05):
        self.alpha = alpha
        
    def calculate_mae(self, y_true, y_pred):
        """Calculate Mean Absolute Error"""
        return np.mean(np.abs(y_true - y_pred))
    
    def calculate_mape(self, y_true, y_pred):
        """Calculate Mean Absolute Percentage Error with handling for small values"""
        # Use a threshold to avoid division by very small numbers
        threshold = 1.0
        mask = y_true > threshold
        if not np.any(mask):
            return np.nan
        return 100 * np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))
    
    def calculate_smape(self, y_true, y_pred):
        """Calculate Symmetric Mean Absolute Percentage Error"""
        return 100 * np.mean(2.0 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))
    
    def calculate_mis(self, y_true, y_pred, lower_bound, upper_bound):
        """Calculate Mean Interval Score with robust handling of outliers"""
        alpha = self.alpha
        n = len(y_true)
        
        # Initialize components of MIS
        coverage_penalty = np.zeros(n)
        width_penalty = np.zeros(n)
        
        # Calculate penalties with outlier-robust handling
        for i in range(n):
            interval_width = upper_bound[i] - lower_bound[i]
            
            if y_true[i] < lower_bound[i]:
                coverage_penalty[i] = 2/alpha * (lower_bound[i] - y_true[i])
            elif y_true[i] > upper_bound[i]:
                coverage_penalty[i] = 2/alpha * (y_true[i] - upper_bound[i])
                
            width_penalty[i] = interval_width
            
        # Use median instead of mean for more robustness
        mis = np.median(width_penalty + coverage_penalty)
        return mis
    
    def evaluate_predictions(self, y_true, y_pred, lower_bound, upper_bound):
        """Evaluate predictions using all metrics"""
        metrics = {
            'MAE': self.calculate_mae(y_true, y_pred),
            'MAPE': self.calculate_mape(y_true, y_pred),
            'SMAPE': self.calculate_smape(y_true, y_pred),
            'MIS': self.calculate_mis(y_true, y_pred, lower_bound, upper_bound)
        }
        return metrics

class NHStaffingPredictor:
    def __init__(self):
        self.training_data = None
        self.groups = []
        self.evaluator = PredictionEvaluator()
        self.models = {}
        self.scalers = {}
        
    def extract_facility_characteristics(self, group_name):
        """Extract facility characteristics from group name"""
        # Group names follow pattern: Group X where X is 1-20
        # Groups are divided by state (IN/FL), bed capacity (5 levels), and quality rating (2 levels)
        group_num = int(group_name.split()[-1])
        
        # Determine characteristics based on group number
        state = 'IN' if group_num <= 10 else 'FL'
        bed_capacity = (group_num - 1) % 5 + 1  # 1-5 levels
        quality_rating = 1 if (group_num - 1) % 10 < 5 else 2  # 2 levels
        
        return {
            'state_IN': 1 if state == 'IN' else 0,
            'state_FL': 1 if state == 'FL' else 0,
            'bed_capacity': bed_capacity,
            'quality_rating': quality_rating
        }
    
    def create_features(self, dates, group_name, staff_data):
        """Create enhanced feature set including facility characteristics and time-based features"""
        # Basic time features
        features = pd.DataFrame({
            'dayofweek': dates.dt.dayofweek,
            'month': dates.dt.month,
            'day': dates.dt.day,
            'is_weekend': dates.dt.dayofweek.isin([5, 6]).astype(int)
        })
        
        # Add facility characteristics
        facility_chars = self.extract_facility_characteristics(group_name)
        for key, value in facility_chars.items():
            features[key] = value
            
        # Add rolling statistics (7-day window)
        if len(staff_data) >= 7:
            features['rolling_mean_7d'] = pd.Series(staff_data).rolling(7, min_periods=1).mean()
            features['rolling_std_7d'] = pd.Series(staff_data).rolling(7, min_periods=1).std()
        else:
            features['rolling_mean_7d'] = np.mean(staff_data)
            features['rolling_std_7d'] = np.std(staff_data)
            
        return features
    
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
                
                # Create enhanced features for train and test sets
                train_features = {}
                test_features = {}
                for staff_type, data in [('CNA', cna_data), ('LPN', lpn_data), ('RN', rn_data)]:
                    train_features[staff_type] = self.create_features(
                        df['Date'][:split_idx],
                        sheet,
                        data[:split_idx]
                    )
                    test_features[staff_type] = self.create_features(
                        df['Date'][split_idx:],
                        sheet,
                        data[split_idx:]
                    )
                
                train_staff_data = {
                    'NH_No': i + 1,
                    'CNA': cna_data[:split_idx].dropna().tolist(),
                    'LPN': lpn_data[:split_idx].dropna().tolist(),
                    'RN': rn_data[:split_idx].dropna().tolist(),
                    'features': train_features
                }
                
                test_staff_data = {
                    'NH_No': i + 1,
                    'CNA': cna_data[split_idx:].dropna().tolist(),
                    'LPN': lpn_data[split_idx:].dropna().tolist(),
                    'RN': rn_data[split_idx:].dropna().tolist(),
                    'features': test_features
                }
                
                train_data.append(train_staff_data)
                test_data.append(test_staff_data)
            
            self.training_data[sheet] = {
                'train': train_data,
                'test': test_data
            }
    
    def train_models(self, group_data, group_name):
        """Train Random Forest models for each staff type using cross-validation"""
        models = {}
        predictions = {}
        
        for staff_type in ['CNA', 'LPN', 'RN']:
            # Combine data from all NHs in the group
            X = pd.concat([nh['features'][staff_type] for nh in group_data])
            y = []
            for nh in group_data:
                y.extend(nh[staff_type])
            y = np.array(y)
            
            if staff_type == 'RN':
                # Use hybrid model for RN predictions
                model = HybridRNPredictor()
                model.fit(X, y)
                
                # Get predictions and intervals
                y_pred = model.predict(X)
                lower_bound, upper_bound = model.predict_interval(X)
                
                # Ensure non-negative predictions
                y_pred = np.maximum(0, y_pred)
                lower_bound = np.maximum(0, lower_bound)
            else:
                # Scale features
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                self.scalers[f"{group_name}_{staff_type}"] = scaler
                
                # Use standard RF for CNA and LPN
                model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=8,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    random_state=42
                )
                model.fit(X_scaled, y)
                
                # Get predictions
                y_pred = model.predict(X_scaled)
                
                # Calculate prediction intervals
                lower_quantile = []
                upper_quantile = []
                
                for estimator in model.estimators_:
                    pred = estimator.predict(X_scaled)
                    lower_quantile.append(pred)
                    upper_quantile.append(pred)
                
                lower_bound = np.maximum(0, np.percentile(lower_quantile, 2.5, axis=0))
                upper_bound = np.percentile(upper_quantile, 97.5, axis=0)
            
            models[staff_type] = model
            predictions[staff_type] = {
                'point_prediction': np.mean(y_pred),
                'lower_bound': np.mean(lower_bound),
                'upper_bound': np.mean(upper_bound)
            }
        
        return models, predictions
    
    def generate_predictions(self, group_data, group_name=None):
        """Generate predictions for a single NH from the group"""
        if group_name is None:
            group_name = "Group 1"  # Default group name
            
        # Train models for this group
        _, predictions = self.train_models(group_data, group_name)
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
        
        # Create Excel writer
        writer = pd.ExcelWriter(output_file, engine='openpyxl')
        
        # Create predictions for each group
        for group in self.groups:
            # Create features for prediction dates
            pred_features = {}
            for staff_type in ['CNA', 'LPN', 'RN']:
                pred_features[staff_type] = self.create_features(
                    pd.Series(dates),
                    group,
                    []  # No historical data for future dates
                )
            
            # Train models and get predictions
            models, _ = self.train_models(self.training_data[group]['train'], group)
            
            # Create data for the sheet
            data = []
            for i, date in enumerate(dates):
                # Get predictions for this date
                row_predictions = {}
                for staff_type in ['CNA', 'LPN', 'RN']:
                    if staff_type == 'RN':
                        # Use the trained hybrid model directly
                        model = models[staff_type]
                        X = pred_features[staff_type].iloc[i:i+1]
                        pred = model.predict(X)[0]
                        lower_bound, upper_bound = model.predict_interval(X)
                        lower_bound = max(0, lower_bound[0])
                        upper_bound = upper_bound[0]
                    else:
                        # Use standard RF model
                        scaler = self.scalers[f"{group}_{staff_type}"]
                        X_scaled = scaler.transform(pred_features[staff_type].iloc[i:i+1])
                        model = models[staff_type]
                        pred = model.predict(X_scaled)[0]
                        
                        # Get prediction intervals
                        preds = []
                        for estimator in model.estimators_:
                            preds.append(estimator.predict(X_scaled)[0])
                        
                        lower_bound = max(0, np.percentile(preds, 2.5))
                        upper_bound = np.percentile(preds, 97.5)
                    
                    row_predictions[f'{staff_type} Point Prediction'] = round(pred, 2)
                    row_predictions[f'{staff_type} Lower Bound'] = round(lower_bound, 2)
                    row_predictions[f'{staff_type} Upper Bound'] = round(upper_bound, 2)
                
                row = {'Date': date.strftime('%-m/%-d/%y')}
                row.update(row_predictions)
                data.append(row)
            
            # Create DataFrame and save to sheet
            df = pd.DataFrame(data)
            df.to_excel(writer, sheet_name=group, index=False)
            
        writer.close()

class HybridRNPredictor:
    def __init__(self):
        self.rf_model = None
        self.exp_smoothing = None
        self.scaler = RobustScaler()
        self.outlier_detector = IsolationForest(contamination=0.1, random_state=42)
        self.facility_calibration = None
        self.optimal_weights = None
        self.feature_columns = None
        self.feature_names = None  # Store feature names after engineering
        
    def _enhance_features(self, X):
        """Add sophisticated engineered features for better predictions"""
        # Convert to DataFrame for easier manipulation
        X_df = pd.DataFrame(X)
        
        # Store original column names
        if self.feature_columns is None:
            self.feature_columns = list(range(X_df.shape[1]))
        
        # Store all generated feature names
        generated_features = []
        
        # 1. Rolling Statistics (multiple windows)
        rolling_windows = ['3d', '7d', '14d'] if len(X_df) >= 7 else []
        for col in self.feature_columns:
            for window in rolling_windows:
                X_df[f'col_{col}_roll_mean_{window}'] = (
                    X_df[col].rolling(window=int(window[:-1]), min_periods=1).mean()
                    if rolling_windows else 0
                )
                X_df[f'col_{col}_roll_std_{window}'] = (
                    X_df[col].rolling(window=int(window[:-1]), min_periods=1).std().fillna(0)
                    if rolling_windows else 0
                )
                generated_features.extend([
                    f'col_{col}_roll_mean_{window}',
                    f'col_{col}_roll_std_{window}'
                ])
            
            if rolling_windows:
                X_df[f'col_{col}_trend_3d'] = X_df[f'col_{col}_roll_mean_3d'] - X_df[f'col_{col}_roll_mean_7d']
                X_df[f'col_{col}_trend_7d'] = X_df[f'col_{col}_roll_mean_7d'] - X_df[f'col_{col}_roll_mean_14d']
            else:
                X_df[f'col_{col}_trend_3d'] = 0
                X_df[f'col_{col}_trend_7d'] = 0
            generated_features.extend([
                f'col_{col}_trend_3d',
                f'col_{col}_trend_7d'
            ])
        
        # 2. Volatility Features
        for col in self.feature_columns:
            if rolling_windows:
                X_df[f'col_{col}_rel_vol_7d'] = (
                    X_df[f'col_{col}_roll_std_7d'] / 
                    X_df[f'col_{col}_roll_mean_7d'].replace(0, 1)
                )
                X_df[f'col_{col}_range_7d'] = (
                    X_df[col].rolling(window=7, min_periods=1).max() -
                    X_df[col].rolling(window=7, min_periods=1).min()
                )
            else:
                X_df[f'col_{col}_rel_vol_7d'] = 0
                X_df[f'col_{col}_range_7d'] = 0
            generated_features.extend([
                f'col_{col}_rel_vol_7d',
                f'col_{col}_range_7d'
            ])
        
        # 3. Temporal Features
        # Use position in sequence for temporal features
        day_num = np.arange(len(X_df)) % 7
        month_num = (np.arange(len(X_df)) % 365) // 30 + 1
        
        temporal_features = [
            'day_sin', 'day_cos', 'month_sin', 'month_cos', 'is_weekend'
        ]
        
        X_df['day_sin'] = np.sin(2 * np.pi * day_num / 7.0)
        X_df['day_cos'] = np.cos(2 * np.pi * day_num / 7.0)
        X_df['month_sin'] = np.sin(2 * np.pi * month_num / 12.0)
        X_df['month_cos'] = np.cos(2 * np.pi * month_num / 12.0)
        X_df['is_weekend'] = (day_num >= 5).astype(float)
        
        generated_features.extend(temporal_features)
        
        # 4. Interaction Features
        for col in self.feature_columns:
            if rolling_windows:
                X_df[f'col_{col}_trend_vol_interact'] = (
                    X_df[f'col_{col}_trend_7d'] * X_df[f'col_{col}_rel_vol_7d']
                )
            else:
                X_df[f'col_{col}_trend_vol_interact'] = 0
            
            X_df[f'col_{col}_weekend_effect'] = X_df[col] * X_df['is_weekend']
            
            generated_features.extend([
                f'col_{col}_trend_vol_interact',
                f'col_{col}_weekend_effect'
            ])
        
        # 5. Feature Ratios and Statistics
        if rolling_windows:
            # Pairwise ratios of original features
            for i, col1 in enumerate(self.feature_columns[:-1]):
                for col2 in self.feature_columns[i+1:]:
                    ratio_name = f'col_{col1}_{col2}_ratio'
                    X_df[ratio_name] = (
                        X_df[col1].rolling(window=7, min_periods=1).mean() /
                        X_df[col2].rolling(window=7, min_periods=1).mean().replace(0, 1)
                    )
                    X_df[f'{ratio_name}_std'] = (
                        X_df[ratio_name].rolling(window=7, min_periods=1).std().fillna(0)
                    )
                    generated_features.extend([ratio_name, f'{ratio_name}_std'])
        
        # Global statistics
        global_stats = ['mean_value', 'std_value', 'max_value', 'min_value']
        X_df['mean_value'] = X_df[self.feature_columns].mean(axis=1)
        X_df['std_value'] = X_df[self.feature_columns].std(axis=1)
        X_df['max_value'] = X_df[self.feature_columns].max(axis=1)
        X_df['min_value'] = X_df[self.feature_columns].min(axis=1)
        generated_features.extend(global_stats)
        
        # Store feature names if not already stored
        if self.feature_names is None:
            self.feature_names = generated_features
        
        # Ensure all features are present and in the same order
        for feature in self.feature_names:
            if feature not in X_df.columns:
                X_df[feature] = 0
        
        # 6. Remove or replace infinities and NaNs
        X_df = X_df.replace([np.inf, -np.inf], 0)
        X_df = X_df.fillna(0)
        
        # Return only the engineered features in the correct order
        return X_df[self.feature_names].values
    
    def fit(self, X, y):
        # Detect and handle outliers
        outlier_labels = self.outlier_detector.fit_predict(y.reshape(-1, 1))
        clean_indices = outlier_labels == 1
        
        if not np.any(clean_indices):
            clean_indices = np.ones_like(outlier_labels, dtype=bool)
        
        X_clean = X[clean_indices]
        y_clean = y[clean_indices]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_clean)
        
        # Add engineered features
        X_enhanced = self._enhance_features(X_scaled)
        
        # Train Random Forest with optimized parameters
        self.rf_model = RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=4,
            random_state=42
        )
        self.rf_model.fit(X_enhanced, y_clean)
        
        # Train Exponential Smoothing if enough data
        if len(y_clean) >= 14:
            self.exp_smoothing = ExponentialSmoothing(
                y_clean,
                seasonal_periods=7,
                trend='add',
                seasonal='add',
                damped_trend=True
            ).fit()
            
            # Find optimal weights through cross-validation
            self.optimal_weights = self._find_optimal_weights(X_enhanced, y_clean)
        else:
            self.optimal_weights = {'rf': 0.8, 'es': 0.2}  # Default weights for small datasets
            
        # Calculate facility calibration factors
        predictions = self.rf_model.predict(X_enhanced)
        self.facility_calibration = self._calculate_calibration(y_clean, predictions)
    
    def _find_optimal_weights(self, X, y, cv_folds=5):
        """Find optimal weights for RF and ES models using cross-validation"""
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        best_weights = {'rf': 0.8, 'es': 0.2}
        best_score = float('inf')
        
        for rf_weight in np.arange(0.6, 0.95, 0.05):
            es_weight = 1 - rf_weight
            cv_scores = []
            
            for train_idx, val_idx in kf.split(X):
                # Get predictions for both models
                rf_pred = self.rf_model.predict(X[val_idx])
                if self.exp_smoothing is not None:
                    es_pred = self.exp_smoothing.forecast(len(val_idx))[:len(val_idx)]
                    # Combine predictions
                    y_pred = rf_weight * rf_pred + es_weight * es_pred
                else:
                    y_pred = rf_pred
                
                # Calculate SMAPE for this fold
                score = self._calculate_smape(y[val_idx], y_pred)
                cv_scores.append(score)
            
            avg_score = np.mean(cv_scores)
            if avg_score < best_score:
                best_score = avg_score
                best_weights = {'rf': rf_weight, 'es': es_weight}
        
        return best_weights
    
    def _calculate_calibration(self, y_true, y_pred):
        """Calculate calibration factors for the facility"""
        if len(y_true) < 2:
            return {'scale': 1.0, 'bias': 0.0}
        
        # Calculate robust scale and bias using median statistics
        scale = np.median(y_true) / np.median(y_pred) if np.median(y_pred) != 0 else 1.0
        bias = np.median(y_true - y_pred)
        
        # Limit the scale factor to prevent extreme adjustments
        scale = np.clip(scale, 0.5, 2.0)
        
        return {'scale': scale, 'bias': bias}
    
    def _calculate_smape(self, y_true, y_pred):
        """Calculate Symmetric Mean Absolute Percentage Error"""
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
        mask = denominator != 0
        return 100.0 * np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask])
    
    def predict(self, X):
        # Scale features
        X_scaled = self.scaler.transform(X) if self.scaler else X
        
        # Generate enhanced features
        X_enhanced = self._enhance_features(X_scaled)
        
        # Get Random Forest prediction
        rf_pred = self.rf_model.predict(X_enhanced)
        
        # Get prediction intervals from Random Forest
        rf_std = np.std([tree.predict(X_enhanced) for tree in self.rf_model.estimators_], axis=0)
        
        # Calculate exponential smoothing prediction if we have enough historical data
        if len(X) > 1:
            exp_pred = self.exp_smoothing.forecast(1)[0] if self.exp_smoothing else rf_pred
        else:
            exp_pred = rf_pred
        
        # Combine predictions using optimal weights
        if self.optimal_weights is not None:
            point_pred = (
                self.optimal_weights['rf'] * rf_pred +
                self.optimal_weights['es'] * exp_pred
            )
        else:
            point_pred = rf_pred
        
        # Apply facility calibration if available
        if self.facility_calibration is not None:
            point_pred = (
                point_pred * self.facility_calibration.get('scale', 1.0) +
                self.facility_calibration.get('bias', 0.0)
            )
        
        # Calculate prediction intervals
        # Use a minimum standard deviation to prevent zero bounds
        min_std = 0.1 * point_pred  # Set minimum std as 10% of prediction
        combined_std = np.maximum(rf_std, min_std)
        
        # Calculate bounds using both aleatory (data) and epistemic (model) uncertainty
        z_score = 1.96  # 95% confidence interval
        
        # Add model uncertainty factor that scales with prediction magnitude
        model_uncertainty = 0.15 * point_pred  # 15% of prediction value
        total_std = np.sqrt(combined_std**2 + model_uncertainty**2)
        
        lower_bound = np.maximum(point_pred - z_score * total_std, 0.2 * point_pred)  # Ensure lower bound is at least 20% of prediction
        upper_bound = point_pred + z_score * total_std
        
        # Handle single predictions vs arrays
        if isinstance(point_pred, np.ndarray) and len(point_pred) == 1:
            return float(point_pred[0]), float(lower_bound[0]), float(upper_bound[0])
        
        return point_pred, lower_bound, upper_bound
    
    def predict_interval(self, X):
        # Scale and enhance features
        X_scaled = self.scaler.transform(X) if self.scaler else X
        X_enhanced = self._enhance_features(X_scaled)
        
        # Get predictions from all RF estimators
        predictions = []
        for estimator in self.rf_model.estimators_:
            pred = estimator.predict(X_enhanced)
            predictions.append(pred)
        
        # Calculate base prediction intervals
        lower_bound = np.percentile(predictions, 2.5, axis=0)
        upper_bound = np.percentile(predictions, 97.5, axis=0)
        
        # Apply facility calibration to intervals
        if self.facility_calibration is not None:
            lower_bound = (
                lower_bound * self.facility_calibration.get('scale', 1.0) +
                self.facility_calibration.get('bias', 0.0)
            )
            upper_bound = (
                upper_bound * self.facility_calibration.get('scale', 1.0) +
                self.facility_calibration.get('bias', 0.0)
            )
        
        # Ensure non-negative bounds
        lower_bound = np.maximum(0, lower_bound)
        
        return lower_bound, upper_bound

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
        'CNA': {'MAE': [], 'MAPE': [], 'SMAPE': [], 'MIS': []},
        'LPN': {'MAE': [], 'MAPE': [], 'SMAPE': [], 'MIS': []},
        'RN': {'MAE': [], 'MAPE': [], 'SMAPE': [], 'MIS': []}
    }
    
    test_metrics = {
        'CNA': {'MAE': [], 'MAPE': [], 'SMAPE': [], 'MIS': []},
        'LPN': {'MAE': [], 'MAPE': [], 'SMAPE': [], 'MIS': []},
        'RN': {'MAE': [], 'MAPE': [], 'SMAPE': [], 'MIS': []}
    }
    
    for group in predictor.groups:
        # Train models on training data
        train_predictions = predictor.generate_predictions(
            predictor.training_data[group]['train'],
            group
        )
        
        # Get predictions for test data
        test_predictions = predictor.generate_predictions(
            predictor.training_data[group]['test'],
            group
        )
        
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
        for metric_name in ['MAE', 'MAPE', 'SMAPE', 'MIS']:
            values = train_metrics[staff_type][metric_name]
            avg_value = np.mean(values)
            print(f"{metric_name}: {avg_value:.2f}")
    
    # Print test metrics
    print("\nTest Set Metrics:")
    print("----------------")
    for staff_type in ['CNA', 'LPN', 'RN']:
        print(f"\n{staff_type} Metrics:")
        print("-" * 20)
        for metric_name in ['MAE', 'MAPE', 'SMAPE', 'MIS']:
            values = test_metrics[staff_type][metric_name]
            avg_value = np.mean(values)
            print(f"{metric_name}: {avg_value:.2f}")
    
if __name__ == "__main__":
    main()
