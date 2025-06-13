import os
import click
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import pearsonr

from model import SpectralConvLSTM, FNO1dLSTM

# Set random seeds for reproducibility
def set_seed(seed):
    pl.seed_everything(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class TimeSeriesDataModule(pl.LightningDataModule):
    """PyTorch Lightning data module for time series data"""
    def __init__(self, data_path=None, batch_size=32, num_samples=1000, seq_length=50, 
                 input_dim=10, output_dim=1):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.input_dim = input_dim
        self.output_dim = output_dim
        
    def prepare_data(self):
        # Nothing to download
        pass
    
    def setup(self, stage=None):
        if self.data_path and os.path.exists(self.data_path):
            # Load real data from file
            print(f"Loading data from {self.data_path}")
            # Implement your data loading logic here
            data = np.load(self.data_path)
            X, y = data['X'], data['y']
        else:
            # Generate synthetic data for testing
            print("Generating synthetic data for testing")
            X, y = self.generate_synthetic_data()
        
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)
        
        # Create dataset
        dataset = TensorDataset(X_tensor, y_tensor)
        
        # Split into train, validation, and test sets
        train_size = int(0.7 * len(dataset))
        val_size = int(0.15 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            dataset, [train_size, val_size, test_size]
        )
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
    
    def generate_synthetic_data(self):
        """Generate synthetic time series data for testing"""
        # Generate random frequencies for each sample
        frequencies = np.random.uniform(0.1, 2.0, (self.num_samples, self.input_dim))
        
        # Time steps
        t = np.linspace(0, 10, self.seq_length)
        
        # Generate input sequences
        X = np.zeros((self.num_samples, self.seq_length, self.input_dim))
        for i in range(self.num_samples):
            for j in range(self.input_dim):
                # Create sinusoidal signals with different frequencies and phases
                X[i, :, j] = np.sin(2 * np.pi * frequencies[i, j] * t + np.random.uniform(0, np.pi))
        
        # Generate target outputs (using a simple function of the last time step)
        y = np.zeros((self.num_samples, self.output_dim))
        for i in range(self.num_samples):
            # Use the last time step's values to generate the output
            y[i] = np.sum(X[i, -1, :]) / self.input_dim
            
            # Add some non-linearity
            if self.output_dim > 1:
                for j in range(1, self.output_dim):
                    y[i, j] = np.sin(y[i, 0] * j)
        
        return X, y

class SpectralConvLSTMLightning(pl.LightningModule):
    """PyTorch Lightning module for SpectralConvLSTM model"""
    def __init__(self, model_type='fno1d_lstm', input_dim=10, hidden_channels=64, n_modes=16, 
                 hidden_dim=128, layer_dim=2, output_dim=1, n_layers=4, dropout=0.1, 
                 learning_rate=0.001, weight_decay=1e-5):
        super().__init__()
        self.save_hyperparameters()
        
        # Create model based on type
        if model_type == 'spectral_conv_lstm':
            self.model = SpectralConvLSTM(
                input_dim=input_dim,
                hidden_channels=hidden_channels,
                n_modes=n_modes,
                hidden_dim=hidden_dim,
                layer_dim=layer_dim,
                output_dim=output_dim,
                dropout=dropout
            )
        else:  # fno1d_lstm
            self.model = FNO1dLSTM(
                modes=n_modes,
                width=hidden_channels,
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                layer_dim=layer_dim,
                output_dim=output_dim,
                n_layers=n_layers
            )
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Save hyperparameters for optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # For tracking metrics
        self.train_mse = []
        self.val_mse = []
        self.test_predictions = None
        self.test_actuals = None
    
    def forward(self, x):
        # Forward pass through the model
        output, _, _ = self.model(x)
        return output
    
    def training_step(self, batch, batch_idx):
        # Training step
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        
        # Log metrics
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        # Validation step
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        
        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        # Test step
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        
        # Log metrics
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        
        # Store predictions and actuals for later analysis
        if self.test_predictions is None:
            self.test_predictions = y_hat.detach().cpu()
            self.test_actuals = y.detach().cpu()
        else:
            self.test_predictions = torch.cat([self.test_predictions, y_hat.detach().cpu()], dim=0)
            self.test_actuals = torch.cat([self.test_actuals, y.detach().cpu()], dim=0)
        
        return loss
    
    def on_test_epoch_end(self):
        # Calculate metrics at the end of test
        predictions = self.test_predictions.numpy()
        actuals = self.test_actuals.numpy()
        
        # Compute comprehensive metrics
        metrics = self.calculate_metrics(predictions, actuals, model_name='FNO-LSTM')
        
        # Log metrics
        for metric_name, metric_value in metrics.items():
            self.log(f'test_{metric_name}', metric_value)
        
        # Print results
        print(f"\nTest Results for FNO-LSTM:")
        for metric_name, metric_value in metrics.items():
            print(f"{metric_name}: {metric_value:.6f}")
        
        # Compare with statistical models
        if hasattr(self, 'trainer') and hasattr(self.trainer, 'datamodule'):
            try:
                stat_predictions, stat_metrics = self.compare_with_statistical_models()
                
                # Print comparison metrics
                print("\nModel Comparison Metrics:")
                metrics_df = pd.DataFrame([metrics] + list(stat_metrics.values()))
                metrics_df['Model'] = ['FNO-LSTM'] + list(stat_metrics.keys())
                metrics_df = metrics_df.set_index('Model')
                print(metrics_df.round(6))
                
                # Save metrics to CSV
                if hasattr(self, 'output_dir') and self.output_dir:
                    metrics_path = os.path.join(self.output_dir, 'model_comparison_metrics.csv')
                    metrics_df.to_csv(metrics_path)
                    print(f"Metrics saved to {metrics_path}")
                
                # Plot results with statistical model comparisons
                if hasattr(self, 'output_dir') and self.output_dir:
                    self.plot_results_with_comparison(predictions, actuals, stat_predictions, stat_metrics)
            except Exception as e:
                print(f"Warning: Could not run statistical model comparison: {e}")
                # Fall back to regular plotting
                if hasattr(self, 'output_dir') and self.output_dir:
                    self.plot_detailed_results(predictions, actuals)
        else:
            # Plot results if output directory is set
            if hasattr(self, 'output_dir') and self.output_dir:
                self.plot_detailed_results(predictions, actuals)
    
    def calculate_metrics(self, predictions, actuals, model_name=None):
        """Calculate comprehensive set of evaluation metrics"""
        # Basic metrics
        mse = np.mean((predictions - actuals) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - actuals))
        
        # R-squared (coefficient of determination)
        # For multivariate output, calculate R² for each dimension and average
        if actuals.shape[1] > 1:
            r2 = np.mean([r2_score(actuals[:, i], predictions[:, i]) for i in range(actuals.shape[1])])
        else:
            r2 = r2_score(actuals, predictions)
        
        # Mean Absolute Percentage Error (MAPE)
        # Avoid division by zero by adding small epsilon
        epsilon = 1e-10
        mape = np.mean(np.abs((actuals - predictions) / (np.abs(actuals) + epsilon))) * 100
        
        # Pearson correlation coefficient (for 1D output)
        if actuals.shape[1] == 1:
            try:
                corr, _ = pearsonr(actuals.flatten(), predictions.flatten())
            except:
                corr = np.nan
        else:
            # For multivariate, average the correlation across dimensions
            corrs = []
            for i in range(actuals.shape[1]):
                try:
                    c, _ = pearsonr(actuals[:, i], predictions[:, i])
                    corrs.append(c)
                except:
                    corrs.append(np.nan)
            corr = np.nanmean(corrs)
        
        # Normalized RMSE (NRMSE)
        # Normalize by the range of the actual values
        range_actuals = np.max(actuals) - np.min(actuals)
        if range_actuals > 0:
            nrmse = rmse / range_actuals
        else:
            nrmse = np.nan
            
        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'MAPE': mape,
            'Correlation': corr,
            'NRMSE': nrmse
        }
        
        return metrics
    
    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }
    
    def compare_with_statistical_models(self):
        """Compare FNO-LSTM with statistical models like ARIMA, SARIMAX, Linear Regression, and Random Forest"""
        # Get test data from datamodule
        test_data = self.trainer.datamodule.test_dataloader()
        
        # Extract all test sequences and targets
        all_x = []
        all_y = []
        for batch in test_data:
            x, y = batch
            all_x.append(x)
            all_y.append(y)
        
        x_test = torch.cat(all_x, dim=0).numpy()
        y_test = torch.cat(all_y, dim=0).numpy()
        
        # Number of test samples and sequence length
        n_samples = x_test.shape[0]
        seq_len = x_test.shape[1]
        input_dim = x_test.shape[2]
        output_dim = y_test.shape[1]
        
        print(f"\nRunning statistical model comparison on {n_samples} test samples...")
        
        # Dictionaries to store predictions and metrics from different models
        stat_predictions = {}
        stat_metrics = {}
        
        # 1. Linear Regression
        print("Training Linear Regression model...")
        # Reshape input for sklearn: (n_samples, seq_len * input_dim)
        x_flat = x_test.reshape(n_samples, -1)
        lr_model = LinearRegression()
        lr_model.fit(x_flat, y_test)
        lr_preds = lr_model.predict(x_flat)
        
        # Calculate comprehensive metrics
        lr_metrics = self.calculate_metrics(lr_preds, y_test, model_name='Linear Regression')
        print(f"Linear Regression - R²: {lr_metrics['R2']:.4f}, RMSE: {lr_metrics['RMSE']:.6f}, MAE: {lr_metrics['MAE']:.6f}")
        
        stat_predictions['Linear Regression'] = lr_preds
        stat_metrics['Linear Regression'] = lr_metrics
        
        # 2. Random Forest
        print("Training Random Forest model...")
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(x_flat, y_test.ravel() if output_dim == 1 else y_test)
        rf_preds = rf_model.predict(x_flat).reshape(y_test.shape)
        
        # Calculate comprehensive metrics
        rf_metrics = self.calculate_metrics(rf_preds, y_test, model_name='Random Forest')
        print(f"Random Forest - R²: {rf_metrics['R2']:.4f}, RMSE: {rf_metrics['RMSE']:.6f}, MAE: {rf_metrics['MAE']:.6f}")
        
        stat_predictions['Random Forest'] = rf_preds
        stat_metrics['Random Forest'] = rf_metrics
        
        # 3. SVR (Support Vector Regression) - for smaller datasets
        try:
            from sklearn.svm import SVR
            if n_samples <= 5000:  # SVR can be slow on large datasets
                print("Training SVR model...")
                # For multivariate output, train separate SVR for each output dimension
                svr_preds = np.zeros_like(y_test)
                
                if output_dim == 1:
                    svr_model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
                    svr_model.fit(x_flat, y_test.ravel())
                    svr_preds = svr_model.predict(x_flat).reshape(y_test.shape)
                else:
                    for i in range(output_dim):
                        svr_model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
                        svr_model.fit(x_flat, y_test[:, i])
                        svr_preds[:, i] = svr_model.predict(x_flat)
                
                # Calculate comprehensive metrics
                svr_metrics = self.calculate_metrics(svr_preds, y_test, model_name='SVR')
                print(f"SVR - R²: {svr_metrics['R2']:.4f}, RMSE: {svr_metrics['RMSE']:.6f}, MAE: {svr_metrics['MAE']:.6f}")
                
                stat_predictions['SVR'] = svr_preds
                stat_metrics['SVR'] = svr_metrics
        except Exception as e:
            print(f"SVR model failed: {e}")
        
        # For time series models, we'll use a subset of data if there are too many samples
        max_arima_samples = min(100, n_samples)  # Limit to avoid long computation times
        
        # 4. ARIMA - only for univariate time series and limited samples
        if input_dim == 1 and output_dim == 1:
            try:
                print(f"Training ARIMA model on {max_arima_samples} samples...")
                arima_preds = np.zeros((max_arima_samples, output_dim))
                
                # Use last value of each sequence as the time series
                time_series = x_test[:max_arima_samples, -1, 0]
                
                # Fit ARIMA model
                arima_model = ARIMA(time_series, order=(5,1,0))
                arima_result = arima_model.fit()
                
                # Make one-step ahead forecasts
                for i in range(max_arima_samples):
                    if i > 0:  # Skip first prediction as we need history
                        history = time_series[:i]
                        model = ARIMA(history, order=(5,1,0))
                        model_fit = model.fit()
                        arima_preds[i, 0] = model_fit.forecast()[0]
                
                # Calculate comprehensive metrics on the subset
                arima_metrics = self.calculate_metrics(arima_preds, y_test[:max_arima_samples], model_name='ARIMA')
                print(f"ARIMA - R²: {arima_metrics['R2']:.4f}, RMSE: {arima_metrics['RMSE']:.6f}, MAE: {arima_metrics['MAE']:.6f}")
                
                stat_predictions['ARIMA'] = arima_preds
                stat_metrics['ARIMA'] = arima_metrics
            except Exception as e:
                print(f"ARIMA model failed: {e}")
        
        # 5. SARIMAX - for more complex time series
        if input_dim <= 3 and output_dim == 1:  # Limit to low-dimensional inputs
            try:
                print(f"Training SARIMAX model on {max_arima_samples} samples...")
                sarimax_preds = np.zeros((max_arima_samples, output_dim))
                
                # Use exogenous variables if available
                endog = x_test[:max_arima_samples, -1, 0]  # Target variable
                exog = None
                if input_dim > 1:
                    exog = x_test[:max_arima_samples, -1, 1:]
                
                # Fit SARIMAX model
                sarimax_model = SARIMAX(endog, exog=exog, order=(1,1,1), seasonal_order=(0,0,0,0))
                sarimax_result = sarimax_model.fit(disp=False)
                
                # Make predictions
                sarimax_preds[:, 0] = sarimax_result.predict()
                
                # Calculate comprehensive metrics
                sarimax_metrics = self.calculate_metrics(sarimax_preds, y_test[:max_arima_samples], model_name='SARIMAX')
                print(f"SARIMAX - R²: {sarimax_metrics['R2']:.4f}, RMSE: {sarimax_metrics['RMSE']:.6f}, MAE: {sarimax_metrics['MAE']:.6f}")
                
                stat_predictions['SARIMAX'] = sarimax_preds
                stat_metrics['SARIMAX'] = sarimax_metrics
            except Exception as e:
                print(f"SARIMAX model failed: {e}")
        
        return stat_predictions, stat_metrics
    
    def plot_results(self, predictions, actuals, save_path=None):
        """Plot predictions vs actual values (basic version)"""
        plt.figure(figsize=(10, 6))
        
        # If output is 1D, create a simple scatter plot
        if actuals.shape[1] == 1:
            plt.scatter(actuals, predictions, alpha=0.5)
            plt.plot([actuals.min(), actuals.max()], [actuals.min(), actuals.max()], 'r--')
            plt.xlabel('Actual Values')
            plt.ylabel('Predicted Values')
            plt.title('Predictions vs Actual Values')
        else:
            # If output is multi-dimensional, plot first two dimensions
            plt.scatter(actuals[:, 0], actuals[:, 1], alpha=0.5, label='Actual')
            plt.scatter(predictions[:, 0], predictions[:, 1], alpha=0.5, label='Predicted')
            plt.xlabel('Dimension 1')
            plt.ylabel('Dimension 2')
            plt.title('Predictions vs Actual Values (First 2 Dimensions)')
            plt.legend()
        
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
        else:
            plt.savefig(os.path.join(self.output_dir, 'predictions_vs_actuals.png'))
            print(f"Plot saved to {os.path.join(self.output_dir, 'predictions_vs_actuals.png')}")
        
        plt.close()
        
    def plot_detailed_results(self, predictions, actuals, save_path=None):
        """Create detailed visualizations comparing predicted vs. real data"""
        # Create output directory for plots if it doesn't exist
        plots_dir = os.path.join(self.output_dir, 'detailed_plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # 1. Scatter plot with regression line and confidence intervals
        plt.figure(figsize=(10, 8))
        
        if actuals.shape[1] == 1:  # For 1D output
            # Flatten arrays for easier plotting
            act_flat = actuals.flatten()
            pred_flat = predictions.flatten()
            
            # Create scatter plot
            plt.scatter(act_flat, pred_flat, alpha=0.5, label='Data Points')
            
            # Add perfect prediction line
            min_val = min(act_flat.min(), pred_flat.min())
            max_val = max(act_flat.max(), pred_flat.max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
            
            # Add regression line
            from sklearn.linear_model import LinearRegression
            reg = LinearRegression().fit(act_flat.reshape(-1, 1), pred_flat)
            reg_y = reg.predict(np.linspace(min_val, max_val, 100).reshape(-1, 1))
            plt.plot(np.linspace(min_val, max_val, 100), reg_y, 'g-', label=f'Regression Line (slope={reg.coef_[0]:.4f})')
            
            # Calculate metrics for annotation
            metrics = self.calculate_metrics(predictions, actuals)
            plt.annotate(f"R² = {metrics['R2']:.4f}\nRMSE = {metrics['RMSE']:.4f}\nMAE = {metrics['MAE']:.4f}", 
                         xy=(0.05, 0.95), xycoords='axes fraction',
                         bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
                         ha='left', va='top')
            
            plt.xlabel('Actual Values')
            plt.ylabel('Predicted Values')
            plt.title('Detailed Comparison: Predicted vs. Actual Values')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Save plot
            scatter_path = os.path.join(plots_dir, 'detailed_scatter.png')
            plt.savefig(scatter_path)
            print(f"Detailed scatter plot saved to {scatter_path}")
            
        else:  # For multi-dimensional output
            # Create subplots for each dimension (up to 6)
            n_dims = min(6, actuals.shape[1])
            fig, axes = plt.subplots(n_dims, 1, figsize=(10, 4*n_dims), sharex=False)
            
            for i in range(n_dims):
                ax = axes[i] if n_dims > 1 else axes
                
                # Extract dimension data
                act_dim = actuals[:, i]
                pred_dim = predictions[:, i]
                
                # Create scatter plot
                ax.scatter(act_dim, pred_dim, alpha=0.5)
                
                # Add perfect prediction line
                min_val = min(act_dim.min(), pred_dim.min())
                max_val = max(act_dim.max(), pred_dim.max())
                ax.plot([min_val, max_val], [min_val, max_val], 'r--')
                
                # Calculate R² for this dimension
                r2 = r2_score(act_dim, pred_dim)
                rmse = np.sqrt(mean_squared_error(act_dim, pred_dim))
                
                ax.set_title(f'Dimension {i+1}: R² = {r2:.4f}, RMSE = {rmse:.4f}')
                ax.set_xlabel('Actual Values')
                ax.set_ylabel('Predicted Values')
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            multi_scatter_path = os.path.join(plots_dir, 'multi_dim_scatter.png')
            plt.savefig(multi_scatter_path)
            print(f"Multi-dimensional scatter plot saved to {multi_scatter_path}")
        
        plt.close()
        
        # 2. Residual plot
        plt.figure(figsize=(10, 6))
        
        if actuals.shape[1] == 1:  # For 1D output
            # Calculate residuals
            residuals = predictions.flatten() - actuals.flatten()
            
            # Create scatter plot of residuals
            plt.scatter(actuals.flatten(), residuals, alpha=0.5)
            plt.axhline(y=0, color='r', linestyle='--')
            
            # Add a smoothed trend line
            try:
                from scipy.signal import savgol_filter
                # Sort points for smooth line
                sorted_indices = np.argsort(actuals.flatten())
                sorted_x = actuals.flatten()[sorted_indices]
                sorted_residuals = residuals[sorted_indices]
                
                if len(sorted_x) > 10:  # Need enough points for smoothing
                    window_size = min(51, len(sorted_x) - 2 - (len(sorted_x) % 2))
                    if window_size > 3:  # Valid window size
                        smoothed = savgol_filter(sorted_residuals, window_size, 3)
                        plt.plot(sorted_x, smoothed, 'g-', linewidth=2, label='Trend')
                        plt.legend()
            except Exception as e:
                print(f"Could not create trend line: {e}")
            
            plt.xlabel('Actual Values')
            plt.ylabel('Residuals (Predicted - Actual)')
            plt.title('Residual Plot')
            plt.grid(True, alpha=0.3)
            
            # Save plot
            residual_path = os.path.join(plots_dir, 'residual_plot.png')
            plt.savefig(residual_path)
            print(f"Residual plot saved to {residual_path}")
            
        else:  # For multi-dimensional output
            # Create subplots for each dimension (up to 6)
            n_dims = min(6, actuals.shape[1])
            fig, axes = plt.subplots(n_dims, 1, figsize=(10, 3*n_dims), sharex=False)
            
            for i in range(n_dims):
                ax = axes[i] if n_dims > 1 else axes
                
                # Calculate residuals for this dimension
                residuals = predictions[:, i] - actuals[:, i]
                
                # Create scatter plot
                ax.scatter(actuals[:, i], residuals, alpha=0.5)
                ax.axhline(y=0, color='r', linestyle='--')
                
                ax.set_title(f'Dimension {i+1} Residuals')
                ax.set_xlabel('Actual Values')
                ax.set_ylabel('Residuals')
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            multi_residual_path = os.path.join(plots_dir, 'multi_dim_residual.png')
            plt.savefig(multi_residual_path)
            print(f"Multi-dimensional residual plot saved to {multi_residual_path}")
        
        plt.close()
        
        # 3. Time series plot (for a sample of points)
        plt.figure(figsize=(12, 6))
        
        # Select a subset of points to plot as a time series
        sample_size = min(100, len(actuals))
        indices = np.arange(sample_size)
        
        if actuals.shape[1] == 1:  # For 1D output
            # Plot actual vs predicted values
            plt.plot(indices, actuals[:sample_size].flatten(), 'b-', label='Actual', linewidth=2)
            plt.plot(indices, predictions[:sample_size].flatten(), 'r--', label='Predicted', linewidth=2)
            
            plt.xlabel('Sample Index')
            plt.ylabel('Value')
            plt.title('Time Series: Actual vs. Predicted Values')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Save plot
            ts_path = os.path.join(plots_dir, 'time_series.png')
            plt.savefig(ts_path)
            print(f"Time series plot saved to {ts_path}")
            
        else:  # For multi-dimensional output
            # Create subplots for each dimension (up to 4)
            n_dims = min(4, actuals.shape[1])
            fig, axes = plt.subplots(n_dims, 1, figsize=(12, 3*n_dims), sharex=True)
            
            for i in range(n_dims):
                ax = axes[i] if n_dims > 1 else axes
                
                # Plot actual vs predicted for this dimension
                ax.plot(indices, actuals[:sample_size, i], 'b-', label='Actual', linewidth=2)
                ax.plot(indices, predictions[:sample_size, i], 'r--', label='Predicted', linewidth=2)
                
                ax.set_title(f'Dimension {i+1}')
                ax.set_ylabel('Value')
                ax.grid(True, alpha=0.3)
                ax.legend()
            
            # Set common x label
            fig.text(0.5, 0.04, 'Sample Index', ha='center')
            fig.suptitle('Time Series: Actual vs. Predicted Values by Dimension')
            plt.tight_layout()
            plt.subplots_adjust(top=0.9, bottom=0.1)
            
            # Save plot
            multi_ts_path = os.path.join(plots_dir, 'multi_dim_time_series.png')
            plt.savefig(multi_ts_path)
            print(f"Multi-dimensional time series plot saved to {multi_ts_path}")
        
        plt.close()
        
        # 4. Error distribution histogram
        plt.figure(figsize=(10, 6))
        
        if actuals.shape[1] == 1:  # For 1D output
            # Calculate absolute errors
            errors = np.abs(predictions.flatten() - actuals.flatten())
            
            # Create histogram
            plt.hist(errors, bins=30, alpha=0.7, color='blue')
            plt.axvline(x=np.mean(errors), color='r', linestyle='--', label=f'Mean Error: {np.mean(errors):.4f}')
            
            plt.xlabel('Absolute Error')
            plt.ylabel('Frequency')
            plt.title('Error Distribution')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Save plot
            hist_path = os.path.join(plots_dir, 'error_histogram.png')
            plt.savefig(hist_path)
            print(f"Error histogram saved to {hist_path}")
            
        else:  # For multi-dimensional output
            # Create subplots for each dimension (up to 6)
            n_dims = min(6, actuals.shape[1])
            fig, axes = plt.subplots(n_dims, 1, figsize=(10, 3*n_dims), sharex=False)
            
            for i in range(n_dims):
                ax = axes[i] if n_dims > 1 else axes
                
                # Calculate absolute errors for this dimension
                errors = np.abs(predictions[:, i] - actuals[:, i])
                
                # Create histogram
                ax.hist(errors, bins=30, alpha=0.7, color='blue')
                ax.axvline(x=np.mean(errors), color='r', linestyle='--', 
                           label=f'Mean Error: {np.mean(errors):.4f}')
                
                ax.set_title(f'Dimension {i+1} Error Distribution')
                ax.set_xlabel('Absolute Error')
                ax.set_ylabel('Frequency')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            multi_hist_path = os.path.join(plots_dir, 'multi_dim_error_histogram.png')
            plt.savefig(multi_hist_path)
            print(f"Multi-dimensional error histogram saved to {multi_hist_path}")
        
        plt.close()
        
        # Create a summary image with all plots
        try:
            from PIL import Image
            import glob
            
            # Get all PNG files in the plots directory
            plot_files = glob.glob(os.path.join(plots_dir, '*.png'))
            
            if len(plot_files) > 0:
                # Open images and get their sizes
                images = [Image.open(f) for f in plot_files]
                widths, heights = zip(*(i.size for i in images))
                
                # Determine layout (2 columns)
                n_images = len(images)
                n_cols = min(2, n_images)
                n_rows = (n_images + n_cols - 1) // n_cols  # Ceiling division
                
                # Create a new image with enough space for all plots
                max_width = max(widths)
                max_height = max(heights)
                summary = Image.new('RGB', (max_width * n_cols, max_height * n_rows), color='white')
                
                # Paste all images into the summary image
                for i, img in enumerate(images):
                    row = i // n_cols
                    col = i % n_cols
                    summary.paste(img, (col * max_width, row * max_height))
                
                # Save the summary image
                summary_path = os.path.join(self.output_dir, 'prediction_analysis_summary.png')
                summary.save(summary_path)
                print(f"Summary visualization saved to {summary_path}")
        except Exception as e:
            print(f"Could not create summary image: {e}")
        
        return plots_dir
        
    def plot_results_with_comparison(self, predictions, actuals, stat_predictions, save_path=None):
        """Plot predictions vs actual values with statistical model comparisons"""
        # 1. Scatter plot comparison
        plt.figure(figsize=(12, 8))
        
        # If output is 1D, create scatter plots for each model
        if actuals.shape[1] == 1:
            plt.scatter(actuals, predictions, alpha=0.5, label='FNO-LSTM')
            
            # Add statistical models
            for model_name, model_preds in stat_predictions.items():
                # Only use predictions with matching shape
                if model_preds.shape == predictions.shape:
                    plt.scatter(actuals, model_preds, alpha=0.3, label=model_name)
            
            plt.plot([actuals.min(), actuals.max()], [actuals.min(), actuals.max()], 'k--')
            plt.xlabel('Actual Values')
            plt.ylabel('Predicted Values')
            plt.title('Model Comparison: Predictions vs Actual Values')
            plt.legend()
        else:
            # For multi-dimensional output, just show first dimension comparison
            plt.scatter(actuals[:, 0], predictions[:, 0], alpha=0.5, label='FNO-LSTM')
            
            # Add statistical models
            for model_name, model_preds in stat_predictions.items():
                if model_preds.shape == predictions.shape:
                    plt.scatter(actuals[:, 0], model_preds[:, 0], alpha=0.3, label=model_name)
            
            plt.xlabel('Actual Values (Dim 1)')
            plt.ylabel('Predicted Values (Dim 1)')
            plt.title('Model Comparison: Predictions vs Actual Values (First Dimension)')
            plt.legend()
        
        plt.grid(True)
        scatter_path = os.path.join(self.output_dir, 'model_comparison_scatter.png')
        plt.savefig(scatter_path)
        print(f"Scatter comparison plot saved to {scatter_path}")
        plt.close()
        
        # 2. Time series plot for a sample of points
        plt.figure(figsize=(14, 8))
        
        # Select a subset of points to plot as a time series
        sample_size = min(100, len(actuals))
        indices = np.arange(sample_size)
        
        # Plot actual values
        if actuals.shape[1] == 1:
            plt.plot(indices, actuals[:sample_size, 0], 'k-', label='Actual', linewidth=2)
            
            # Plot FNO-LSTM predictions
            plt.plot(indices, predictions[:sample_size, 0], 'b-', label='FNO-LSTM', linewidth=1.5)
            
            # Plot statistical model predictions
            colors = ['r', 'g', 'm', 'c', 'y']
            for i, (model_name, model_preds) in enumerate(stat_predictions.items()):
                if model_preds.shape[0] >= sample_size:
                    plt.plot(indices, model_preds[:sample_size, 0], 
                             color=colors[i % len(colors)], linestyle='-', 
                             label=model_name, linewidth=1.5, alpha=0.7)
        else:
            # For multi-dimensional output, plot first dimension
            plt.plot(indices, actuals[:sample_size, 0], 'k-', label='Actual (Dim 1)', linewidth=2)
            plt.plot(indices, predictions[:sample_size, 0], 'b-', label='FNO-LSTM (Dim 1)', linewidth=1.5)
            
            # Plot statistical model predictions for first dimension
            colors = ['r', 'g', 'm', 'c', 'y']
            for i, (model_name, model_preds) in enumerate(stat_predictions.items()):
                if model_preds.shape[0] >= sample_size:
                    plt.plot(indices, model_preds[:sample_size, 0], 
                             color=colors[i % len(colors)], linestyle='-', 
                             label=f"{model_name} (Dim 1)", linewidth=1.5, alpha=0.7)
        
        plt.xlabel('Sample Index')
        plt.ylabel('Value')
        plt.title('Time Series Comparison of Different Models')
        plt.legend()
        plt.grid(True)
        
        # Save time series plot
        ts_path = os.path.join(self.output_dir, 'model_comparison_timeseries.png')
        plt.savefig(ts_path)
        print(f"Time series comparison plot saved to {ts_path}")
        plt.close()
        
        # 3. Bar chart of MSE and MAE for each model
        plt.figure(figsize=(12, 6))
        
        # Calculate metrics for each model
        model_names = ['FNO-LSTM'] + list(stat_predictions.keys())
        mse_values = []
        mae_values = []
        
        # FNO-LSTM metrics
        mse_values.append(np.mean((predictions - actuals) ** 2))
        mae_values.append(np.mean(np.abs(predictions - actuals)))
        
        # Statistical model metrics
        for model_name, model_preds in stat_predictions.items():
            # Handle different sized predictions (some models might only predict a subset)
            if model_preds.shape == predictions.shape:
                mse = np.mean((model_preds - actuals) ** 2)
                mae = np.mean(np.abs(model_preds - actuals))
            else:
                # For models with different prediction shapes, use the common subset
                min_samples = min(model_preds.shape[0], actuals.shape[0])
                mse = np.mean((model_preds[:min_samples] - actuals[:min_samples]) ** 2)
                mae = np.mean(np.abs(model_preds[:min_samples] - actuals[:min_samples]))
            
            mse_values.append(mse)
            mae_values.append(mae)
        
        # Create bar chart
        x = np.arange(len(model_names))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 6))
        rects1 = ax.bar(x - width/2, mse_values, width, label='MSE')
        rects2 = ax.bar(x + width/2, mae_values, width, label='MAE')
        
        ax.set_ylabel('Error')
        ax.set_title('Model Comparison: MSE and MAE')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names)
        ax.legend()
        
        # Add value labels on top of bars
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.4f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', rotation=90, fontsize=8)
        
        autolabel(rects1)
        autolabel(rects2)
        
        fig.tight_layout()
        
        # Save metrics bar chart
        metrics_path = os.path.join(self.output_dir, 'model_comparison_metrics.png')
        plt.savefig(metrics_path)
        print(f"Metrics comparison plot saved to {metrics_path}")
        plt.close()

@click.group()
def cli():
    """Command line interface for training and evaluating FNO-LSTM models."""
    pass


@cli.command()
@click.option('--data_path', type=click.Path(exists=True), help='Path to data file (.npz format with X and y arrays)')
@click.option('--num_samples', type=int, default=1000, help='Number of samples for synthetic data')
@click.option('--seq_length', type=int, default=50, help='Sequence length for time series')
@click.option('--input_dim', type=int, default=10, help='Input dimension')
@click.option('--output_dim', type=int, default=1, help='Output dimension')
@click.option('--model_type', type=click.Choice(['spectral_conv_lstm', 'fno1d_lstm']), default='fno1d_lstm', help='Type of model to use')
@click.option('--hidden_channels', type=int, default=64, help='Number of hidden channels in spectral convolution')
@click.option('--n_modes', type=int, default=16, help='Number of Fourier modes to keep')
@click.option('--hidden_dim', type=int, default=128, help='Hidden dimension of LSTM')
@click.option('--layer_dim', type=int, default=2, help='Number of LSTM layers')
@click.option('--n_layers', type=int, default=4, help='Number of spectral convolution layers (for FNO1dLSTM)')
@click.option('--dropout', type=float, default=0.1, help='Dropout probability')
@click.option('--batch_size', type=int, default=32, help='Batch size for training')
@click.option('--epochs', type=int, default=100, help='Number of epochs to train')
@click.option('--learning_rate', type=float, default=0.001, help='Learning rate')
@click.option('--weight_decay', type=float, default=1e-5, help='Weight decay (L2 penalty)')
@click.option('--patience', type=int, default=10, help='Patience for early stopping')
@click.option('--no_cuda', is_flag=True, help='Disable CUDA')
@click.option('--seed', type=int, default=42, help='Random seed')
@click.option('--output_dir', type=click.Path(), default='./output', help='Directory to save outputs')
@click.option('--accelerator', type=str, default='auto', help='Accelerator to use (auto, cpu, gpu, tpu)')
@click.option('--devices', type=int, default=1, help='Number of devices to use')
def train(data_path, num_samples, seq_length, input_dim, output_dim, model_type, hidden_channels,
         n_modes, hidden_dim, layer_dim, n_layers, dropout, batch_size, epochs, learning_rate,
         weight_decay, patience, no_cuda, seed, output_dir, accelerator, devices):
    """Train a FNO-LSTM model."""
    # Set random seed
    set_seed(seed)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create data module
    data_module = TimeSeriesDataModule(
        data_path=data_path,
        batch_size=batch_size,
        num_samples=num_samples,
        seq_length=seq_length,
        input_dim=input_dim,
        output_dim=output_dim
    )
    
    # Create model
    model = SpectralConvLSTMLightning(
        model_type=model_type,
        input_dim=input_dim,
        hidden_channels=hidden_channels,
        n_modes=n_modes,
        hidden_dim=hidden_dim,
        layer_dim=layer_dim,
        output_dim=output_dim,
        n_layers=n_layers,
        dropout=dropout,
        learning_rate=learning_rate,
        weight_decay=weight_decay
    )
    
    # Set output directory for plotting
    model.output_dir = output_dir
    
    # Create callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir,
        filename='best_model',
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min'
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=patience,
        verbose=True,
        mode='min'
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    # Create logger
    logger = TensorBoardLogger(save_dir=output_dir, name='lightning_logs')
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator=accelerator,
        devices=devices,
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
        logger=logger,
        log_every_n_steps=10,
        deterministic=True
    )
    
    # Train model
    trainer.fit(model, data_module)
    
    # Test model
    trainer.test(model=model, datamodule=data_module, ckpt_path='best')
    
    print(f"\nTraining completed. Model saved to {os.path.join(output_dir, 'best_model.ckpt')}")


@cli.command()
@click.option('--data_path', type=click.Path(exists=True), required=True, help='Path to data file (.npz format with X and y arrays)')
@click.option('--model_path', type=click.Path(exists=True), required=True, help='Path to saved model checkpoint')
@click.option('--output_dir', type=click.Path(), default='./output', help='Directory to save outputs')
@click.option('--batch_size', type=int, default=32, help='Batch size for evaluation')
@click.option('--accelerator', type=str, default='auto', help='Accelerator to use (auto, cpu, gpu, tpu)')
def evaluate(data_path, model_path, output_dir, batch_size, accelerator):
    """Evaluate a trained FNO-LSTM model."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    model = SpectralConvLSTMLightning.load_from_checkpoint(model_path)
    model.output_dir = output_dir
    
    # Create data module
    data_module = TimeSeriesDataModule(
        data_path=data_path,
        batch_size=batch_size
    )
    
    # Create trainer
    trainer = pl.Trainer(
        accelerator=accelerator,
        logger=False
    )
    
    # Test model
    trainer.test(model=model, datamodule=data_module)
    
    print(f"\nEvaluation completed. Results saved to {output_dir}")


if __name__ == "__main__":
    cli()
