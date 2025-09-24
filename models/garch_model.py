"""
GARCH(1,1) model implementation for volatility modeling
"""

import numpy as np
import pandas as pd
import logging
from typing import Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

from arch import arch_model
from arch.univariate import GARCH, ConstantMean, ZeroMean
from scipy import stats

from config.settings import GARCH_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GARCHModel:
    """
    GARCH(1,1) model for volatility forecasting and standardized residuals
    """

    def __init__(self, mean_model: str = 'Zero', vol_model: str = 'GARCH',
                 dist: str = 'Normal', p: int = 1, q: int = 1):
        """
        Initialize GARCH model

        Parameters:
        -----------
        mean_model : str, mean model specification
        vol_model : str, volatility model specification
        dist : str, error distribution
        p : int, GARCH lag order
        q : int, ARCH lag order
        """
        self.mean_model = mean_model
        self.vol_model = vol_model
        self.dist = dist
        self.p = p
        self.q = q
        self.model = None
        self.fitted_model = None
        self.is_fitted = False

    def prepare_data(self, returns: pd.Series) -> pd.Series:
        """
        Prepare returns data for GARCH modeling
        """
        # Convert to percentage returns for better numerical stability
        returns_pct = returns * 100

        # Remove extreme outliers (beyond 5 standard deviations)
        std_threshold = 5
        mean_ret = returns_pct.mean()
        std_ret = returns_pct.std()
        outlier_mask = np.abs(returns_pct - mean_ret) > (std_threshold * std_ret)

        if outlier_mask.any():
            logger.warning(f"Removing {outlier_mask.sum()} extreme outliers")
            returns_pct = returns_pct[~outlier_mask]

        # Ensure we have enough data
        if len(returns_pct) < 100:
            raise ValueError("Insufficient data for GARCH modeling (need at least 100 observations)")

        return returns_pct

    def fit(self, returns: pd.Series) -> 'GARCHModel':
        """
        Fit GARCH model to returns data
        """
        try:
            logger.info("Fitting GARCH(1,1) model...")

            # Prepare data
            prepared_returns = self.prepare_data(returns)

            # Create GARCH model
            if self.mean_model.lower() == 'zero':
                self.model = arch_model(
                    prepared_returns,
                    mean='zero',
                    vol='GARCH',
                    p=self.p,
                    q=self.q,
                    dist=self.dist.lower(),
                    rescale=False
                )
            else:
                self.model = arch_model(
                    prepared_returns,
                    mean='constant',
                    vol='GARCH',
                    p=self.p,
                    q=self.q,
                    dist=self.dist.lower(),
                    rescale=False
                )

            # Fit the model
            self.fitted_model = self.model.fit(disp='off', show_warning=False)
            self.is_fitted = True

            logger.info("GARCH model fitted successfully")
            logger.info(f"Log-likelihood: {self.fitted_model.loglikelihood:.2f}")
            logger.info(f"AIC: {self.fitted_model.aic:.2f}")
            logger.info(f"BIC: {self.fitted_model.bic:.2f}")

            return self

        except Exception as e:
            logger.error(f"Error fitting GARCH model: {str(e)}")
            raise

    def get_model_summary(self) -> str:
        """
        Get detailed model summary
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        return str(self.fitted_model.summary())

    def extract_components(self, returns: pd.Series) -> Dict[str, pd.Series]:
        """
        Extract key GARCH model components for HMM observations

        Returns:
        --------
        dict containing:
        - conditional_volatility: Fitted conditional volatility
        - standardized_residuals: Standardized residuals
        - volatility_forecast: One-step-ahead volatility forecasts
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        try:
            # Prepare data (same as used in fitting)
            prepared_returns = self.prepare_data(returns)

            # Re-fit on the specific returns data for component extraction
            temp_model = arch_model(
                prepared_returns,
                mean='zero' if self.mean_model.lower() == 'zero' else 'constant',
                vol='GARCH',
                p=self.p,
                q=self.q,
                dist=self.dist.lower(),
                rescale=False
            )
            temp_result = temp_model.fit(disp='off', show_warning=False)

            # Extract components using correct attribute names
            conditional_vol = temp_result.conditional_volatility
            residuals = temp_result.resid

            # Calculate standardized residuals
            standardized_residuals = residuals / conditional_vol

            # Generate one-step-ahead volatility forecasts
            vol_forecast = self._generate_volatility_forecasts(prepared_returns, temp_result)

            # Align with original returns index (account for removed outliers)
            original_index = returns.index

            # Create output series aligned with original data
            result = {
                'conditional_volatility': self._align_series(conditional_vol, original_index),
                'standardized_residuals': self._align_series(standardized_residuals, original_index),
                'volatility_forecast': self._align_series(vol_forecast, original_index),
                'residuals': self._align_series(residuals, original_index)
            }

            logger.info("Extracted GARCH components successfully")

            return result

        except Exception as e:
            logger.error(f"Error extracting GARCH components: {str(e)}")
            raise

    def _generate_volatility_forecasts(self, returns: pd.Series, fitted_result=None) -> pd.Series:
        """
        Generate rolling one-step-ahead volatility forecasts
        """
        if fitted_result is None:
            fitted_result = self.fitted_model

        forecasts = []

        # Use the fitted parameters for forecasting
        params = fitted_result.params

        # Get the fitted values to use as starting values
        conditional_vol = fitted_result.conditional_volatility
        residuals = fitted_result.resid

        # Generate forecasts
        for i in range(len(returns)):
            if i == 0:
                # Use unconditional volatility for first forecast
                forecast = np.sqrt(params['omega'] / (1 - params['alpha[1]'] - params['beta[1]']))
            else:
                # Use GARCH(1,1) formula: σ²(t+1) = ω + α*ε²(t) + β*σ²(t)
                omega = params['omega']
                alpha = params['alpha[1]']
                beta = params['beta[1]']

                prev_resid_sq = residuals.iloc[i-1]**2 if i-1 < len(residuals) else 0
                prev_vol_sq = conditional_vol.iloc[i-1]**2 if i-1 < len(conditional_vol) else forecast**2

                forecast = np.sqrt(omega + alpha * prev_resid_sq + beta * prev_vol_sq)

            forecasts.append(forecast)

        return pd.Series(forecasts, index=returns.index)

    def _align_series(self, series: pd.Series, target_index: pd.Index) -> pd.Series:
        """
        Align series with target index, filling missing values appropriately
        """
        # Reindex to target and forward fill missing values
        aligned = series.reindex(target_index)

        # Forward fill and backward fill to handle missing values
        aligned = aligned.fillna(method='ffill').fillna(method='bfill')

        return aligned

    def forecast_volatility(self, horizon: int = 1) -> Dict:
        """
        Generate volatility forecasts for specified horizon
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        try:
            forecasts = self.fitted_model.forecast(horizon=horizon, reindex=False)

            return {
                'mean': forecasts.mean.iloc[-1, :],
                'variance': forecasts.variance.iloc[-1, :],
                'volatility': np.sqrt(forecasts.variance.iloc[-1, :])
            }

        except Exception as e:
            logger.error(f"Error generating volatility forecasts: {str(e)}")
            raise

    def model_diagnostics(self) -> Dict:
        """
        Perform model diagnostics and tests
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        diagnostics = {}

        try:
            # Standardized residuals
            conditional_vol = self.fitted_model.conditional_volatility
            residuals = self.fitted_model.resid
            std_resid = residuals / conditional_vol

            # Ljung-Box test for serial correlation in residuals
            from statsmodels.stats.diagnostic import acorr_ljungbox
            lb_test = acorr_ljungbox(std_resid, lags=10, return_df=True)
            diagnostics['ljung_box_pvalue'] = lb_test['lb_pvalue'].iloc[-1]

            # Ljung-Box test for serial correlation in squared residuals
            lb_test_sq = acorr_ljungbox(std_resid**2, lags=10, return_df=True)
            diagnostics['ljung_box_squared_pvalue'] = lb_test_sq['lb_pvalue'].iloc[-1]

            # Jarque-Bera test for normality
            jb_stat, jb_pvalue = stats.jarque_bera(std_resid.dropna())
            diagnostics['jarque_bera_pvalue'] = jb_pvalue

            # Basic statistics
            diagnostics['log_likelihood'] = self.fitted_model.loglikelihood
            diagnostics['aic'] = self.fitted_model.aic
            diagnostics['bic'] = self.fitted_model.bic
            diagnostics['num_observations'] = self.fitted_model.nobs

            return diagnostics

        except Exception as e:
            logger.error(f"Error in model diagnostics: {str(e)}")
            return diagnostics

    def get_parameters(self) -> Dict:
        """
        Get fitted GARCH parameters
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        params = self.fitted_model.params.to_dict()

        # Add derived statistics
        if 'alpha[1]' in params and 'beta[1]' in params:
            params['persistence'] = params['alpha[1]'] + params['beta[1]']
            params['unconditional_vol'] = np.sqrt(params['omega'] / (1 - params['persistence']))

        return params


def main():
    """
    Example usage of GARCH model
    """
    # This would typically be called with actual return data
    # For demonstration, we'll create some dummy data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=1000, freq='D')
    returns = pd.Series(np.random.normal(0, 0.02, 1000), index=dates)

    # Initialize and fit GARCH model
    garch = GARCHModel()
    garch.fit(returns)

    # Extract components
    components = garch.extract_components(returns)

    # Get diagnostics
    diagnostics = garch.model_diagnostics()

    print("GARCH Model Diagnostics:")
    for key, value in diagnostics.items():
        print(f"{key}: {value}")

    return garch, components


if __name__ == "__main__":
    main()