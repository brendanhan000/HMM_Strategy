"""
Enhanced GARCH model with better feature engineering for regime detection
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from arch import arch_model
from scipy import stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedGARCHModel:
    """
    Enhanced GARCH model with improved feature extraction for HMM
    """

    def __init__(self, p: int = 1, q: int = 1):
        self.p = p
        self.q = q
        self.model = None
        self.fitted_model = None
        self.is_fitted = False

    def fit(self, returns: pd.Series) -> 'EnhancedGARCHModel':
        """Fit GARCH model with robust error handling"""
        try:
            logger.info("Fitting Enhanced GARCH model...")

            # Clean and prepare returns
            clean_returns = returns.dropna() * 100  # Convert to percentage for numerical stability

            # Remove extreme outliers (beyond 3 standard deviations)
            mean_ret = clean_returns.mean()
            std_ret = clean_returns.std()
            outlier_mask = np.abs(clean_returns - mean_ret) <= (3 * std_ret)
            clean_returns = clean_returns[outlier_mask]

            logger.info(f"Using {len(clean_returns)} observations after cleaning")

            # Fit GARCH(1,1) model
            self.model = arch_model(clean_returns, mean='constant', vol='GARCH', p=self.p, q=self.q)
            self.fitted_model = self.model.fit(disp='off', show_warning=False)
            self.is_fitted = True

            logger.info(f"GARCH fitted - LL: {self.fitted_model.loglikelihood:.2f}")
            return self

        except Exception as e:
            logger.error(f"Error fitting GARCH model: {str(e)}")
            raise

    def extract_regime_features(self, returns: pd.Series) -> pd.DataFrame:
        """
        Extract comprehensive features for regime detection
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        try:
            # Clean returns the same way as in fitting
            clean_returns = returns.dropna() * 100
            mean_ret = clean_returns.mean()
            std_ret = clean_returns.std()
            outlier_mask = np.abs(clean_returns - mean_ret) <= (3 * std_ret)
            clean_returns = clean_returns[outlier_mask]

            # Refit model on the specific data to get components
            temp_model = arch_model(clean_returns, mean='constant', vol='GARCH', p=self.p, q=self.q)
            temp_result = temp_model.fit(disp='off', show_warning=False)

            # Extract basic GARCH components
            conditional_vol = temp_result.conditional_volatility
            residuals = temp_result.resid
            standardized_residuals = residuals / conditional_vol

            # Create comprehensive feature set
            features_df = pd.DataFrame(index=clean_returns.index)

            # Feature 1: Log volatility (captures volatility regime)
            features_df['log_volatility'] = np.log(conditional_vol)

            # Feature 2: Standardized residuals (captures return regime)
            features_df['std_residuals'] = standardized_residuals

            # Feature 3: Return momentum (5-day rolling average)
            features_df['return_momentum'] = clean_returns.rolling(5).mean()

            # Feature 4: Volatility momentum (change in volatility)
            features_df['vol_momentum'] = conditional_vol.pct_change(5)

            # Feature 5: Regime indicator based on returns and volatility
            # Bull: High returns, Low volatility
            # Bear: Low returns, High volatility
            # Sideways: Medium returns, Medium volatility
            vol_percentile = conditional_vol.rolling(60).rank(pct=True)
            ret_percentile = clean_returns.rolling(60).rank(pct=True)
            features_df['regime_score'] = ret_percentile - vol_percentile  # High = bullish

            # Drop NaN values
            features_df = features_df.dropna()

            # Standardize features for HMM
            for col in features_df.columns:
                features_df[col] = (features_df[col] - features_df[col].mean()) / features_df[col].std()

            logger.info(f"Extracted {len(features_df)} feature observations with {len(features_df.columns)} features")

            # Align with original returns index
            aligned_features = features_df.reindex(returns.index).fillna(method='ffill').fillna(method='bfill')

            return aligned_features

        except Exception as e:
            logger.error(f"Error extracting regime features: {str(e)}")
            raise

    def get_volatility_forecast(self, returns: pd.Series, horizon: int = 1) -> pd.Series:
        """Get volatility forecasts for position sizing"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        try:
            # Use the model's forecast method
            forecast = self.fitted_model.forecast(horizon=horizon, reindex=False)
            vol_forecast = np.sqrt(forecast.variance.iloc[-1, :])

            # Convert back from percentage and annualize
            vol_forecast = vol_forecast / 100 * np.sqrt(252)

            # Create series aligned with returns
            forecast_series = pd.Series(vol_forecast.iloc[0], index=returns.index[-len(vol_forecast):])

            # Forward fill to align with full returns series
            aligned_forecast = forecast_series.reindex(returns.index).fillna(method='ffill')

            return aligned_forecast

        except Exception as e:
            logger.warning(f"Error in volatility forecast: {str(e)}")
            # Return simple rolling volatility as fallback
            return returns.rolling(20).std() * np.sqrt(252)


def main():
    """Test the enhanced GARCH model"""
    # Create test data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=500, freq='D')
    returns = pd.Series(np.random.normal(0, 0.02, 500), index=dates)

    # Test the model
    garch = EnhancedGARCHModel()
    garch.fit(returns)

    features = garch.extract_regime_features(returns)
    vol_forecast = garch.get_volatility_forecast(returns)

    print("Enhanced GARCH model test completed successfully!")
    print(f"Features shape: {features.shape}")
    print(f"Volatility forecast shape: {vol_forecast.shape}")

    return garch, features, vol_forecast


if __name__ == "__main__":
    main()