"""
Hidden Markov Model implementation for regime detection
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')

from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from scipy import stats

from config.settings import HMM_CONFIG, REGIME_LABELS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RegimeHMM:
    """
    3-state Hidden Markov Model for Bull/Sideways/Bear regime detection
    """

    def __init__(self, n_components: int = 3, covariance_type: str = 'full',
                 algorithm: str = 'viterbi', n_iter: int = 1000, tol: float = 1e-6,
                 random_state: int = 42):
        """
        Initialize HMM model

        Parameters:
        -----------
        n_components : int, number of hidden states (regimes)
        covariance_type : str, covariance matrix type
        algorithm : str, decoding algorithm
        n_iter : int, maximum iterations
        tol : float, convergence tolerance
        random_state : int, random seed
        """
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.algorithm = algorithm
        self.n_iter = n_iter
        self.tol = tol
        self.random_state = random_state

        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_names = None
        self.regime_stats = {}

    def prepare_observations(self, garch_components: Dict[str, pd.Series]) -> np.ndarray:
        """
        Prepare multivariate observation sequence from GARCH outputs

        Parameters:
        -----------
        garch_components : dict containing GARCH model outputs

        Returns:
        --------
        observations : 2D array of shape (n_samples, n_features)
        """
        try:
            # Extract relevant features for HMM observations
            features = []
            self.feature_names = []

            # 1. Conditional volatility (log-transformed for stability)
            if 'conditional_volatility' in garch_components:
                vol = garch_components['conditional_volatility'].dropna()
                features.append(np.log(vol + 1e-8))  # Add small constant to avoid log(0)
                self.feature_names.append('log_volatility')

            # 2. Standardized residuals
            if 'standardized_residuals' in garch_components:
                std_resid = garch_components['standardized_residuals'].dropna()
                features.append(std_resid)
                self.feature_names.append('standardized_residuals')

            # 3. Volatility forecast (log-transformed)
            if 'volatility_forecast' in garch_components:
                vol_forecast = garch_components['volatility_forecast'].dropna()
                features.append(np.log(vol_forecast + 1e-8))
                self.feature_names.append('log_vol_forecast')

            # 4. Absolute standardized residuals (for volatility clustering)
            if 'standardized_residuals' in garch_components:
                abs_std_resid = np.abs(garch_components['standardized_residuals'].dropna())
                features.append(abs_std_resid)
                self.feature_names.append('abs_standardized_residuals')

            if not features:
                raise ValueError("No valid features found in GARCH components")

            # Combine features and align indices
            min_length = min(len(f) for f in features)
            observations = np.column_stack([f.iloc[-min_length:] for f in features])

            # Remove any remaining NaN or infinite values
            valid_mask = np.isfinite(observations).all(axis=1)
            observations = observations[valid_mask]

            logger.info(f"Prepared {observations.shape[0]} observations with {observations.shape[1]} features")
            logger.info(f"Features: {self.feature_names}")

            return observations

        except Exception as e:
            logger.error(f"Error preparing observations: {str(e)}")
            raise

    def initialize_model_parameters(self, observations: np.ndarray) -> None:
        """
        Initialize HMM parameters based on data characteristics
        """
        try:
            # Use Gaussian Mixture Model to initialize state parameters
            gmm = GaussianMixture(n_components=self.n_components, random_state=self.random_state)
            gmm.fit(observations)

            # Initialize HMM
            self.model = hmm.GaussianHMM(
                n_components=self.n_components,
                covariance_type=self.covariance_type,
                algorithm=self.algorithm,
                n_iter=self.n_iter,
                tol=self.tol,
                random_state=self.random_state
            )

            # Set initial parameters from GMM
            self.model.means_ = gmm.means_
            self.model.covars_ = gmm.covariances_

            # Initialize transition matrix (slight preference for staying in same state)
            n = self.n_components
            transition_matrix = np.full((n, n), 0.1 / (n - 1))
            np.fill_diagonal(transition_matrix, 0.9)
            self.model.transmat_ = transition_matrix

            # Initialize start probabilities (uniform)
            self.model.startprob_ = np.full(n, 1.0 / n)

            logger.info("Initialized HMM parameters using GMM")

        except Exception as e:
            logger.error(f"Error initializing model parameters: {str(e)}")
            raise

    def fit(self, garch_components: Dict[str, pd.Series]) -> 'RegimeHMM':
        """
        Fit HMM model to GARCH observations
        """
        try:
            logger.info("Fitting HMM model for regime detection...")

            # Prepare observations
            observations = self.prepare_observations(garch_components)

            # Standardize features
            observations_scaled = self.scaler.fit_transform(observations)

            # Initialize model parameters
            self.initialize_model_parameters(observations_scaled)

            # Fit the model
            self.model.fit(observations_scaled)

            self.is_fitted = True

            # Calculate regime statistics
            self._calculate_regime_statistics(observations_scaled)

            logger.info("HMM model fitted successfully")
            logger.info(f"Log-likelihood: {self.model.score(observations_scaled):.2f}")
            logger.info(f"Number of parameters: {self._count_parameters()}")

            return self

        except Exception as e:
            logger.error(f"Error fitting HMM model: {str(e)}")
            raise

    def predict_regimes(self, garch_components: Dict[str, pd.Series]) -> Dict[str, np.ndarray]:
        """
        Predict regime sequence and probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        try:
            # Prepare observations
            observations = self.prepare_observations(garch_components)
            observations_scaled = self.scaler.transform(observations)

            # Predict most likely state sequence (Viterbi algorithm)
            log_likelihood, state_sequence = self.model.decode(observations_scaled, algorithm='viterbi')

            # Calculate state probabilities (forward-backward algorithm)
            log_prob = self.model.score_samples(observations_scaled)
            state_probabilities = np.exp(self.model.predict_proba(observations_scaled))

            results = {
                'state_sequence': state_sequence,
                'state_probabilities': state_probabilities,
                'log_likelihood': log_likelihood,
                'log_prob': log_prob
            }

            logger.info(f"Predicted regimes with log-likelihood: {log_likelihood:.2f}")

            return results

        except Exception as e:
            logger.error(f"Error predicting regimes: {str(e)}")
            raise

    def forecast_regimes(self, horizon: int = 20) -> Dict[str, np.ndarray]:
        """
        Forecast regime probabilities for future periods
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        try:
            # Get transition matrix
            P = self.model.transmat_

            # Start with current state probabilities (assume uniform if not provided)
            current_probs = np.full(self.n_components, 1.0 / self.n_components)

            # Forecast probabilities for each horizon
            forecast_probs = []
            probs = current_probs.copy()

            for h in range(horizon):
                # Calculate probabilities for next period
                probs = probs @ P  # Matrix multiplication for probability evolution
                forecast_probs.append(probs.copy())

            forecast_probs = np.array(forecast_probs)

            # Calculate most likely regime for each period
            most_likely_regimes = np.argmax(forecast_probs, axis=1)

            # Calculate confidence intervals (based on probability mass)
            confidence_intervals = self._calculate_confidence_intervals(forecast_probs)

            results = {
                'forecast_probabilities': forecast_probs,
                'most_likely_regimes': most_likely_regimes,
                'confidence_intervals': confidence_intervals,
                'horizon': horizon
            }

            logger.info(f"Generated regime forecasts for {horizon} periods ahead")

            return results

        except Exception as e:
            logger.error(f"Error forecasting regimes: {str(e)}")
            raise

    def _calculate_confidence_intervals(self, forecast_probs: np.ndarray,
                                      confidence_level: float = 0.95) -> Dict:
        """
        Calculate confidence intervals for regime predictions
        """
        alpha = 1 - confidence_level
        lower_percentile = alpha / 2 * 100
        upper_percentile = (1 - alpha / 2) * 100

        # For each time period, calculate regime probability CI
        ci_results = {}

        for regime in range(self.n_components):
            regime_probs = forecast_probs[:, regime]
            ci_results[f'regime_{regime}_lower'] = np.full_like(regime_probs, lower_percentile)
            ci_results[f'regime_{regime}_upper'] = np.full_like(regime_probs, upper_percentile)

        return ci_results

    def _calculate_regime_statistics(self, observations: np.ndarray) -> None:
        """
        Calculate statistics for each regime
        """
        # Get most likely state sequence
        state_sequence = self.model.predict(observations)

        self.regime_stats = {}

        for regime in range(self.n_components):
            regime_mask = state_sequence == regime
            regime_obs = observations[regime_mask]

            if len(regime_obs) > 0:
                self.regime_stats[regime] = {
                    'count': len(regime_obs),
                    'frequency': len(regime_obs) / len(observations),
                    'mean_features': np.mean(regime_obs, axis=0),
                    'std_features': np.std(regime_obs, axis=0),
                    'duration_avg': self._calculate_average_duration(state_sequence, regime)
                }

        logger.info("Calculated regime statistics")

    def _calculate_average_duration(self, state_sequence: np.ndarray, regime: int) -> float:
        """
        Calculate average duration of specified regime
        """
        durations = []
        current_duration = 0

        for state in state_sequence:
            if state == regime:
                current_duration += 1
            else:
                if current_duration > 0:
                    durations.append(current_duration)
                    current_duration = 0

        # Add final duration if sequence ends in the regime
        if current_duration > 0:
            durations.append(current_duration)

        return np.mean(durations) if durations else 0

    def _count_parameters(self) -> int:
        """
        Count number of model parameters
        """
        n = self.n_components
        d = len(self.feature_names) if self.feature_names else 2

        # Transition matrix parameters (n x n - n for normalization)
        trans_params = n * (n - 1)

        # Initial state probabilities (n - 1 for normalization)
        init_params = n - 1

        # Emission parameters (means and covariances)
        if self.covariance_type == 'full':
            # Mean: n * d, Covariance: n * d * (d + 1) / 2
            emission_params = n * d + n * d * (d + 1) // 2
        elif self.covariance_type == 'diag':
            # Mean: n * d, Diagonal covariance: n * d
            emission_params = n * d + n * d
        else:  # spherical or tied
            emission_params = n * d + n

        return trans_params + init_params + emission_params

    def get_model_summary(self) -> Dict:
        """
        Get comprehensive model summary
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        summary = {
            'n_components': self.n_components,
            'covariance_type': self.covariance_type,
            'n_parameters': self._count_parameters(),
            'feature_names': self.feature_names,
            'regime_statistics': self.regime_stats,
            'transition_matrix': self.model.transmat_.tolist(),
            'initial_probabilities': self.model.startprob_.tolist(),
        }

        # Add regime interpretations
        summary['regime_interpretations'] = self._interpret_regimes()

        return summary

    def _interpret_regimes(self) -> Dict:
        """
        Interpret regimes based on their characteristics
        """
        interpretations = {}

        if not self.regime_stats:
            return interpretations

        # Sort regimes by volatility (first feature is typically log volatility)
        regimes_by_vol = sorted(self.regime_stats.keys(),
                              key=lambda r: self.regime_stats[r]['mean_features'][0])

        # Assign interpretations based on volatility ranking
        if len(regimes_by_vol) >= 3:
            interpretations[regimes_by_vol[0]] = "Low Volatility (Bull/Stable)"
            interpretations[regimes_by_vol[1]] = "Medium Volatility (Sideways)"
            interpretations[regimes_by_vol[2]] = "High Volatility (Bear/Crisis)"
        else:
            for i, regime in enumerate(regimes_by_vol):
                interpretations[regime] = f"Regime {regime}"

        return interpretations

    def calculate_regime_metrics(self, returns: pd.Series, state_sequence: np.ndarray) -> Dict:
        """
        Calculate performance metrics for each regime
        """
        metrics = {}

        for regime in range(self.n_components):
            regime_mask = state_sequence == regime
            regime_returns = returns[regime_mask]

            if len(regime_returns) > 0:
                metrics[regime] = {
                    'count': len(regime_returns),
                    'mean_return': regime_returns.mean(),
                    'volatility': regime_returns.std(),
                    'sharpe_ratio': regime_returns.mean() / regime_returns.std() if regime_returns.std() > 0 else 0,
                    'skewness': regime_returns.skew(),
                    'kurtosis': regime_returns.kurtosis(),
                    'min_return': regime_returns.min(),
                    'max_return': regime_returns.max(),
                    'frequency': len(regime_returns) / len(returns)
                }

        return metrics


def main():
    """
    Example usage of RegimeHMM
    """
    # This would typically be called with actual GARCH components
    # For demonstration, we'll create some dummy data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=1000, freq='D')

    # Simulate GARCH-like components
    dummy_components = {
        'conditional_volatility': pd.Series(np.random.gamma(2, 0.01, 1000), index=dates),
        'standardized_residuals': pd.Series(np.random.normal(0, 1, 1000), index=dates),
        'volatility_forecast': pd.Series(np.random.gamma(2, 0.01, 1000), index=dates)
    }

    # Initialize and fit HMM
    hmm_model = RegimeHMM()
    hmm_model.fit(dummy_components)

    # Predict regimes
    predictions = hmm_model.predict_regimes(dummy_components)

    # Generate forecasts
    forecasts = hmm_model.forecast_regimes(horizon=20)

    # Get model summary
    summary = hmm_model.get_model_summary()

    print("HMM Model Summary:")
    for key, value in summary.items():
        if key != 'regime_statistics':
            print(f"{key}: {value}")

    return hmm_model, predictions, forecasts


if __name__ == "__main__":
    main()