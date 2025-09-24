"""
Enhanced HMM model with improved regime identification and interpretation
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')

from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedRegimeHMM:
    """
    Enhanced HMM with intelligent regime interpretation
    """

    def __init__(self, n_components: int = 3, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.regime_mapping = {}  # Maps HMM states to Bull/Sideways/Bear
        self.regime_characteristics = {}

    def fit(self, features_df: pd.DataFrame, returns: pd.Series) -> 'EnhancedRegimeHMM':
        """
        Fit HMM model and intelligently identify regimes
        """
        try:
            logger.info("Fitting Enhanced HMM model...")

            # Prepare observations
            observations = features_df.values
            observations_scaled = self.scaler.fit_transform(observations)

            # Initialize HMM model
            self.model = hmm.GaussianHMM(
                n_components=self.n_components,
                covariance_type='full',
                n_iter=1000,
                random_state=self.random_state
            )

            # Use K-means for better initialization
            kmeans = KMeans(n_clusters=self.n_components, random_state=self.random_state)
            kmeans_labels = kmeans.fit_predict(observations_scaled)

            # Initialize HMM parameters
            self.model.startprob_ = np.ones(self.n_components) / self.n_components

            # Initialize transition matrix with persistence bias
            self.model.transmat_ = np.full((self.n_components, self.n_components), 0.05)
            np.fill_diagonal(self.model.transmat_, 0.9)
            self.model.transmat_ = self.model.transmat_ / self.model.transmat_.sum(axis=1, keepdims=True)

            # Initialize means and covariances from K-means
            self.model.means_ = kmeans.cluster_centers_
            self.model.covars_ = np.array([np.cov(observations_scaled[kmeans_labels == i].T) + np.eye(observations_scaled.shape[1]) * 0.01
                                         for i in range(self.n_components)])

            # Fit the model
            self.model.fit(observations_scaled)
            self.is_fitted = True

            # Decode most likely state sequence
            log_prob, state_sequence = self.model.decode(observations_scaled, algorithm='viterbi')

            # Intelligently map states to regimes based on market characteristics
            self._identify_regimes(state_sequence, returns, features_df)

            logger.info(f"HMM fitted with log-likelihood: {log_prob:.2f}")
            logger.info(f"Regime mapping: {self.regime_mapping}")

            return self

        except Exception as e:
            logger.error(f"Error fitting Enhanced HMM: {str(e)}")
            raise

    def _identify_regimes(self, state_sequence: np.ndarray, returns: pd.Series, features_df: pd.DataFrame):
        """
        Intelligently identify which HMM state corresponds to which market regime
        """
        # Align data
        min_len = min(len(state_sequence), len(returns), len(features_df))
        states = state_sequence[:min_len]
        rets = returns.iloc[:min_len]
        features = features_df.iloc[:min_len]

        regime_stats = {}

        for state in range(self.n_components):
            state_mask = states == state
            if np.sum(state_mask) > 0:
                state_returns = rets[state_mask]
                state_features = features[state_mask]

                # Calculate regime characteristics
                avg_return = state_returns.mean()
                volatility = state_returns.std()
                sharpe = avg_return / volatility if volatility > 0 else 0

                # Additional characteristics from features
                avg_vol_feature = state_features['log_volatility'].mean() if 'log_volatility' in state_features.columns else 0
                avg_momentum = state_features['return_momentum'].mean() if 'return_momentum' in state_features.columns else 0
                regime_score = state_features['regime_score'].mean() if 'regime_score' in state_features.columns else 0

                regime_stats[state] = {
                    'avg_return': avg_return,
                    'volatility': volatility,
                    'sharpe': sharpe,
                    'avg_vol_feature': avg_vol_feature,
                    'avg_momentum': avg_momentum,
                    'regime_score': regime_score,
                    'frequency': np.sum(state_mask) / len(states)
                }

        # Intelligent regime mapping based on multiple criteria
        # Sort states by regime score (combination of returns and volatility characteristics)
        sorted_states = sorted(regime_stats.keys(),
                             key=lambda s: regime_stats[s]['regime_score'],
                             reverse=True)

        if len(sorted_states) >= 3:
            # Highest regime score = Bull (high returns, low vol)
            self.regime_mapping[sorted_states[0]] = 'Bull'
            # Lowest regime score = Bear (low returns, high vol)
            self.regime_mapping[sorted_states[-1]] = 'Bear'
            # Middle = Sideways
            self.regime_mapping[sorted_states[1]] = 'Sideways'
        else:
            # Fallback mapping
            for i, state in enumerate(sorted_states):
                if i == 0:
                    self.regime_mapping[state] = 'Bull'
                elif i == len(sorted_states) - 1:
                    self.regime_mapping[state] = 'Bear'
                else:
                    self.regime_mapping[state] = 'Sideways'

        self.regime_characteristics = regime_stats

        # Log regime characteristics
        for state, regime_name in self.regime_mapping.items():
            stats = regime_stats[state]
            logger.info(f"{regime_name} (State {state}): Return={stats['avg_return']:.4f}, "
                       f"Vol={stats['volatility']:.4f}, Sharpe={stats['sharpe']:.2f}, "
                       f"Freq={stats['frequency']:.1%}")

    def predict_regimes(self, features_df: pd.DataFrame) -> Dict:
        """
        Predict regime sequence with enhanced mapping
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        try:
            observations = features_df.values
            observations_scaled = self.scaler.transform(observations)

            # Get state sequence and probabilities
            log_prob, state_sequence = self.model.decode(observations_scaled, algorithm='viterbi')
            state_probabilities = self.model.predict_proba(observations_scaled)

            # Map states to regime names
            regime_sequence = np.array([self.regime_mapping.get(state, f'State_{state}')
                                      for state in state_sequence])

            # Map probabilities to regimes
            regime_probs = pd.DataFrame(index=features_df.index)
            for state, regime_name in self.regime_mapping.items():
                if state < state_probabilities.shape[1]:
                    regime_probs[regime_name] = state_probabilities[:, state]

            results = {
                'state_sequence': state_sequence,
                'regime_sequence': regime_sequence,
                'state_probabilities': state_probabilities,
                'regime_probabilities': regime_probs,
                'log_likelihood': log_prob
            }

            return results

        except Exception as e:
            logger.error(f"Error predicting regimes: {str(e)}")
            raise

    def get_regime_positions(self) -> Dict[str, float]:
        """
        Get optimal position sizes for each regime based on historical performance
        """
        if not self.regime_characteristics:
            # Default positions
            return {'Bull': 1.0, 'Sideways': 0.0, 'Bear': -1.0}

        positions = {}
        for state, regime_name in self.regime_mapping.items():
            stats = self.regime_characteristics[state]

            if regime_name == 'Bull':
                # Aggressive long position if strong bull characteristics
                if stats['sharpe'] > 1.0:
                    positions[regime_name] = 1.0
                elif stats['avg_return'] > 0:
                    positions[regime_name] = 0.8
                else:
                    positions[regime_name] = 0.5

            elif regime_name == 'Bear':
                # Short position based on negative characteristics
                if stats['sharpe'] < -0.5:
                    positions[regime_name] = -1.0
                elif stats['avg_return'] < 0:
                    positions[regime_name] = -0.8
                else:
                    positions[regime_name] = -0.3

            else:  # Sideways
                # Small position or cash based on slight directional bias
                if stats['avg_return'] > 0.001:
                    positions[regime_name] = 0.2
                elif stats['avg_return'] < -0.001:
                    positions[regime_name] = -0.2
                else:
                    positions[regime_name] = 0.0

        logger.info(f"Optimized regime positions: {positions}")
        return positions

    def calculate_regime_persistence(self) -> Dict:
        """Calculate regime persistence metrics"""
        if not self.is_fitted:
            return {}

        transition_matrix = self.model.transmat_
        persistence = {}

        for state, regime_name in self.regime_mapping.items():
            self_transition = transition_matrix[state, state]
            expected_duration = 1 / (1 - self_transition) if self_transition < 1 else float('inf')
            persistence[regime_name] = {
                'self_transition_prob': self_transition,
                'expected_duration': expected_duration
            }

        return persistence

    def get_model_summary(self) -> Dict:
        """Get comprehensive model summary"""
        if not self.is_fitted:
            return {}

        summary = {
            'n_components': self.n_components,
            'regime_mapping': self.regime_mapping,
            'regime_characteristics': self.regime_characteristics,
            'transition_matrix': self.model.transmat_.tolist(),
            'regime_persistence': self.calculate_regime_persistence(),
            'optimal_positions': self.get_regime_positions()
        }

        return summary


def main():
    """Test the enhanced HMM model"""
    # Create test data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=500, freq='D')

    # Create realistic features
    features = pd.DataFrame(index=dates)
    features['log_volatility'] = np.random.normal(0, 1, 500)
    features['std_residuals'] = np.random.normal(0, 1, 500)
    features['return_momentum'] = np.random.normal(0, 1, 500)
    features['vol_momentum'] = np.random.normal(0, 1, 500)
    features['regime_score'] = np.random.normal(0, 1, 500)

    returns = pd.Series(np.random.normal(0.001, 0.02, 500), index=dates)

    # Test the model
    hmm_model = EnhancedRegimeHMM()
    hmm_model.fit(features, returns)

    predictions = hmm_model.predict_regimes(features)
    summary = hmm_model.get_model_summary()

    print("Enhanced HMM model test completed successfully!")
    print(f"Regime mapping: {summary['regime_mapping']}")
    print(f"Optimal positions: {summary['optimal_positions']}")

    return hmm_model, predictions, summary


if __name__ == "__main__":
    main()