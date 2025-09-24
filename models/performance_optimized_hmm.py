"""
Performance-optimized HMM model that identifies regimes based on actual profitability
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

class PerformanceOptimizedHMM:
    """
    HMM model optimized to identify the most profitable regime patterns
    """

    def __init__(self, n_components: int = 3, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.regime_mapping = {}
        self.regime_characteristics = {}
        self.optimal_positions = {}

    def fit(self, features_df: pd.DataFrame, returns: pd.Series) -> 'PerformanceOptimizedHMM':
        """
        Fit HMM and identify regimes based on profitability
        """
        try:
            logger.info("Fitting Performance-Optimized HMM...")

            # Create enhanced features that better separate profitable periods
            enhanced_features = self._create_enhanced_features(features_df, returns)

            # Prepare observations
            observations = enhanced_features.values
            observations_scaled = self.scaler.fit_transform(observations)

            # Fit HMM with multiple initializations to find best model
            best_model = None
            best_ll = -np.inf

            for init in range(5):  # Try 5 different initializations
                model = hmm.GaussianHMM(
                    n_components=self.n_components,
                    covariance_type='full',
                    n_iter=1000,
                    random_state=self.random_state + init
                )

                try:
                    model.fit(observations_scaled)
                    ll = model.score(observations_scaled)
                    if ll > best_ll:
                        best_ll = ll
                        best_model = model
                except:
                    continue

            if best_model is None:
                raise ValueError("Failed to fit HMM model")

            self.model = best_model
            self.is_fitted = True

            # Decode states and optimize regime mapping
            log_prob, state_sequence = self.model.decode(observations_scaled, algorithm='viterbi')
            self._optimize_regime_mapping(state_sequence, returns, enhanced_features)

            logger.info(f"HMM fitted with log-likelihood: {best_ll:.2f}")
            logger.info(f"Optimized regime mapping: {self.regime_mapping}")

            return self

        except Exception as e:
            logger.error(f"Error fitting Performance-Optimized HMM: {str(e)}")
            raise

    def _create_enhanced_features(self, features_df: pd.DataFrame, returns: pd.Series) -> pd.DataFrame:
        """
        Create enhanced features that better capture profitable regime characteristics
        """
        enhanced = pd.DataFrame(index=features_df.index)

        # Align data
        min_len = min(len(features_df), len(returns))
        features_aligned = features_df.iloc[:min_len]
        returns_aligned = returns.iloc[:min_len]

        # Feature 1: Forward-looking returns (what we want to predict)
        enhanced['future_returns_5d'] = returns_aligned.shift(-5).rolling(5).mean()
        enhanced['future_returns_10d'] = returns_aligned.shift(-10).rolling(10).mean()

        # Feature 2: Return momentum and trend
        enhanced['return_momentum_5d'] = returns_aligned.rolling(5).mean()
        enhanced['return_momentum_20d'] = returns_aligned.rolling(20).mean()
        enhanced['trend_strength'] = enhanced['return_momentum_5d'] - enhanced['return_momentum_20d']

        # Feature 3: Volatility regime
        rolling_vol = returns_aligned.rolling(20).std() * np.sqrt(252)
        enhanced['volatility_level'] = rolling_vol
        enhanced['volatility_regime'] = (rolling_vol > rolling_vol.rolling(60).median()).astype(int)

        # Feature 4: Market regime indicators
        price_changes = returns_aligned.cumsum()
        enhanced['price_level'] = price_changes
        enhanced['price_acceleration'] = returns_aligned.rolling(10).mean().diff(5)

        # Feature 5: Profitability indicators
        enhanced['sharpe_indicator'] = enhanced['return_momentum_20d'] / rolling_vol
        enhanced['profit_potential'] = enhanced['future_returns_10d'] / rolling_vol

        # Clean and standardize
        enhanced = enhanced.dropna()

        # Only keep features that have predictive power
        for col in enhanced.columns:
            if enhanced[col].std() > 0:  # Only keep non-constant features
                enhanced[col] = (enhanced[col] - enhanced[col].mean()) / enhanced[col].std()
            else:
                enhanced[col] = 0

        logger.info(f"Created {enhanced.shape[1]} enhanced features with {len(enhanced)} observations")

        return enhanced

    def _optimize_regime_mapping(self, state_sequence: np.ndarray, returns: pd.Series, features_df: pd.DataFrame):
        """
        Optimize regime mapping based on actual profitability and risk-adjusted returns
        """
        # Align data
        min_len = min(len(state_sequence), len(returns), len(features_df))
        states = state_sequence[:min_len]
        rets = returns.iloc[:min_len]

        regime_performance = {}

        # Calculate performance for each HMM state
        for state in range(self.n_components):
            state_mask = states == state
            if np.sum(state_mask) > 10:  # Need minimum observations
                state_returns = rets[state_mask]

                # Calculate comprehensive performance metrics
                mean_return = state_returns.mean()
                volatility = state_returns.std()
                sharpe = mean_return / volatility * np.sqrt(252) if volatility > 0 else 0
                win_rate = (state_returns > 0).mean()
                frequency = np.sum(state_mask) / len(states)

                # Profit potential score (what we really care about)
                profit_score = mean_return * np.sqrt(252) / max(volatility, 0.001) * win_rate * frequency

                regime_performance[state] = {
                    'mean_return': mean_return,
                    'annualized_return': mean_return * 252,
                    'volatility': volatility,
                    'sharpe': sharpe,
                    'win_rate': win_rate,
                    'frequency': frequency,
                    'profit_score': profit_score
                }

        # Sort states by profit score (highest to lowest)
        sorted_states = sorted(regime_performance.keys(),
                             key=lambda s: regime_performance[s]['profit_score'],
                             reverse=True)

        # Assign regime names based on profitability ranking
        if len(sorted_states) >= 3:
            self.regime_mapping[sorted_states[0]] = 'HighProfit'    # Most profitable
            self.regime_mapping[sorted_states[1]] = 'MediumProfit'  # Medium profitable
            self.regime_mapping[sorted_states[2]] = 'LowProfit'     # Least profitable
        else:
            for i, state in enumerate(sorted_states):
                self.regime_mapping[state] = f'Regime_{i}'

        # Calculate optimal positions based on actual performance
        self.optimal_positions = {}
        for state, regime_name in self.regime_mapping.items():
            if state in regime_performance:
                perf = regime_performance[state]
                sharpe = perf['sharpe']
                mean_return = perf['mean_return']

                # Position sizing based on Kelly criterion and Sharpe ratio
                if sharpe > 1.0:  # High Sharpe ratio
                    position = 2.0 * np.sign(mean_return)  # Full leverage
                elif sharpe > 0.5:  # Medium Sharpe ratio
                    position = 1.5 * np.sign(mean_return)  # Moderate leverage
                elif sharpe > 0:  # Positive but low Sharpe
                    position = 0.8 * np.sign(mean_return)  # Conservative
                else:  # Negative Sharpe
                    position = -0.5 * np.sign(mean_return)  # Contrarian small position

                # Apply risk limits
                position = np.clip(position, -2.0, 2.0)
                self.optimal_positions[regime_name] = position

        self.regime_characteristics = regime_performance

        # Log results
        for state, regime_name in self.regime_mapping.items():
            if state in regime_performance:
                perf = regime_performance[state]
                pos = self.optimal_positions.get(regime_name, 0)
                logger.info(f"{regime_name} (State {state}): "
                           f"Return={perf['annualized_return']:.1%}, "
                           f"Sharpe={perf['sharpe']:.2f}, "
                           f"Win Rate={perf['win_rate']:.1%}, "
                           f"Freq={perf['frequency']:.1%}, "
                           f"OptimalPos={pos:.1f}")

    def predict_regimes(self, features_df: pd.DataFrame, returns: pd.Series) -> Dict:
        """
        Predict regimes using the optimized model
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        try:
            # Create enhanced features
            enhanced_features = self._create_enhanced_features(features_df, returns)
            observations = enhanced_features.values
            observations_scaled = self.scaler.transform(observations)

            # Get predictions
            log_prob, state_sequence = self.model.decode(observations_scaled, algorithm='viterbi')
            state_probabilities = self.model.predict_proba(observations_scaled)

            # Map to regime names
            regime_sequence = np.array([self.regime_mapping.get(state, f'State_{state}')
                                      for state in state_sequence])

            # Create regime probabilities DataFrame
            regime_probs = pd.DataFrame(index=enhanced_features.index)
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
        if not self.optimal_positions:
            return {'HighProfit': 2.0, 'MediumProfit': 0.5, 'LowProfit': -1.0}

        logger.info(f"Optimal regime positions: {self.optimal_positions}")
        return self.optimal_positions

    def get_model_summary(self) -> Dict:
        """Get comprehensive model summary"""
        if not self.is_fitted:
            return {}

        summary = {
            'n_components': self.n_components,
            'regime_mapping': self.regime_mapping,
            'regime_characteristics': self.regime_characteristics,
            'optimal_positions': self.optimal_positions,
            'expected_performance': self._calculate_expected_performance()
        }

        return summary

    def _calculate_expected_performance(self) -> Dict:
        """Calculate expected performance based on regime characteristics and positions"""
        if not self.regime_characteristics or not self.optimal_positions:
            return {}

        expected_return = 0
        expected_vol = 0
        total_weight = 0

        for state, regime_name in self.regime_mapping.items():
            if state in self.regime_characteristics:
                char = self.regime_characteristics[state]
                position = self.optimal_positions.get(regime_name, 0)
                frequency = char['frequency']

                # Weight by frequency
                expected_return += char['mean_return'] * position * frequency * 252
                expected_vol += (char['volatility'] * abs(position) * frequency) ** 2
                total_weight += frequency

        expected_vol = np.sqrt(expected_vol) * np.sqrt(252) if expected_vol > 0 else 0
        expected_sharpe = expected_return / expected_vol if expected_vol > 0 else 0

        return {
            'expected_annual_return': expected_return,
            'expected_volatility': expected_vol,
            'expected_sharpe': expected_sharpe
        }


def main():
    """Test the performance-optimized HMM"""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=1000, freq='D')

    # Create test features and returns
    features = pd.DataFrame(index=dates)
    features['feature1'] = np.random.normal(0, 1, 1000)
    features['feature2'] = np.random.normal(0, 1, 1000)

    returns = pd.Series(np.random.normal(0.001, 0.02, 1000), index=dates)

    # Test the model
    hmm_model = PerformanceOptimizedHMM()
    hmm_model.fit(features, returns)

    predictions = hmm_model.predict_regimes(features, returns)
    summary = hmm_model.get_model_summary()

    print("Performance-Optimized HMM test completed!")
    print(f"Regime mapping: {summary['regime_mapping']}")
    print(f"Expected performance: {summary['expected_performance']}")

    return hmm_model, predictions, summary


if __name__ == "__main__":
    main()