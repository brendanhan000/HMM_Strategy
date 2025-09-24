"""
Regime prediction and signal generation module
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from scipy import stats
from datetime import datetime, timedelta

from config.settings import PREDICTION_CONFIG, REGIME_LABELS, REGIME_POSITIONS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RegimePredictor:
    """
    Advanced regime prediction and trading signal generation
    """

    def __init__(self, forecast_horizon: int = None, confidence_level: float = None,
                 min_prob_threshold: float = None):
        """
        Initialize regime predictor

        Parameters:
        -----------
        forecast_horizon : int, number of periods to forecast
        confidence_level : float, confidence level for intervals
        min_prob_threshold : float, minimum probability for regime signals
        """
        self.forecast_horizon = forecast_horizon or PREDICTION_CONFIG['forecast_horizon']
        self.confidence_level = confidence_level or 0.95
        self.min_prob_threshold = min_prob_threshold or PREDICTION_CONFIG['min_prob_threshold']

    def generate_regime_forecast(self, hmm_model, current_observations: np.ndarray,
                                volatility_forecast: pd.Series) -> Dict:
        """
        Generate comprehensive regime forecast

        Parameters:
        -----------
        hmm_model : fitted HMM model
        current_observations : recent observations for context
        volatility_forecast : volatility forecasts for position sizing

        Returns:
        --------
        forecast : Dict with comprehensive forecast results
        """
        try:
            logger.info(f"Generating {self.forecast_horizon}-period regime forecast...")

            # Get model transition matrix and current state probabilities
            transition_matrix = hmm_model.model.transmat_

            # Estimate current state probabilities from recent observations
            if len(current_observations) > 0:
                current_state_probs = hmm_model.model.predict_proba(current_observations[-1:])[-1]
            else:
                current_state_probs = hmm_model.model.startprob_

            # Generate forecast probabilities
            forecast_probs = self._forecast_probabilities(
                transition_matrix, current_state_probs, self.forecast_horizon
            )

            # Calculate most likely regime sequence
            most_likely_regimes = np.argmax(forecast_probs, axis=1)

            # Calculate confidence metrics
            confidence_metrics = self._calculate_confidence_metrics(forecast_probs)

            # Generate trading signals
            trading_signals = self._generate_trading_signals(
                forecast_probs, most_likely_regimes, volatility_forecast, confidence_metrics
            )

            # Calculate regime transition probabilities
            transition_analysis = self._analyze_regime_transitions(
                forecast_probs, transition_matrix
            )

            forecast = {
                'forecast_probabilities': forecast_probs,
                'most_likely_regimes': most_likely_regimes,
                'confidence_metrics': confidence_metrics,
                'trading_signals': trading_signals,
                'transition_analysis': transition_analysis,
                'forecast_horizon': self.forecast_horizon,
                'forecast_dates': self._generate_forecast_dates()
            }

            logger.info("Regime forecast generated successfully")
            return forecast

        except Exception as e:
            logger.error(f"Error generating regime forecast: {str(e)}")
            raise

    def _forecast_probabilities(self, transition_matrix: np.ndarray,
                               initial_probs: np.ndarray,
                               horizon: int) -> np.ndarray:
        """
        Forecast regime probabilities using transition matrix

        Parameters:
        -----------
        transition_matrix : transition probability matrix
        initial_probs : initial state probabilities
        horizon : forecast horizon

        Returns:
        --------
        forecast_probs : array of shape (horizon, n_states)
        """
        n_states = len(initial_probs)
        forecast_probs = np.zeros((horizon, n_states))

        current_probs = initial_probs.copy()

        for t in range(horizon):
            # Evolve probabilities one step forward
            current_probs = current_probs @ transition_matrix
            forecast_probs[t] = current_probs.copy()

        return forecast_probs

    def _calculate_confidence_metrics(self, forecast_probs: np.ndarray) -> Dict:
        """
        Calculate confidence metrics for forecasts

        Parameters:
        -----------
        forecast_probs : forecast probability matrix

        Returns:
        --------
        confidence_metrics : Dict with confidence measures
        """
        # Maximum probability (confidence in most likely state)
        max_probs = np.max(forecast_probs, axis=1)

        # Entropy (uncertainty measure)
        entropy = -np.sum(forecast_probs * np.log(forecast_probs + 1e-8), axis=1)

        # Gini coefficient (concentration measure)
        gini = self._calculate_gini_coefficient(forecast_probs)

        # Regime stability (consistency of most likely regime)
        most_likely = np.argmax(forecast_probs, axis=1)
        stability = self._calculate_regime_stability(most_likely)

        return {
            'max_probabilities': max_probs,
            'entropy': entropy,
            'gini_coefficient': gini,
            'regime_stability': stability,
            'avg_confidence': np.mean(max_probs),
            'min_confidence': np.min(max_probs),
            'confidence_trend': self._calculate_confidence_trend(max_probs)
        }

    def _calculate_gini_coefficient(self, probs: np.ndarray) -> np.ndarray:
        """Calculate Gini coefficient for each time period"""
        gini_coeffs = []
        for t in range(len(probs)):
            p = np.sort(probs[t])
            n = len(p)
            index = np.arange(1, n + 1)
            gini = (2 * np.sum(index * p)) / (n * np.sum(p)) - (n + 1) / n
            gini_coeffs.append(gini)
        return np.array(gini_coeffs)

    def _calculate_regime_stability(self, regime_sequence: np.ndarray) -> Dict:
        """Calculate regime stability metrics"""
        # Number of regime changes
        changes = np.sum(np.diff(regime_sequence) != 0)

        # Average regime duration
        durations = []
        current_regime = regime_sequence[0]
        current_duration = 1

        for i in range(1, len(regime_sequence)):
            if regime_sequence[i] == current_regime:
                current_duration += 1
            else:
                durations.append(current_duration)
                current_regime = regime_sequence[i]
                current_duration = 1
        durations.append(current_duration)

        return {
            'num_changes': changes,
            'change_frequency': changes / len(regime_sequence),
            'avg_duration': np.mean(durations),
            'max_duration': max(durations),
            'stability_score': 1 - (changes / len(regime_sequence))
        }

    def _calculate_confidence_trend(self, max_probs: np.ndarray) -> Dict:
        """Calculate confidence trend over forecast horizon"""
        if len(max_probs) < 2:
            return {'slope': 0, 'trend': 'stable'}

        # Linear regression to find trend
        x = np.arange(len(max_probs))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, max_probs)

        # Categorize trend
        if slope > 0.01:
            trend = 'increasing'
        elif slope < -0.01:
            trend = 'decreasing'
        else:
            trend = 'stable'

        return {
            'slope': slope,
            'r_squared': r_value**2,
            'p_value': p_value,
            'trend': trend
        }

    def _generate_trading_signals(self, forecast_probs: np.ndarray,
                                 most_likely_regimes: np.ndarray,
                                 volatility_forecast: pd.Series,
                                 confidence_metrics: Dict) -> pd.DataFrame:
        """
        Generate actionable trading signals

        Parameters:
        -----------
        forecast_probs : forecast probability matrix
        most_likely_regimes : most likely regime sequence
        volatility_forecast : volatility forecasts
        confidence_metrics : confidence measures

        Returns:
        --------
        signals_df : DataFrame with trading signals
        """
        signals = []
        forecast_dates = self._generate_forecast_dates()

        for t in range(len(forecast_probs)):
            regime = most_likely_regimes[t]
            regime_prob = forecast_probs[t, regime]
            confidence = confidence_metrics['max_probabilities'][t]

            # Base position from regime
            base_position = REGIME_POSITIONS.get(regime, 0.0)

            # Adjust position based on confidence
            confidence_adj = 1.0 if confidence >= self.min_prob_threshold else confidence / self.min_prob_threshold

            # Adjust position based on volatility (if available)
            vol_adj = 1.0
            if volatility_forecast is not None and t < len(volatility_forecast):
                vol_adj = self._calculate_volatility_adjustment(volatility_forecast.iloc[t])

            # Final position
            final_position = base_position * confidence_adj * vol_adj
            final_position = np.clip(final_position, -1.0, 1.0)  # Position limits

            # Generate signal strength and recommendations
            signal_strength = self._calculate_signal_strength(regime_prob, confidence)
            recommendations = self._generate_recommendations(
                regime, regime_prob, confidence, final_position
            )

            signal = {
                'date': forecast_dates[t],
                'regime': regime,
                'regime_label': REGIME_LABELS.get(regime, f'Regime_{regime}'),
                'regime_probability': regime_prob,
                'confidence': confidence,
                'base_position': base_position,
                'confidence_adjustment': confidence_adj,
                'volatility_adjustment': vol_adj,
                'final_position': final_position,
                'signal_strength': signal_strength,
                'action': self._determine_action(final_position),
                'risk_level': self._assess_risk_level(confidence, regime),
                'recommendations': recommendations
            }

            signals.append(signal)

        return pd.DataFrame(signals)

    def _calculate_volatility_adjustment(self, volatility: float,
                                       target_vol: float = 0.15) -> float:
        """Calculate position adjustment based on volatility forecast"""
        if volatility <= 0:
            return 1.0

        # Inverse volatility scaling with limits
        adjustment = target_vol / volatility
        return np.clip(adjustment, 0.25, 2.0)

    def _calculate_signal_strength(self, regime_prob: float, confidence: float) -> str:
        """Calculate signal strength category"""
        strength_score = (regime_prob + confidence) / 2

        if strength_score >= 0.8:
            return 'Strong'
        elif strength_score >= 0.6:
            return 'Moderate'
        elif strength_score >= 0.4:
            return 'Weak'
        else:
            return 'Very Weak'

    def _determine_action(self, position: float) -> str:
        """Determine trading action based on position"""
        if position > 0.1:
            return 'BUY'
        elif position < -0.1:
            return 'SELL'
        else:
            return 'HOLD'

    def _assess_risk_level(self, confidence: float, regime: int) -> str:
        """Assess risk level of the signal"""
        if confidence >= 0.8:
            risk = 'Low'
        elif confidence >= 0.6:
            risk = 'Medium'
        elif confidence >= 0.4:
            risk = 'High'
        else:
            risk = 'Very High'

        # Adjust for regime volatility
        if regime == 2:  # Bear regime (assuming higher volatility)
            risk_levels = {'Low': 'Medium', 'Medium': 'High', 'High': 'Very High', 'Very High': 'Extreme'}
            risk = risk_levels.get(risk, risk)

        return risk

    def _generate_recommendations(self, regime: int, regime_prob: float,
                                confidence: float, position: float) -> Dict:
        """Generate specific trading recommendations"""
        recommendations = {
            'entry_price': 'Market',
            'position_size': f"{abs(position):.1%}",
            'stop_loss': None,
            'take_profit': None,
            'hold_period': None
        }

        # Regime-specific recommendations
        if regime == 0:  # Bull regime
            recommendations.update({
                'stop_loss': '5% below entry',
                'take_profit': '15% above entry',
                'hold_period': '5-20 days'
            })
        elif regime == 1:  # Sideways regime
            recommendations.update({
                'stop_loss': '3% from entry',
                'take_profit': '3% from entry',
                'hold_period': '1-5 days'
            })
        elif regime == 2:  # Bear regime
            recommendations.update({
                'stop_loss': '5% above entry (for short)',
                'take_profit': '15% below entry',
                'hold_period': '5-20 days'
            })

        # Adjust based on confidence
        if confidence < 0.6:
            recommendations['position_size'] = f"{abs(position) * 0.5:.1%} (reduced due to low confidence)"

        return recommendations

    def _analyze_regime_transitions(self, forecast_probs: np.ndarray,
                                  transition_matrix: np.ndarray) -> Dict:
        """Analyze regime transition probabilities"""
        most_likely = np.argmax(forecast_probs, axis=1)

        # Expected number of transitions
        transition_probs = []
        for t in range(len(forecast_probs) - 1):
            current_probs = forecast_probs[t]
            next_period_probs = forecast_probs[t + 1]

            # Probability of regime change
            same_regime_prob = np.sum(current_probs * np.diag(transition_matrix))
            change_prob = 1 - same_regime_prob
            transition_probs.append(change_prob)

        # Regime duration expectations
        regime_durations = {}
        for regime in range(transition_matrix.shape[0]):
            # Expected duration = 1 / (1 - self_transition_prob)
            self_prob = transition_matrix[regime, regime]
            expected_duration = 1 / (1 - self_prob) if self_prob < 1 else float('inf')
            regime_durations[REGIME_LABELS.get(regime, f'Regime_{regime}')] = expected_duration

        return {
            'transition_probabilities': transition_probs,
            'avg_transition_prob': np.mean(transition_probs) if transition_probs else 0,
            'regime_durations': regime_durations,
            'most_stable_regime': max(regime_durations, key=regime_durations.get),
            'least_stable_regime': min(regime_durations, key=regime_durations.get)
        }

    def _generate_forecast_dates(self) -> List[datetime]:
        """Generate forecast dates"""
        start_date = datetime.now()
        return [start_date + timedelta(days=i) for i in range(1, self.forecast_horizon + 1)]

    def generate_risk_warnings(self, forecast_results: Dict) -> List[str]:
        """
        Generate risk warnings based on forecast results

        Parameters:
        -----------
        forecast_results : forecast results dictionary

        Returns:
        --------
        warnings : List of risk warning messages
        """
        warnings = []

        confidence_metrics = forecast_results.get('confidence_metrics', {})
        trading_signals = forecast_results.get('trading_signals')

        # Low confidence warnings
        avg_confidence = confidence_metrics.get('avg_confidence', 1.0)
        if avg_confidence < 0.6:
            warnings.append(f"LOW CONFIDENCE: Average forecast confidence is {avg_confidence:.1%}")

        # High volatility warnings
        if trading_signals is not None:
            high_risk_signals = trading_signals[trading_signals['risk_level'].isin(['High', 'Very High', 'Extreme'])]
            if len(high_risk_signals) > len(trading_signals) * 0.5:
                warnings.append("HIGH RISK: More than 50% of signals are high risk")

        # Regime instability warnings
        stability = confidence_metrics.get('regime_stability', {})
        if stability.get('change_frequency', 0) > 0.3:
            warnings.append("REGIME INSTABILITY: High frequency of regime changes predicted")

        # Trend warnings
        confidence_trend = confidence_metrics.get('confidence_trend', {})
        if confidence_trend.get('trend') == 'decreasing':
            warnings.append("DECREASING CONFIDENCE: Forecast confidence is declining over time")

        return warnings

    def calculate_forecast_accuracy(self, forecast_results: Dict,
                                  actual_regimes: np.ndarray) -> Dict:
        """
        Calculate forecast accuracy metrics (for backtesting)

        Parameters:
        -----------
        forecast_results : historical forecast results
        actual_regimes : actual regime sequence

        Returns:
        --------
        accuracy_metrics : Dict with accuracy measures
        """
        predicted_regimes = forecast_results.get('most_likely_regimes', np.array([]))
        forecast_probs = forecast_results.get('forecast_probabilities', np.array([]))

        if len(predicted_regimes) == 0 or len(actual_regimes) == 0:
            return {}

        # Align data
        min_length = min(len(predicted_regimes), len(actual_regimes))
        predicted = predicted_regimes[:min_length]
        actual = actual_regimes[:min_length]

        # Classification accuracy
        accuracy = np.mean(predicted == actual)

        # Regime-specific accuracy
        regime_accuracy = {}
        for regime in np.unique(actual):
            regime_mask = actual == regime
            if np.sum(regime_mask) > 0:
                regime_accuracy[regime] = np.mean(predicted[regime_mask] == actual[regime_mask])

        # Probabilistic accuracy (Brier score)
        if len(forecast_probs) > 0:
            probs_aligned = forecast_probs[:min_length]
            brier_scores = []
            for t in range(min_length):
                true_regime = actual[t]
                predicted_probs = probs_aligned[t]
                # Brier score for this time point
                brier = np.sum((predicted_probs - (np.arange(len(predicted_probs)) == true_regime))**2)
                brier_scores.append(brier)
            avg_brier_score = np.mean(brier_scores)
        else:
            avg_brier_score = None

        return {
            'overall_accuracy': accuracy,
            'regime_specific_accuracy': regime_accuracy,
            'brier_score': avg_brier_score,
            'num_predictions': min_length
        }


def main():
    """
    Example usage of RegimePredictor
    """
    predictor = RegimePredictor(forecast_horizon=20)
    print("RegimePredictor initialized successfully")
    print("Use predictor.generate_regime_forecast(hmm_model, observations, volatility) for predictions")

    return predictor


if __name__ == "__main__":
    main()