"""
Advanced future predictions module with confidence intervals and trading signals
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from scipy import stats
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedPredictor:
    """
    Advanced prediction system for regime forecasting and trading signals
    """

    def __init__(self, forecast_horizon: int = 30, confidence_levels: List[float] = [0.68, 0.95]):
        self.forecast_horizon = forecast_horizon
        self.confidence_levels = confidence_levels
        self.predictions = {}
        self.signals = {}

    def generate_future_predictions(self, hmm_model, garch_model,
                                  recent_data: pd.DataFrame,
                                  current_price: float) -> Dict:
        """
        Generate comprehensive future predictions
        """
        try:
            logger.info(f"🔮 Generating {self.forecast_horizon}-day future predictions...")

            # Get current market state
            current_state = self._assess_current_state(hmm_model, garch_model, recent_data)

            # Generate regime probability forecasts
            regime_forecasts = self._forecast_regime_probabilities(hmm_model, current_state)

            # Generate price forecasts with uncertainty
            price_forecasts = self._forecast_prices(garch_model, recent_data, current_price)

            # Generate volatility forecasts
            volatility_forecasts = self._forecast_volatility(garch_model, recent_data)

            # Calculate confidence intervals
            confidence_intervals = self._calculate_confidence_intervals(
                regime_forecasts, price_forecasts, volatility_forecasts
            )

            # Generate trading signals
            trading_signals = self._generate_future_signals(
                regime_forecasts, price_forecasts, volatility_forecasts,
                hmm_model.get_regime_positions()
            )

            # Create forecast dates
            forecast_dates = self._generate_forecast_dates()

            predictions = {
                'forecast_dates': forecast_dates,
                'current_state': current_state,
                'regime_forecasts': regime_forecasts,
                'price_forecasts': price_forecasts,
                'volatility_forecasts': volatility_forecasts,
                'confidence_intervals': confidence_intervals,
                'trading_signals': trading_signals,
                'forecast_horizon': self.forecast_horizon,
                'prediction_timestamp': datetime.now()
            }

            logger.info("✅ Future predictions generated successfully")

            return predictions

        except Exception as e:
            logger.error(f"❌ Error generating predictions: {str(e)}")
            raise

    def _assess_current_state(self, hmm_model, garch_model, recent_data: pd.DataFrame) -> Dict:
        """
        Assess current market state for prediction initialization
        """
        # Get recent features
        recent_features = garch_model.extract_regime_features(recent_data['Returns'].tail(100))

        # Get current regime probabilities
        current_predictions = hmm_model.predict_regimes(recent_features, recent_data['Returns'].tail(100))
        current_regime_probs = current_predictions['regime_probabilities'].iloc[-1]
        current_regime = current_predictions['regime_sequence'][-1]

        # Calculate current market metrics
        recent_returns = recent_data['Returns'].tail(20)
        current_vol = recent_returns.std() * np.sqrt(252)
        current_momentum = recent_data['Close'].pct_change(10).iloc[-1]
        current_trend = (recent_data['Close'].iloc[-1] / recent_data['Close'].iloc[-50] - 1)

        current_state = {
            'current_regime': current_regime,
            'regime_probabilities': current_regime_probs.to_dict(),
            'current_volatility': current_vol,
            'current_momentum': current_momentum,
            'current_trend': current_trend,
            'current_price': recent_data['Close'].iloc[-1],
            'confidence': current_regime_probs.max()
        }

        logger.info(f"📊 Current State: Regime={current_regime}, Confidence={current_regime_probs.max():.2f}")

        return current_state

    def _forecast_regime_probabilities(self, hmm_model, current_state: Dict) -> pd.DataFrame:
        """
        Forecast regime probabilities using transition matrix
        """
        # Get transition matrix
        if hasattr(hmm_model.model, 'transmat_'):
            transition_matrix = hmm_model.model.transmat_
        else:
            # Default transition matrix
            n = hmm_model.n_components
            transition_matrix = np.full((n, n), 0.1 / (n - 1))
            np.fill_diagonal(transition_matrix, 0.8)

        # Initial probabilities from current state
        regime_mapping = hmm_model.regime_mapping
        initial_probs = np.zeros(hmm_model.n_components)

        for state_id, regime_name in regime_mapping.items():
            if regime_name in current_state['regime_probabilities']:
                initial_probs[state_id] = current_state['regime_probabilities'][regime_name]

        # Normalize
        if initial_probs.sum() > 0:
            initial_probs /= initial_probs.sum()
        else:
            initial_probs = np.ones(hmm_model.n_components) / hmm_model.n_components

        # Forecast evolution
        forecast_probs = []
        current_probs = initial_probs.copy()

        for t in range(self.forecast_horizon):
            # Evolve probabilities
            current_probs = current_probs @ transition_matrix
            forecast_probs.append(current_probs.copy())

        # Convert to DataFrame with regime names
        forecast_df = pd.DataFrame(forecast_probs)
        forecast_df.columns = [regime_mapping.get(i, f'State_{i}') for i in range(hmm_model.n_components)]
        forecast_df.index = pd.date_range(start=datetime.now() + timedelta(days=1),
                                         periods=self.forecast_horizon, freq='D')

        return forecast_df

    def _forecast_prices(self, garch_model, recent_data: pd.DataFrame, current_price: float) -> Dict:
        """
        Forecast price paths with Monte Carlo simulation
        """
        # Get recent returns and volatility parameters
        recent_returns = recent_data['Returns'].tail(252)

        # Estimate drift and volatility
        annual_drift = recent_returns.mean() * 252
        current_vol = recent_returns.std() * np.sqrt(252)

        # Monte Carlo simulation
        n_simulations = 1000
        dt = 1/252  # Daily time step

        price_paths = []

        for _ in range(n_simulations):
            prices = [current_price]

            for t in range(self.forecast_horizon):
                # Generate random shock
                shock = np.random.normal(0, 1)

                # Calculate next price (geometric Brownian motion)
                next_price = prices[-1] * np.exp(
                    (annual_drift - 0.5 * current_vol**2) * dt +
                    current_vol * np.sqrt(dt) * shock
                )
                prices.append(next_price)

            price_paths.append(prices[1:])  # Exclude initial price

        price_paths = np.array(price_paths)

        # Calculate statistics
        price_stats = {
            'mean': np.mean(price_paths, axis=0),
            'median': np.median(price_paths, axis=0),
            'std': np.std(price_paths, axis=0),
            'percentiles': {
                '5': np.percentile(price_paths, 5, axis=0),
                '25': np.percentile(price_paths, 25, axis=0),
                '75': np.percentile(price_paths, 75, axis=0),
                '95': np.percentile(price_paths, 95, axis=0)
            },
            'all_paths': price_paths
        }

        return price_stats

    def _forecast_volatility(self, garch_model, recent_data: pd.DataFrame) -> np.ndarray:
        """
        Forecast volatility using GARCH model
        """
        try:
            # Use GARCH model's forecast method if available
            vol_forecast = garch_model.get_volatility_forecast(
                recent_data['Returns'], horizon=self.forecast_horizon
            )

            if len(vol_forecast) >= self.forecast_horizon:
                return vol_forecast.iloc[-self.forecast_horizon:].values
            else:
                # Fallback: use current volatility
                current_vol = recent_data['Returns'].tail(20).std() * np.sqrt(252)
                return np.full(self.forecast_horizon, current_vol)

        except:
            # Simple volatility forecast
            recent_vol = recent_data['Returns'].tail(20).std() * np.sqrt(252)
            return np.full(self.forecast_horizon, recent_vol)

    def _calculate_confidence_intervals(self, regime_forecasts: pd.DataFrame,
                                      price_forecasts: Dict,
                                      volatility_forecasts: np.ndarray) -> Dict:
        """
        Calculate confidence intervals for all forecasts
        """
        confidence_intervals = {}

        # Regime confidence intervals (based on probability concentration)
        regime_confidence = []
        for _, row in regime_forecasts.iterrows():
            max_prob = row.max()
            entropy = -np.sum(row * np.log(row + 1e-8))  # Avoid log(0)
            confidence = max_prob * (1 - entropy / np.log(len(row)))  # Normalized confidence
            regime_confidence.append(confidence)

        confidence_intervals['regime_confidence'] = np.array(regime_confidence)

        # Price confidence intervals (already calculated in price forecasts)
        confidence_intervals['price_intervals'] = price_forecasts['percentiles']

        # Volatility confidence intervals (assume ±20% uncertainty)
        vol_uncertainty = 0.2
        confidence_intervals['volatility_intervals'] = {
            'lower': volatility_forecasts * (1 - vol_uncertainty),
            'upper': volatility_forecasts * (1 + vol_uncertainty)
        }

        return confidence_intervals

    def _generate_future_signals(self, regime_forecasts: pd.DataFrame,
                                price_forecasts: Dict, volatility_forecasts: np.ndarray,
                                regime_positions: Dict[str, float]) -> pd.DataFrame:
        """
        Generate detailed trading signals for future periods
        """
        signals = []
        forecast_dates = self._generate_forecast_dates()

        for i in range(self.forecast_horizon):
            date = forecast_dates[i]

            # Get most likely regime
            regime_probs = regime_forecasts.iloc[i]
            most_likely_regime = regime_probs.idxmax()
            regime_confidence = regime_probs.max()

            # Get base position
            base_position = regime_positions.get(most_likely_regime, 0.0)

            # Adjust for confidence
            confidence_adj = 0.5 + 1.5 * regime_confidence  # Range: 0.5 to 2.0
            adjusted_position = base_position * confidence_adj

            # Volatility adjustment
            target_vol = 0.15
            vol_adj = target_vol / volatility_forecasts[i] if volatility_forecasts[i] > 0 else 1.0
            vol_adj = np.clip(vol_adj, 0.5, 2.0)

            final_position = adjusted_position * vol_adj
            final_position = np.clip(final_position, -3.0, 3.0)

            # Price targets
            current_price = price_forecasts['mean'][0] if i == 0 else price_forecasts['mean'][i-1]
            target_price = price_forecasts['mean'][i]
            expected_return = (target_price - current_price) / current_price

            # Generate signal
            action = 'BUY' if final_position > 0.1 else 'SELL' if final_position < -0.1 else 'HOLD'

            signal = {
                'date': date,
                'regime': most_likely_regime,
                'regime_confidence': regime_confidence,
                'position': final_position,
                'action': action,
                'expected_return': expected_return,
                'target_price': target_price,
                'price_lower_95': price_forecasts['percentiles']['5'][i],
                'price_upper_95': price_forecasts['percentiles']['95'][i],
                'volatility_forecast': volatility_forecasts[i],
                'signal_strength': self._calculate_signal_strength(regime_confidence, abs(final_position)),
                'risk_level': self._assess_risk_level(volatility_forecasts[i], regime_confidence),
                'stop_loss': self._calculate_stop_loss(target_price, volatility_forecasts[i], final_position),
                'take_profit': self._calculate_take_profit(target_price, expected_return, final_position)
            }

            signals.append(signal)

        return pd.DataFrame(signals)

    def _calculate_signal_strength(self, confidence: float, position_magnitude: float) -> str:
        """Calculate signal strength"""
        strength = confidence * position_magnitude

        if strength >= 2.0:
            return 'Very Strong'
        elif strength >= 1.5:
            return 'Strong'
        elif strength >= 1.0:
            return 'Moderate'
        elif strength >= 0.5:
            return 'Weak'
        else:
            return 'Very Weak'

    def _assess_risk_level(self, volatility: float, confidence: float) -> str:
        """Assess risk level of signal"""
        risk_score = volatility / confidence

        if risk_score <= 0.1:
            return 'Low'
        elif risk_score <= 0.2:
            return 'Medium'
        elif risk_score <= 0.3:
            return 'High'
        else:
            return 'Very High'

    def _calculate_stop_loss(self, target_price: float, volatility: float, position: float) -> float:
        """Calculate stop loss level"""
        vol_multiplier = 2.0  # 2 standard deviations
        stop_distance = target_price * volatility / np.sqrt(252) * vol_multiplier

        if position > 0:  # Long position
            return target_price - stop_distance
        else:  # Short position
            return target_price + stop_distance

    def _calculate_take_profit(self, target_price: float, expected_return: float, position: float) -> float:
        """Calculate take profit level"""
        if abs(expected_return) < 0.01:  # Less than 1% expected move
            profit_target = 0.03  # 3% profit target
        else:
            profit_target = abs(expected_return) * 2  # 2x expected return

        if position > 0:  # Long position
            return target_price * (1 + profit_target)
        else:  # Short position
            return target_price * (1 - profit_target)

    def _generate_forecast_dates(self) -> List[datetime]:
        """Generate forecast dates (business days only)"""
        dates = []
        current_date = datetime.now()

        while len(dates) < self.forecast_horizon:
            current_date += timedelta(days=1)
            # Skip weekends
            if current_date.weekday() < 5:
                dates.append(current_date)

        return dates

    def generate_prediction_summary(self, predictions: Dict) -> str:
        """
        Generate human-readable prediction summary
        """
        current_state = predictions['current_state']
        signals = predictions['trading_signals']

        # Next 5 days analysis
        next_5_days = signals.head(5)

        summary = f"""
🔮 FUTURE MARKET PREDICTIONS - {self.forecast_horizon} Day Outlook
{'='*70}

📊 CURRENT MARKET STATE:
   Current Regime:    {current_state['current_regime']}
   Confidence:        {current_state['confidence']:.1%}
   Current Price:     ${current_state['current_price']:.2f}
   Volatility:        {current_state['current_volatility']:.1%}
   Momentum:          {current_state['current_momentum']:.1%}

🎯 NEXT 5 TRADING DAYS OUTLOOK:
"""

        for i, (_, signal) in enumerate(next_5_days.iterrows(), 1):
            summary += f"""
Day {i} ({signal['date'].strftime('%Y-%m-%d')}):
   Action:           {signal['action']} ({signal['signal_strength']})
   Regime:           {signal['regime']} (Confidence: {signal['regime_confidence']:.1%})
   Target Price:     ${signal['target_price']:.2f}
   Expected Return:  {signal['expected_return']:.1%}
   Risk Level:       {signal['risk_level']}
   Stop Loss:        ${signal['stop_loss']:.2f}
   Take Profit:      ${signal['take_profit']:.2f}
"""

        # Overall outlook
        strong_signals = signals[signals['signal_strength'].isin(['Strong', 'Very Strong'])]
        buy_signals = len(signals[signals['action'] == 'BUY'])
        sell_signals = len(signals[signals['action'] == 'SELL'])

        summary += f"""
📈 {self.forecast_horizon}-DAY STRATEGIC OUTLOOK:
   Strong Signals:    {len(strong_signals)}/{len(signals)}
   Buy Signals:       {buy_signals}
   Sell Signals:      {sell_signals}
   Hold Periods:      {len(signals) - buy_signals - sell_signals}

   Overall Bias:      {'BULLISH' if buy_signals > sell_signals else 'BEARISH' if sell_signals > buy_signals else 'NEUTRAL'}

⚠️  RISK WARNINGS:
"""

        # Add risk warnings
        high_risk_days = len(signals[signals['risk_level'].isin(['High', 'Very High'])])
        if high_risk_days > self.forecast_horizon * 0.3:
            summary += f"   • {high_risk_days} high-risk trading days ahead\n"

        low_confidence_days = len(signals[signals['regime_confidence'] < 0.6])
        if low_confidence_days > self.forecast_horizon * 0.3:
            summary += f"   • {low_confidence_days} low-confidence prediction days\n"

        summary += f"\n📅 Prediction Generated: {predictions['prediction_timestamp'].strftime('%Y-%m-%d %H:%M:%S')}"
        summary += f"\n{'='*70}"

        return summary


def main():
    """Test the advanced predictor"""
    # This would be called with real models and data
    predictor = AdvancedPredictor(forecast_horizon=20)
    print("Advanced Predictor initialized successfully")
    print("Use predictor.generate_future_predictions() with trained models")

    return predictor


if __name__ == "__main__":
    main()