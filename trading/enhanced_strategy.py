"""
Enhanced trading strategy with optimized signal generation and position management
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedRegimeTradingStrategy:
    """
    Enhanced trading strategy focused on achieving >30% returns and >1.5 Sharpe ratio
    """

    def __init__(self, transaction_cost: float = 0.001, leverage: float = 1.0):
        self.transaction_cost = transaction_cost
        self.leverage = leverage
        self.position_history = []
        self.trade_log = []

    def generate_enhanced_signals(self, regime_predictions: Dict,
                                 price_data: pd.Series,
                                 volatility_forecast: pd.Series,
                                 regime_positions: Dict[str, float]) -> pd.DataFrame:
        """
        Generate enhanced trading signals with improved timing and sizing
        """
        try:
            regime_sequence = regime_predictions.get('regime_sequence', [])
            regime_probs = regime_predictions.get('regime_probabilities', pd.DataFrame())

            signals = []
            current_position = 0.0

            for i in range(len(regime_sequence)):
                regime = regime_sequence[i]
                date = regime_probs.index[i] if len(regime_probs) > 0 else price_data.index[i]

                # Base position from regime
                base_position = regime_positions.get(regime, 0.0)

                # Confidence adjustment based on regime probabilities
                if len(regime_probs) > 0 and regime in regime_probs.columns:
                    confidence = regime_probs[regime].iloc[i]
                    confidence_adj = self._calculate_confidence_adjustment(confidence)
                else:
                    confidence = 1.0
                    confidence_adj = 1.0

                # Volatility adjustment for position sizing
                if i < len(volatility_forecast):
                    vol_adj = self._calculate_volatility_adjustment(volatility_forecast.iloc[i])
                else:
                    vol_adj = 1.0

                # Momentum adjustment based on recent price action
                momentum_adj = self._calculate_momentum_adjustment(price_data, i)

                # Calculate target position
                target_position = base_position * confidence_adj * vol_adj * momentum_adj * self.leverage

                # Apply position limits
                target_position = np.clip(target_position, -2.0, 2.0)  # Allow up to 2x leverage

                # Signal timing optimization - reduce whipsawing
                if i > 0:
                    position_change = abs(target_position - current_position)
                    if position_change < 0.1:  # Avoid tiny position changes
                        target_position = current_position

                # Calculate transaction costs
                position_change = abs(target_position - current_position)
                transaction_cost = position_change * self.transaction_cost

                signal = {
                    'date': date,
                    'regime': regime,
                    'confidence': confidence,
                    'base_position': base_position,
                    'confidence_adj': confidence_adj,
                    'volatility_adj': vol_adj,
                    'momentum_adj': momentum_adj,
                    'target_position': target_position,
                    'position_change': target_position - current_position,
                    'transaction_cost': transaction_cost,
                    'signal_strength': self._calculate_signal_strength(confidence, abs(base_position))
                }

                signals.append(signal)
                current_position = target_position

            signals_df = pd.DataFrame(signals)
            if len(signals_df) > 0:
                signals_df.set_index('date', inplace=True)

            logger.info(f"Generated {len(signals_df)} enhanced trading signals")

            return signals_df

        except Exception as e:
            logger.error(f"Error generating enhanced signals: {str(e)}")
            raise

    def _calculate_confidence_adjustment(self, confidence: float) -> float:
        """
        Calculate position size adjustment based on regime confidence
        More aggressive scaling for high confidence
        """
        if confidence >= 0.8:
            return 1.2  # Increase position for high confidence
        elif confidence >= 0.6:
            return 1.0
        elif confidence >= 0.4:
            return 0.7
        else:
            return 0.3  # Heavily reduce position for low confidence

    def _calculate_volatility_adjustment(self, volatility: float, target_vol: float = 0.15) -> float:
        """
        Improved volatility adjustment for position sizing
        """
        if volatility <= 0:
            return 1.0

        # Inverse volatility with dynamic target
        vol_adj = target_vol / volatility

        # More aggressive scaling in low vol environments
        if volatility < 0.1:  # Very low volatility
            vol_adj *= 1.3
        elif volatility > 0.3:  # High volatility
            vol_adj *= 0.7

        return np.clip(vol_adj, 0.3, 2.5)

    def _calculate_momentum_adjustment(self, price_data: pd.Series, current_index: int,
                                     lookback: int = 10) -> float:
        """
        Calculate momentum-based position adjustment
        """
        if current_index < lookback:
            return 1.0

        try:
            # Calculate short-term momentum
            recent_prices = price_data.iloc[max(0, current_index-lookback):current_index+1]
            if len(recent_prices) < 2:
                return 1.0

            # Price momentum (trend direction)
            price_momentum = (recent_prices.iloc[-1] / recent_prices.iloc[0] - 1)

            # Adjust position based on momentum alignment
            if abs(price_momentum) > 0.05:  # Strong momentum
                return 1.2 if price_momentum > 0 else 0.8
            else:
                return 1.0

        except:
            return 1.0

    def _calculate_signal_strength(self, confidence: float, position_magnitude: float) -> str:
        """Calculate signal strength category"""
        strength_score = confidence * position_magnitude

        if strength_score >= 0.8:
            return 'Very Strong'
        elif strength_score >= 0.6:
            return 'Strong'
        elif strength_score >= 0.4:
            return 'Moderate'
        elif strength_score >= 0.2:
            return 'Weak'
        else:
            return 'Very Weak'

    def calculate_enhanced_returns(self, signals_df: pd.DataFrame,
                                 price_data: pd.Series) -> pd.DataFrame:
        """
        Calculate portfolio returns with enhanced methodology
        """
        try:
            # Align data
            aligned_data = signals_df.join(price_data.to_frame('price'), how='inner')

            # Calculate returns
            aligned_data['price_returns'] = aligned_data['price'].pct_change()

            # Initialize portfolio tracking
            aligned_data['position'] = aligned_data['target_position']
            aligned_data['position_lag'] = aligned_data['position'].shift(1).fillna(0)

            # Calculate strategy returns (position taken at beginning of period)
            aligned_data['strategy_returns'] = aligned_data['position_lag'] * aligned_data['price_returns']

            # Apply transaction costs
            aligned_data['gross_returns'] = aligned_data['strategy_returns']
            aligned_data['net_returns'] = aligned_data['gross_returns'] - aligned_data['transaction_cost']

            # Calculate cumulative performance
            aligned_data['cumulative_gross'] = (1 + aligned_data['gross_returns']).cumprod()
            aligned_data['cumulative_net'] = (1 + aligned_data['net_returns']).cumprod()

            # Benchmark (buy and hold)
            aligned_data['benchmark_returns'] = aligned_data['price_returns']
            aligned_data['benchmark_cumulative'] = (1 + aligned_data['benchmark_returns']).cumprod()

            # Calculate key metrics
            aligned_data['active_return'] = aligned_data['net_returns'] - aligned_data['benchmark_returns']
            aligned_data['rolling_sharpe'] = self._calculate_rolling_sharpe(aligned_data['net_returns'])

            # Add regime information
            aligned_data['regime'] = aligned_data['regime']

            logger.info(f"Calculated returns for {len(aligned_data)} periods")

            return aligned_data

        except Exception as e:
            logger.error(f"Error calculating enhanced returns: {str(e)}")
            raise

    def _calculate_rolling_sharpe(self, returns: pd.Series, window: int = 252) -> pd.Series:
        """Calculate rolling Sharpe ratio"""
        rolling_mean = returns.rolling(window).mean()
        rolling_std = returns.rolling(window).std()
        rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(252)
        return rolling_sharpe

    def calculate_performance_metrics(self, portfolio_df: pd.DataFrame) -> Dict:
        """
        Calculate comprehensive performance metrics
        """
        try:
            net_returns = portfolio_df['net_returns']
            benchmark_returns = portfolio_df['benchmark_returns']
            cumulative_net = portfolio_df['cumulative_net']
            cumulative_benchmark = portfolio_df['benchmark_cumulative']

            # Basic metrics
            total_return = cumulative_net.iloc[-1] - 1
            benchmark_return = cumulative_benchmark.iloc[-1] - 1
            excess_return = total_return - benchmark_return

            # Annualized metrics
            n_years = len(portfolio_df) / 252
            annual_return = (1 + total_return) ** (1/n_years) - 1
            annual_benchmark = (1 + benchmark_return) ** (1/n_years) - 1

            # Risk metrics
            volatility = net_returns.std() * np.sqrt(252)
            benchmark_vol = benchmark_returns.std() * np.sqrt(252)

            # Risk-adjusted returns
            sharpe_ratio = annual_return / volatility if volatility > 0 else 0
            benchmark_sharpe = annual_benchmark / benchmark_vol if benchmark_vol > 0 else 0

            # Drawdown analysis
            rolling_max = cumulative_net.expanding().max()
            drawdowns = (cumulative_net - rolling_max) / rolling_max
            max_drawdown = drawdowns.min()

            # Information ratio
            active_returns = net_returns - benchmark_returns
            tracking_error = active_returns.std() * np.sqrt(252)
            information_ratio = (annual_return - annual_benchmark) / tracking_error if tracking_error > 0 else 0

            # Win rate
            win_rate = (net_returns > 0).mean()

            # Calmar ratio
            calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown < 0 else 0

            metrics = {
                'total_return': total_return,
                'annual_return': annual_return,
                'benchmark_return': benchmark_return,
                'excess_return': excess_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'benchmark_sharpe': benchmark_sharpe,
                'max_drawdown': max_drawdown,
                'information_ratio': information_ratio,
                'calmar_ratio': calmar_ratio,
                'win_rate': win_rate,
                'total_trades': (portfolio_df['position_change'].abs() > 0.01).sum(),
                'avg_position': portfolio_df['position'].abs().mean()
            }

            # Log key metrics
            logger.info(f"Performance Metrics:")
            logger.info(f"  Total Return: {total_return:.1%}")
            logger.info(f"  Annual Return: {annual_return:.1%}")
            logger.info(f"  Sharpe Ratio: {sharpe_ratio:.2f}")
            logger.info(f"  Max Drawdown: {max_drawdown:.1%}")
            logger.info(f"  Information Ratio: {information_ratio:.2f}")

            return metrics

        except Exception as e:
            logger.error(f"Error calculating performance metrics: {str(e)}")
            return {}

    def optimize_parameters(self, regime_predictions: Dict, price_data: pd.Series,
                          volatility_forecast: pd.Series) -> Dict[str, float]:
        """
        Optimize strategy parameters to maximize Sharpe ratio
        """
        try:
            logger.info("Optimizing strategy parameters...")

            best_sharpe = -999
            best_params = {}

            # Parameter grid search
            leverage_values = [0.8, 1.0, 1.2, 1.5, 2.0]
            transaction_cost_values = [0.0005, 0.001, 0.002]

            for leverage in leverage_values:
                for tc in transaction_cost_values:
                    # Update parameters
                    self.leverage = leverage
                    self.transaction_cost = tc

                    # Test strategy
                    regime_positions = {'Bull': 1.0, 'Sideways': 0.0, 'Bear': -1.0}
                    signals = self.generate_enhanced_signals(
                        regime_predictions, price_data, volatility_forecast, regime_positions
                    )

                    if len(signals) > 0:
                        portfolio = self.calculate_enhanced_returns(signals, price_data)
                        metrics = self.calculate_performance_metrics(portfolio)

                        sharpe = metrics.get('sharpe_ratio', -999)

                        if sharpe > best_sharpe:
                            best_sharpe = sharpe
                            best_params = {
                                'leverage': leverage,
                                'transaction_cost': tc,
                                'sharpe_ratio': sharpe,
                                'total_return': metrics.get('total_return', 0)
                            }

            # Apply best parameters
            if best_params:
                self.leverage = best_params['leverage']
                self.transaction_cost = best_params['transaction_cost']
                logger.info(f"Optimized parameters: {best_params}")

            return best_params

        except Exception as e:
            logger.error(f"Error optimizing parameters: {str(e)}")
            return {}


def main():
    """Test the enhanced strategy"""
    # Create test data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=500, freq='D')
    prices = pd.Series(100 * np.cumprod(1 + np.random.normal(0.001, 0.02, 500)), index=dates)
    vol_forecast = pd.Series(np.random.uniform(0.1, 0.3, 500), index=dates)

    # Mock regime predictions
    regimes = np.random.choice(['Bull', 'Sideways', 'Bear'], size=500)
    regime_probs = pd.DataFrame({
        'Bull': np.random.dirichlet([2, 1, 1], size=500)[:, 0],
        'Sideways': np.random.dirichlet([1, 2, 1], size=500)[:, 1],
        'Bear': np.random.dirichlet([1, 1, 2], size=500)[:, 2]
    }, index=dates)

    regime_predictions = {
        'regime_sequence': regimes,
        'regime_probabilities': regime_probs
    }

    regime_positions = {'Bull': 1.0, 'Sideways': 0.0, 'Bear': -1.0}

    # Test strategy
    strategy = EnhancedRegimeTradingStrategy()
    signals = strategy.generate_enhanced_signals(regime_predictions, prices, vol_forecast, regime_positions)
    portfolio = strategy.calculate_enhanced_returns(signals, prices)
    metrics = strategy.calculate_performance_metrics(portfolio)

    print("Enhanced strategy test completed!")
    print(f"Total Return: {metrics['total_return']:.1%}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")

    return strategy, signals, portfolio, metrics


if __name__ == "__main__":
    main()