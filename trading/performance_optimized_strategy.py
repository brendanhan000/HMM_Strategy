"""
Performance-optimized trading strategy targeting >30% return and >1.5 Sharpe ratio
Based on actual regime profitability analysis
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerformanceOptimizedStrategy:
    """
    Trading strategy optimized based on actual regime performance data
    """

    def __init__(self, transaction_cost: float = 0.0005):
        self.transaction_cost = transaction_cost
        self.target_return = 0.30  # 30% target
        self.target_sharpe = 1.5   # 1.5 Sharpe target

    def create_optimized_signals(self, regime_predictions: Dict, price_data: pd.Series,
                                regime_positions: Dict[str, float]) -> pd.DataFrame:
        """
        Create trading signals optimized for maximum profitability
        """
        try:
            regime_sequence = regime_predictions.get('regime_sequence', [])
            regime_probs = regime_predictions.get('regime_probabilities', pd.DataFrame())

            # Align data
            min_length = min(len(regime_sequence), len(price_data))

            # Create signals DataFrame
            signals_df = pd.DataFrame(index=price_data.index[:min_length])
            signals_df['price'] = price_data.iloc[:min_length]
            signals_df['regime'] = regime_sequence[:min_length]

            # Apply optimized position mapping
            optimized_positions = self._optimize_position_mapping(regime_positions)
            signals_df['base_position'] = signals_df['regime'].map(optimized_positions)

            # Add dynamic position sizing based on confidence and market conditions
            signals_df['confidence'] = self._calculate_confidence(signals_df, regime_probs, min_length)
            signals_df['market_condition'] = self._assess_market_condition(price_data.iloc[:min_length])
            signals_df['volatility_adj'] = self._calculate_volatility_adjustment(price_data.iloc[:min_length])

            # Calculate final positions
            signals_df['position'] = self._calculate_final_positions(signals_df)

            # Add position change tracking for transaction costs
            signals_df['position_change'] = signals_df['position'].diff().abs().fillna(0)

            logger.info(f"Created {len(signals_df)} optimized trading signals")
            logger.info(f"Average position magnitude: {signals_df['position'].abs().mean():.2f}")

            return signals_df

        except Exception as e:
            logger.error(f"Error creating optimized signals: {str(e)}")
            raise

    def _optimize_position_mapping(self, regime_positions: Dict[str, float]) -> Dict[str, float]:
        """
        Optimize position sizes based on target performance
        Key insight: Need to be more aggressive in profitable regimes
        """
        optimized = {}

        for regime, base_position in regime_positions.items():
            if 'HighProfit' in regime or 'High' in regime:
                # Maximum leverage for high-profit regimes
                optimized[regime] = base_position * 2.0
            elif 'MediumProfit' in regime or 'Medium' in regime:
                # Moderate position for medium-profit regimes
                optimized[regime] = base_position * 1.2
            elif 'LowProfit' in regime or 'Low' in regime:
                # Small contrarian position for low-profit regimes
                optimized[regime] = -abs(base_position) * 0.5
            else:
                # Handle legacy naming (Bull/Bear/Sideways)
                if 'Bull' in regime:
                    optimized[regime] = 2.0  # Full leverage long
                elif 'Bear' in regime:
                    optimized[regime] = -1.5  # Strong short
                else:  # Sideways
                    optimized[regime] = 0.3   # Small long bias

        logger.info(f"Optimized position mapping: {optimized}")
        return optimized

    def _calculate_confidence(self, signals_df: pd.DataFrame, regime_probs: pd.DataFrame,
                            min_length: int) -> pd.Series:
        """Calculate confidence in regime predictions"""
        if regime_probs.empty or len(regime_probs) < min_length:
            return pd.Series(0.8, index=signals_df.index)  # Default high confidence

        confidence_series = pd.Series(0.8, index=signals_df.index)

        for i, regime in enumerate(signals_df['regime']):
            if regime in regime_probs.columns and i < len(regime_probs):
                confidence_series.iloc[i] = regime_probs[regime].iloc[i]

        return confidence_series

    def _assess_market_condition(self, prices: pd.Series) -> pd.Series:
        """
        Assess overall market condition for additional position sizing
        """
        # Calculate trend strength
        ma_short = prices.rolling(10).mean()
        ma_long = prices.rolling(50).mean()
        trend_strength = (ma_short - ma_long) / ma_long

        # Calculate momentum
        momentum = prices.pct_change(20)

        # Combine indicators
        market_condition = (trend_strength + momentum) / 2
        return market_condition.fillna(0)

    def _calculate_volatility_adjustment(self, prices: pd.Series) -> pd.Series:
        """
        Calculate volatility-based position adjustment
        """
        returns = prices.pct_change()
        rolling_vol = returns.rolling(20).std() * np.sqrt(252)

        # Target 15% volatility
        target_vol = 0.15

        # Inverse volatility scaling
        vol_adj = target_vol / rolling_vol.fillna(target_vol)
        vol_adj = vol_adj.clip(0.5, 2.5)  # Reasonable limits

        return vol_adj

    def _calculate_final_positions(self, signals_df: pd.DataFrame) -> pd.Series:
        """
        Calculate final position sizes with all adjustments
        """
        # Base position
        positions = signals_df['base_position'].copy()

        # Confidence adjustment (scale based on confidence)
        confidence_mult = 0.3 + 1.7 * signals_df['confidence']  # Range: 0.3 to 2.0
        positions *= confidence_mult

        # Market condition adjustment
        market_mult = 1.0 + 0.5 * signals_df['market_condition']  # Amplify good conditions
        positions *= market_mult

        # Volatility adjustment
        positions *= signals_df['volatility_adj']

        # Apply final position limits
        positions = positions.clip(-3.0, 3.0)  # Allow up to 3x leverage

        return positions

    def calculate_returns(self, signals_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate portfolio returns with optimized methodology
        """
        try:
            signals_df = signals_df.copy()

            # Calculate price returns
            signals_df['price_returns'] = signals_df['price'].pct_change()

            # Remove first row with NaN
            signals_df = signals_df.iloc[1:].copy()

            # Position lag for proper timing
            signals_df['position_lag'] = signals_df['position'].shift(1).fillna(0)

            # Calculate gross returns
            signals_df['gross_returns'] = signals_df['position_lag'] * signals_df['price_returns']

            # Transaction costs
            signals_df['transaction_costs'] = signals_df['position_change'].shift(1) * self.transaction_cost

            # Net returns
            signals_df['net_returns'] = signals_df['gross_returns'] - signals_df['transaction_costs'].fillna(0)

            # Cumulative performance
            signals_df['cumulative_gross'] = (1 + signals_df['gross_returns']).cumprod()
            signals_df['cumulative_net'] = (1 + signals_df['net_returns']).cumprod()

            # Benchmark
            signals_df['benchmark_returns'] = signals_df['price_returns']
            signals_df['benchmark_cumulative'] = (1 + signals_df['benchmark_returns']).cumprod()

            # Clean any NaN or inf values
            numeric_cols = ['gross_returns', 'net_returns', 'cumulative_gross', 'cumulative_net']
            for col in numeric_cols:
                signals_df[col] = signals_df[col].fillna(0).replace([np.inf, -np.inf], 0)

            logger.info(f"Calculated returns for {len(signals_df)} periods")

            return signals_df

        except Exception as e:
            logger.error(f"Error calculating returns: {str(e)}")
            raise

    def calculate_performance_metrics(self, portfolio_df: pd.DataFrame) -> Dict:
        """
        Calculate performance metrics with focus on target achievement
        """
        try:
            clean_df = portfolio_df.dropna(subset=['net_returns'])

            if len(clean_df) == 0:
                return {}

            net_returns = clean_df['net_returns']
            benchmark_returns = clean_df['benchmark_returns']

            # Core metrics
            total_return = clean_df['cumulative_net'].iloc[-1] - 1
            benchmark_return = clean_df['benchmark_cumulative'].iloc[-1] - 1

            # Annualized metrics
            n_years = len(clean_df) / 252
            annual_return = (1 + total_return) ** (1/n_years) - 1 if n_years > 0 else total_return
            annual_benchmark = (1 + benchmark_return) ** (1/n_years) - 1 if n_years > 0 else benchmark_return

            # Risk metrics
            volatility = net_returns.std() * np.sqrt(252) if len(net_returns) > 1 else 0
            benchmark_vol = benchmark_returns.std() * np.sqrt(252) if len(benchmark_returns) > 1 else 0

            # Sharpe ratio
            sharpe_ratio = annual_return / volatility if volatility > 0 else 0
            benchmark_sharpe = annual_benchmark / benchmark_vol if benchmark_vol > 0 else 0

            # Drawdown
            cumulative = clean_df['cumulative_net']
            rolling_max = cumulative.expanding().max()
            drawdowns = (cumulative - rolling_max) / rolling_max
            max_drawdown = drawdowns.min()

            # Other metrics
            excess_returns = net_returns - benchmark_returns
            tracking_error = excess_returns.std() * np.sqrt(252) if len(excess_returns) > 1 else 0
            information_ratio = excess_returns.mean() * np.sqrt(252) / tracking_error if tracking_error > 0 else 0

            win_rate = (net_returns > 0).mean()
            calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown < 0 else 0

            metrics = {
                'total_return': total_return,
                'annual_return': annual_return,
                'benchmark_return': benchmark_return,
                'excess_return': total_return - benchmark_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'benchmark_sharpe': benchmark_sharpe,
                'max_drawdown': max_drawdown,
                'information_ratio': information_ratio,
                'calmar_ratio': calmar_ratio,
                'win_rate': win_rate,
                'tracking_error': tracking_error,
                'total_trades': (clean_df['position_change'] > 0.01).sum(),
                'n_observations': len(clean_df),
                'n_years': n_years
            }

            # Target achievement analysis
            targets_met = {
                'return_target': total_return >= self.target_return,
                'sharpe_target': sharpe_ratio >= self.target_sharpe,
                'return_gap': total_return - self.target_return,
                'sharpe_gap': sharpe_ratio - self.target_sharpe
            }

            metrics['targets_met'] = targets_met

            logger.info(f"Performance calculated:")
            logger.info(f"  Total Return: {total_return:.2%} (Target: {self.target_return:.0%})")
            logger.info(f"  Sharpe Ratio: {sharpe_ratio:.2f} (Target: {self.target_sharpe:.1f})")

            return metrics

        except Exception as e:
            logger.error(f"Error calculating performance: {str(e)}")
            return {}

    def run_optimized_backtest(self, regime_predictions: Dict, price_data: pd.Series,
                              regime_positions: Dict[str, float]) -> Dict:
        """
        Run complete optimized backtest
        """
        try:
            logger.info("Running performance-optimized backtest...")

            # Create optimized signals
            signals = self.create_optimized_signals(regime_predictions, price_data, regime_positions)

            # Calculate returns
            portfolio = self.calculate_returns(signals)

            # Calculate metrics
            metrics = self.calculate_performance_metrics(portfolio)

            results = {
                'portfolio': portfolio,
                'metrics': metrics,
                'signals': signals
            }

            # Log results
            if metrics:
                targets = metrics.get('targets_met', {})
                logger.info("🎯 TARGET ACHIEVEMENT:")
                logger.info(f"  Return: {targets.get('return_target', False)} "
                           f"(Gap: {targets.get('return_gap', 0):.1%})")
                logger.info(f"  Sharpe: {targets.get('sharpe_target', False)} "
                           f"(Gap: {targets.get('sharpe_gap', 0):.2f})")

            return results

        except Exception as e:
            logger.error(f"Error in optimized backtest: {str(e)}")
            raise


def main():
    """Test the performance-optimized strategy"""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=1000, freq='D')

    # Create test data with realistic regime patterns
    prices_list = [100]
    regime_sequence = []

    for i in range(999):
        # Create regime-dependent returns
        if i < 400:  # High profit regime
            ret = np.random.normal(0.002, 0.015)  # 50% annual, 15% vol
            regime = 'HighProfit'
        elif i < 700:  # Low profit regime
            ret = np.random.normal(-0.001, 0.025)  # -25% annual, 25% vol
            regime = 'LowProfit'
        else:  # Medium profit regime
            ret = np.random.normal(0.0005, 0.018)  # 12.5% annual, 18% vol
            regime = 'MediumProfit'

        prices_list.append(prices_list[-1] * (1 + ret))
        regime_sequence.append(regime)

    prices = pd.Series(prices_list, index=dates)
    regime_predictions = {'regime_sequence': regime_sequence}
    regime_positions = {'HighProfit': 2.0, 'MediumProfit': 0.5, 'LowProfit': -1.0}

    # Test strategy
    strategy = PerformanceOptimizedStrategy()
    results = strategy.run_optimized_backtest(regime_predictions, prices, regime_positions)

    print("Performance-Optimized Strategy Test:")
    if results['metrics']:
        print(f"Total Return: {results['metrics']['total_return']:.1%}")
        print(f"Sharpe Ratio: {results['metrics']['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {results['metrics']['max_drawdown']:.1%}")

    return strategy, results


if __name__ == "__main__":
    main()