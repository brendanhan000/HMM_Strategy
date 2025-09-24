"""
Regime-based trading strategy implementation
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from config.settings import TRADING_CONFIG, REGIME_POSITIONS, REGIME_LABELS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RegimeTradingStrategy:
    """
    Regime-based trading strategy with volatility-adjusted position sizing
    """

    def __init__(self, transaction_cost: float = None, max_position: float = None,
                 min_position: float = None, risk_free_rate: float = None):
        """
        Initialize trading strategy

        Parameters:
        -----------
        transaction_cost : float, transaction cost per trade (default from config)
        max_position : float, maximum position size
        min_position : float, minimum position size
        risk_free_rate : float, annual risk-free rate
        """
        self.transaction_cost = transaction_cost or TRADING_CONFIG['transaction_cost']
        self.max_position = max_position or TRADING_CONFIG['max_position']
        self.min_position = min_position or TRADING_CONFIG['min_position']
        self.risk_free_rate = risk_free_rate or TRADING_CONFIG['risk_free_rate']

        self.positions = None
        self.trades = None
        self.performance_metrics = {}

    def generate_signals(self, regime_predictions: Dict[str, np.ndarray],
                        regime_probabilities: Optional[np.ndarray] = None,
                        volatility_forecast: Optional[pd.Series] = None,
                        confidence_threshold: float = 0.6) -> pd.DataFrame:
        """
        Generate trading signals based on regime predictions

        Parameters:
        -----------
        regime_predictions : dict containing regime state sequence
        regime_probabilities : array of regime probabilities
        volatility_forecast : series of volatility forecasts for position sizing
        confidence_threshold : minimum probability threshold for taking positions

        Returns:
        --------
        signals_df : DataFrame with trading signals and position sizes
        """
        try:
            state_sequence = regime_predictions['state_sequence']
            state_probs = regime_probabilities

            if state_probs is None and 'state_probabilities' in regime_predictions:
                state_probs = regime_predictions['state_probabilities']

            # Create signals DataFrame
            signals = []

            for i, regime in enumerate(state_sequence):
                signal = {
                    'regime': regime,
                    'regime_label': REGIME_LABELS.get(regime, f'Regime_{regime}'),
                    'base_position': REGIME_POSITIONS.get(regime, 0.0),
                    'confidence': 1.0,  # Default confidence
                    'volatility_adj': 1.0,  # Default volatility adjustment
                    'final_position': REGIME_POSITIONS.get(regime, 0.0)
                }

                # Add confidence if probabilities available
                if state_probs is not None and i < len(state_probs):
                    regime_prob = state_probs[i, regime]
                    signal['confidence'] = regime_prob

                    # Apply confidence threshold
                    if regime_prob < confidence_threshold:
                        signal['final_position'] = 0.0  # No position if low confidence

                # Apply volatility adjustment for position sizing
                if volatility_forecast is not None and i < len(volatility_forecast):
                    vol_adj = self._calculate_volatility_adjustment(volatility_forecast.iloc[i])
                    signal['volatility_adj'] = vol_adj
                    signal['final_position'] *= vol_adj

                # Apply position limits
                signal['final_position'] = np.clip(signal['final_position'],
                                                 self.min_position, self.max_position)

                signals.append(signal)

            signals_df = pd.DataFrame(signals)

            logger.info(f"Generated {len(signals_df)} trading signals")
            logger.info(f"Regime distribution: {signals_df['regime'].value_counts().to_dict()}")

            return signals_df

        except Exception as e:
            logger.error(f"Error generating signals: {str(e)}")
            raise

    def _calculate_volatility_adjustment(self, volatility: float,
                                       vol_target: float = 0.15,
                                       max_adjustment: float = 2.0) -> float:
        """
        Calculate position size adjustment based on volatility

        Parameters:
        -----------
        volatility : float, current volatility forecast
        vol_target : float, target volatility level
        max_adjustment : float, maximum adjustment factor

        Returns:
        --------
        adjustment : float, position size multiplier
        """
        if volatility <= 0:
            return max_adjustment

        # Inverse volatility scaling
        adjustment = vol_target / volatility

        # Apply limits
        adjustment = np.clip(adjustment, 1/max_adjustment, max_adjustment)

        return adjustment

    def calculate_portfolio_returns(self, signals_df: pd.DataFrame,
                                  price_data: pd.Series,
                                  include_costs: bool = True) -> pd.DataFrame:
        """
        Calculate portfolio returns based on trading signals

        Parameters:
        -----------
        signals_df : DataFrame with trading signals
        price_data : Series of price data aligned with signals
        include_costs : bool, whether to include transaction costs

        Returns:
        --------
        portfolio_df : DataFrame with portfolio performance
        """
        try:
            # Ensure data alignment
            min_length = min(len(signals_df), len(price_data))
            signals = signals_df.iloc[:min_length].copy()
            prices = price_data.iloc[:min_length].copy()

            # Calculate returns
            returns = prices.pct_change().fillna(0)

            # Initialize portfolio DataFrame
            portfolio = pd.DataFrame(index=prices.index)
            portfolio['price'] = prices
            portfolio['returns'] = returns
            portfolio['position'] = signals['final_position'].values
            portfolio['regime'] = signals['regime'].values
            portfolio['regime_label'] = signals['regime_label'].values

            # Calculate position changes for transaction costs
            portfolio['position_change'] = portfolio['position'].diff().fillna(portfolio['position'])
            portfolio['trade_flag'] = np.abs(portfolio['position_change']) > 1e-6

            # Calculate gross portfolio returns
            portfolio['gross_returns'] = portfolio['position'].shift(1) * portfolio['returns']

            # Calculate transaction costs
            if include_costs:
                portfolio['transaction_costs'] = (np.abs(portfolio['position_change']) *
                                                self.transaction_cost)
                portfolio['net_returns'] = (portfolio['gross_returns'] -
                                          portfolio['transaction_costs'])
            else:
                portfolio['transaction_costs'] = 0.0
                portfolio['net_returns'] = portfolio['gross_returns']

            # Calculate cumulative performance
            portfolio['cumulative_gross'] = (1 + portfolio['gross_returns']).cumprod()
            portfolio['cumulative_net'] = (1 + portfolio['net_returns']).cumprod()

            # Calculate benchmark (buy-and-hold)
            portfolio['benchmark_returns'] = returns
            portfolio['benchmark_cumulative'] = (1 + portfolio['benchmark_returns']).cumprod()

            logger.info("Calculated portfolio returns successfully")

            return portfolio

        except Exception as e:
            logger.error(f"Error calculating portfolio returns: {str(e)}")
            raise

    def calculate_trade_statistics(self, portfolio_df: pd.DataFrame) -> Dict:
        """
        Calculate detailed trade statistics

        Parameters:
        -----------
        portfolio_df : DataFrame with portfolio performance

        Returns:
        --------
        trade_stats : Dict with trade statistics
        """
        try:
            trades = self._extract_trades(portfolio_df)

            if len(trades) == 0:
                return {'num_trades': 0}

            # Calculate trade statistics
            trade_returns = [trade['return'] for trade in trades]
            trade_durations = [trade['duration'] for trade in trades]

            winning_trades = [r for r in trade_returns if r > 0]
            losing_trades = [r for r in trade_returns if r < 0]

            stats = {
                'num_trades': len(trades),
                'num_winning_trades': len(winning_trades),
                'num_losing_trades': len(losing_trades),
                'win_rate': len(winning_trades) / len(trades) if trades else 0,
                'avg_trade_return': np.mean(trade_returns),
                'avg_winning_trade': np.mean(winning_trades) if winning_trades else 0,
                'avg_losing_trade': np.mean(losing_trades) if losing_trades else 0,
                'avg_trade_duration': np.mean(trade_durations),
                'max_trade_return': max(trade_returns) if trade_returns else 0,
                'min_trade_return': min(trade_returns) if trade_returns else 0,
                'profit_factor': (abs(sum(winning_trades)) / abs(sum(losing_trades))
                                if losing_trades else float('inf')),
                'total_transaction_costs': portfolio_df['transaction_costs'].sum(),
            }

            # Calculate regime-specific trade statistics
            regime_stats = {}
            for regime_label in portfolio_df['regime_label'].unique():
                regime_mask = portfolio_df['regime_label'] == regime_label
                regime_trades = [t for t in trades if t['regime'] == regime_label]

                if regime_trades:
                    regime_returns = [t['return'] for t in regime_trades]
                    regime_stats[regime_label] = {
                        'num_trades': len(regime_trades),
                        'avg_return': np.mean(regime_returns),
                        'win_rate': sum(1 for r in regime_returns if r > 0) / len(regime_returns),
                        'total_return': sum(regime_returns)
                    }

            stats['regime_breakdown'] = regime_stats

            logger.info(f"Calculated statistics for {len(trades)} trades")

            return stats

        except Exception as e:
            logger.error(f"Error calculating trade statistics: {str(e)}")
            raise

    def _extract_trades(self, portfolio_df: pd.DataFrame) -> List[Dict]:
        """
        Extract individual trades from portfolio DataFrame
        """
        trades = []
        current_trade = None

        for i, row in portfolio_df.iterrows():
            position = row['position']

            # Check if we're entering a new position
            if current_trade is None and abs(position) > 1e-6:
                current_trade = {
                    'entry_date': i,
                    'entry_position': position,
                    'entry_price': row['price'],
                    'regime': row['regime_label'],
                    'cumulative_return': 0.0
                }

            # Check if we're in a trade
            elif current_trade is not None:
                # Update cumulative return
                if abs(position) > 1e-6:
                    current_trade['cumulative_return'] += row['net_returns']

                # Check if trade is closing
                if abs(position) < 1e-6 or np.sign(position) != np.sign(current_trade['entry_position']):
                    # Close the trade
                    current_trade['exit_date'] = i
                    current_trade['exit_price'] = row['price']
                    current_trade['duration'] = (i - current_trade['entry_date']).days
                    current_trade['return'] = current_trade['cumulative_return']

                    trades.append(current_trade)
                    current_trade = None

                    # Check if starting a new trade
                    if abs(position) > 1e-6:
                        current_trade = {
                            'entry_date': i,
                            'entry_position': position,
                            'entry_price': row['price'],
                            'regime': row['regime_label'],
                            'cumulative_return': 0.0
                        }

        # Close any remaining open trade
        if current_trade is not None:
            last_row = portfolio_df.iloc[-1]
            current_trade['exit_date'] = last_row.name
            current_trade['exit_price'] = last_row['price']
            current_trade['duration'] = (last_row.name - current_trade['entry_date']).days
            current_trade['return'] = current_trade['cumulative_return']
            trades.append(current_trade)

        return trades

    def generate_signals_with_risk_management(self, regime_predictions: Dict,
                                            price_data: pd.Series,
                                            volatility_forecast: pd.Series,
                                            max_drawdown: float = 0.1,
                                            stop_loss: float = 0.05) -> pd.DataFrame:
        """
        Generate signals with additional risk management rules

        Parameters:
        -----------
        regime_predictions : dict with regime predictions
        price_data : series of price data
        volatility_forecast : series of volatility forecasts
        max_drawdown : float, maximum allowed portfolio drawdown
        stop_loss : float, stop-loss threshold per trade

        Returns:
        --------
        signals_df : DataFrame with risk-adjusted signals
        """
        try:
            # Generate base signals
            signals_df = self.generate_signals(regime_predictions, volatility_forecast=volatility_forecast)

            # Calculate rolling performance for risk management
            portfolio_temp = self.calculate_portfolio_returns(signals_df, price_data)

            # Apply risk management adjustments
            for i in range(len(signals_df)):
                # Check portfolio drawdown
                if i > 0:
                    current_drawdown = self._calculate_current_drawdown(portfolio_temp.iloc[:i+1])
                    if current_drawdown > max_drawdown:
                        signals_df.loc[i, 'final_position'] *= 0.5  # Reduce position size

                # Apply stop-loss (simplified version)
                if i > 0:
                    recent_return = portfolio_temp['net_returns'].iloc[i]
                    if recent_return < -stop_loss:
                        signals_df.loc[i, 'final_position'] = 0.0  # Exit position

            logger.info("Applied risk management to trading signals")

            return signals_df

        except Exception as e:
            logger.error(f"Error in risk management: {str(e)}")
            return signals_df  # Return original signals if risk management fails

    def _calculate_current_drawdown(self, portfolio_subset: pd.DataFrame) -> float:
        """
        Calculate current drawdown from peak
        """
        cumulative = portfolio_subset['cumulative_net']
        peak = cumulative.expanding().max()
        drawdown = (cumulative.iloc[-1] - peak.iloc[-1]) / peak.iloc[-1]
        return abs(drawdown)


def main():
    """
    Example usage of RegimeTradingStrategy
    """
    # Create dummy data for demonstration
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=1000, freq='D')
    prices = pd.Series(100 * np.cumprod(1 + np.random.normal(0, 0.02, 1000)), index=dates)

    # Dummy regime predictions
    regime_predictions = {
        'state_sequence': np.random.choice([0, 1, 2], size=1000),
        'state_probabilities': np.random.dirichlet([1, 1, 1], size=1000)
    }

    # Dummy volatility forecast
    volatility_forecast = pd.Series(np.random.gamma(2, 0.01, 1000), index=dates)

    # Initialize strategy
    strategy = RegimeTradingStrategy()

    # Generate signals
    signals = strategy.generate_signals(regime_predictions, volatility_forecast=volatility_forecast)

    # Calculate portfolio returns
    portfolio = strategy.calculate_portfolio_returns(signals, prices)

    # Calculate trade statistics
    trade_stats = strategy.calculate_trade_statistics(portfolio)

    print("Trade Statistics:")
    for key, value in trade_stats.items():
        if key != 'regime_breakdown':
            print(f"{key}: {value}")

    return strategy, signals, portfolio, trade_stats


if __name__ == "__main__":
    main()