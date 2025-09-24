"""
Comprehensive backtesting framework for regime trading strategy
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from datetime import datetime, timedelta
from scipy import stats

from config.settings import PERFORMANCE_CONFIG, TRADING_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RegimeBacktester:
    """
    Comprehensive backtesting framework with walk-forward analysis
    """

    def __init__(self, initial_capital: float = 100000,
                 benchmark_symbol: str = None,
                 risk_free_rate: float = None):
        """
        Initialize backtester

        Parameters:
        -----------
        initial_capital : float, starting capital
        benchmark_symbol : str, benchmark symbol for comparison
        risk_free_rate : float, annual risk-free rate
        """
        self.initial_capital = initial_capital
        self.benchmark_symbol = benchmark_symbol or PERFORMANCE_CONFIG['benchmark']
        self.risk_free_rate = risk_free_rate or TRADING_CONFIG['risk_free_rate']
        self.periods_per_year = PERFORMANCE_CONFIG['periods_per_year']

        self.results = {}
        self.portfolio_history = None
        self.performance_metrics = {}

    def run_backtest(self, strategy, data: pd.DataFrame,
                    train_ratio: float = 0.7,
                    walk_forward: bool = True,
                    rebalance_frequency: str = 'daily') -> Dict:
        """
        Run comprehensive backtest

        Parameters:
        -----------
        strategy : trading strategy object
        data : DataFrame with price and features data
        train_ratio : float, proportion of data for initial training
        walk_forward : bool, whether to use walk-forward analysis
        rebalance_frequency : str, rebalancing frequency

        Returns:
        --------
        results : Dict with backtest results
        """
        try:
            logger.info("Starting comprehensive backtest...")

            if walk_forward:
                results = self._run_walk_forward_backtest(strategy, data, train_ratio)
            else:
                results = self._run_static_backtest(strategy, data, train_ratio)

            # Calculate comprehensive performance metrics
            self.performance_metrics = self._calculate_performance_metrics(results['portfolio'])

            # Store results
            self.results = {
                'portfolio': results['portfolio'],
                'trades': results.get('trades', []),
                'performance_metrics': self.performance_metrics,
                'regime_analysis': results.get('regime_analysis', {}),
                'walk_forward_results': results.get('walk_forward_results', [])
            }

            logger.info("Backtest completed successfully")

            return self.results

        except Exception as e:
            logger.error(f"Error in backtesting: {str(e)}")
            raise

    def _run_walk_forward_backtest(self, strategy, data: pd.DataFrame,
                                  initial_train_ratio: float = 0.7,
                                  window_size: int = 252,
                                  step_size: int = 21) -> Dict:
        """
        Run walk-forward backtesting with rolling window

        Parameters:
        -----------
        strategy : trading strategy object
        data : DataFrame with all required data
        initial_train_ratio : float, initial training data proportion
        window_size : int, training window size in periods
        step_size : int, forward step size in periods
        """
        logger.info("Running walk-forward backtest...")

        portfolio_results = []
        walk_forward_results = []
        all_trades = []

        # Initial training size
        initial_train_size = int(len(data) * initial_train_ratio)

        # Start walk-forward process
        start_idx = initial_train_size
        end_idx = min(start_idx + step_size, len(data))

        while end_idx <= len(data):
            logger.info(f"Processing period {start_idx} to {end_idx}")

            # Define training and testing windows
            train_start = max(0, start_idx - window_size)
            train_end = start_idx
            test_start = start_idx
            test_end = end_idx

            # Extract data windows
            train_data = data.iloc[train_start:train_end]
            test_data = data.iloc[test_start:test_end]

            try:
                # Run strategy for this period
                period_result = self._run_period_backtest(strategy, train_data, test_data)

                portfolio_results.append(period_result['portfolio'])
                all_trades.extend(period_result.get('trades', []))

                # Store walk-forward metrics
                wf_metrics = {
                    'period_start': test_data.index[0],
                    'period_end': test_data.index[-1],
                    'total_return': period_result['portfolio']['cumulative_net'].iloc[-1] - 1,
                    'sharpe_ratio': self._calculate_sharpe_ratio(period_result['portfolio']['net_returns']),
                    'max_drawdown': self._calculate_max_drawdown(period_result['portfolio']['cumulative_net']),
                    'num_trades': len(period_result.get('trades', []))
                }
                walk_forward_results.append(wf_metrics)

            except Exception as e:
                logger.warning(f"Error in period {start_idx}-{end_idx}: {str(e)}")

            # Move to next period
            start_idx = end_idx
            end_idx = min(start_idx + step_size, len(data))

        # Combine all portfolio results
        combined_portfolio = pd.concat(portfolio_results, ignore_index=False)

        return {
            'portfolio': combined_portfolio,
            'trades': all_trades,
            'walk_forward_results': walk_forward_results,
            'regime_analysis': self._analyze_regime_performance(combined_portfolio)
        }

    def _run_static_backtest(self, strategy, data: pd.DataFrame,
                           train_ratio: float = 0.7) -> Dict:
        """
        Run static backtest with single train/test split
        """
        logger.info("Running static backtest...")

        # Split data
        split_idx = int(len(data) * train_ratio)
        train_data = data.iloc[:split_idx]
        test_data = data.iloc[split_idx:]

        # Run backtest
        result = self._run_period_backtest(strategy, train_data, test_data)

        return {
            'portfolio': result['portfolio'],
            'trades': result.get('trades', []),
            'regime_analysis': self._analyze_regime_performance(result['portfolio'])
        }

    def _run_period_backtest(self, strategy, train_data: pd.DataFrame,
                           test_data: pd.DataFrame) -> Dict:
        """
        Run backtest for a specific period
        """
        try:
            # Import required modules (assumed to be available)
            from models.garch_model import GARCHModel
            from models.hmm_model import RegimeHMM

            # Fit GARCH model on training data
            garch_model = GARCHModel()
            garch_model.fit(train_data['Returns'])

            # Extract GARCH components for training
            train_components = garch_model.extract_components(train_data['Returns'])

            # Fit HMM model
            hmm_model = RegimeHMM()
            hmm_model.fit(train_components)

            # Generate predictions for test period
            test_components = garch_model.extract_components(test_data['Returns'])
            regime_predictions = hmm_model.predict_regimes(test_components)

            # Generate trading signals
            signals = strategy.generate_signals(
                regime_predictions,
                volatility_forecast=test_components.get('volatility_forecast')
            )

            # Calculate portfolio performance
            portfolio = strategy.calculate_portfolio_returns(
                signals, test_data['Close']
            )

            # Calculate trade statistics
            trades = strategy.calculate_trade_statistics(portfolio)

            return {
                'portfolio': portfolio,
                'trades': trades,
                'garch_model': garch_model,
                'hmm_model': hmm_model,
                'regime_predictions': regime_predictions
            }

        except Exception as e:
            logger.error(f"Error in period backtest: {str(e)}")
            raise

    def _calculate_performance_metrics(self, portfolio: pd.DataFrame) -> Dict:
        """
        Calculate comprehensive performance metrics
        """
        try:
            net_returns = portfolio['net_returns']
            gross_returns = portfolio['gross_returns']
            benchmark_returns = portfolio['benchmark_returns']
            cumulative_net = portfolio['cumulative_net']
            cumulative_benchmark = portfolio['benchmark_cumulative']

            metrics = {}

            # Basic returns metrics
            metrics['total_return'] = cumulative_net.iloc[-1] - 1
            metrics['benchmark_return'] = cumulative_benchmark.iloc[-1] - 1
            metrics['excess_return'] = metrics['total_return'] - metrics['benchmark_return']

            # Annualized metrics
            n_years = len(portfolio) / self.periods_per_year
            metrics['annualized_return'] = (1 + metrics['total_return']) ** (1/n_years) - 1
            metrics['annualized_benchmark'] = (1 + metrics['benchmark_return']) ** (1/n_years) - 1
            metrics['annualized_excess'] = metrics['annualized_return'] - metrics['annualized_benchmark']

            # Risk metrics
            metrics['volatility'] = net_returns.std() * np.sqrt(self.periods_per_year)
            metrics['benchmark_volatility'] = benchmark_returns.std() * np.sqrt(self.periods_per_year)

            # Risk-adjusted returns
            metrics['sharpe_ratio'] = self._calculate_sharpe_ratio(net_returns)
            metrics['benchmark_sharpe'] = self._calculate_sharpe_ratio(benchmark_returns)
            metrics['information_ratio'] = self._calculate_information_ratio(net_returns, benchmark_returns)

            # Drawdown metrics
            metrics['max_drawdown'] = self._calculate_max_drawdown(cumulative_net)
            metrics['benchmark_max_drawdown'] = self._calculate_max_drawdown(cumulative_benchmark)
            metrics['calmar_ratio'] = (metrics['annualized_return'] / abs(metrics['max_drawdown'])
                                     if metrics['max_drawdown'] != 0 else 0)

            # Additional risk metrics
            metrics['downside_deviation'] = self._calculate_downside_deviation(net_returns)
            metrics['sortino_ratio'] = ((metrics['annualized_return'] - self.risk_free_rate) /
                                      metrics['downside_deviation'] if metrics['downside_deviation'] > 0 else 0)

            # Value at Risk
            metrics['var_95'] = np.percentile(net_returns, 5)
            metrics['cvar_95'] = net_returns[net_returns <= metrics['var_95']].mean()

            # Beta and correlation
            metrics['beta'] = self._calculate_beta(net_returns, benchmark_returns)
            metrics['correlation'] = net_returns.corr(benchmark_returns)

            # Win/Loss metrics
            positive_returns = net_returns[net_returns > 0]
            negative_returns = net_returns[net_returns < 0]
            metrics['win_rate'] = len(positive_returns) / len(net_returns)
            metrics['avg_win'] = positive_returns.mean() if len(positive_returns) > 0 else 0
            metrics['avg_loss'] = negative_returns.mean() if len(negative_returns) > 0 else 0
            metrics['profit_factor'] = abs(positive_returns.sum() / negative_returns.sum()) if len(negative_returns) > 0 else float('inf')

            # Skewness and Kurtosis
            metrics['skewness'] = net_returns.skew()
            metrics['kurtosis'] = net_returns.kurtosis()

            # Transaction costs impact
            metrics['gross_return'] = portfolio['gross_returns'].sum()
            metrics['total_costs'] = portfolio['transaction_costs'].sum()
            metrics['cost_impact'] = metrics['total_costs'] / self.initial_capital

            # Monthly and yearly breakdowns
            metrics['monthly_returns'] = self._calculate_monthly_returns(portfolio)
            metrics['yearly_returns'] = self._calculate_yearly_returns(portfolio)

            return metrics

        except Exception as e:
            logger.error(f"Error calculating performance metrics: {str(e)}")
            return {}

    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate annualized Sharpe ratio"""
        if returns.std() == 0:
            return 0
        excess_return = returns.mean() - self.risk_free_rate / self.periods_per_year
        return excess_return / returns.std() * np.sqrt(self.periods_per_year)

    def _calculate_information_ratio(self, returns: pd.Series, benchmark: pd.Series) -> float:
        """Calculate Information Ratio"""
        excess_returns = returns - benchmark
        if excess_returns.std() == 0:
            return 0
        return excess_returns.mean() / excess_returns.std() * np.sqrt(self.periods_per_year)

    def _calculate_max_drawdown(self, cumulative_returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        return drawdown.min()

    def _calculate_downside_deviation(self, returns: pd.Series) -> float:
        """Calculate downside deviation"""
        downside_returns = returns[returns < 0]
        return downside_returns.std() * np.sqrt(self.periods_per_year)

    def _calculate_beta(self, returns: pd.Series, benchmark: pd.Series) -> float:
        """Calculate beta vs benchmark"""
        if benchmark.var() == 0:
            return 0
        return np.cov(returns, benchmark)[0, 1] / benchmark.var()

    def _calculate_monthly_returns(self, portfolio: pd.DataFrame) -> pd.Series:
        """Calculate monthly returns"""
        portfolio_monthly = portfolio.resample('M', on=portfolio.index)['cumulative_net'].last()
        monthly_returns = portfolio_monthly.pct_change().dropna()
        return monthly_returns

    def _calculate_yearly_returns(self, portfolio: pd.DataFrame) -> pd.Series:
        """Calculate yearly returns"""
        portfolio_yearly = portfolio.resample('Y', on=portfolio.index)['cumulative_net'].last()
        yearly_returns = portfolio_yearly.pct_change().dropna()
        return yearly_returns

    def _analyze_regime_performance(self, portfolio: pd.DataFrame) -> Dict:
        """
        Analyze performance by regime
        """
        regime_analysis = {}

        for regime_label in portfolio['regime_label'].unique():
            regime_mask = portfolio['regime_label'] == regime_label
            regime_data = portfolio[regime_mask]

            if len(regime_data) > 0:
                regime_analysis[regime_label] = {
                    'frequency': len(regime_data) / len(portfolio),
                    'avg_return': regime_data['net_returns'].mean(),
                    'volatility': regime_data['net_returns'].std(),
                    'sharpe_ratio': self._calculate_sharpe_ratio(regime_data['net_returns']),
                    'total_return': regime_data['net_returns'].sum(),
                    'win_rate': (regime_data['net_returns'] > 0).mean(),
                    'avg_position': regime_data['position'].mean()
                }

        return regime_analysis

    def generate_performance_report(self) -> str:
        """
        Generate comprehensive performance report
        """
        if not self.performance_metrics:
            return "No performance metrics available. Run backtest first."

        metrics = self.performance_metrics

        report = f"""
REGIME TRADING STRATEGY - PERFORMANCE REPORT
============================================

OVERALL PERFORMANCE:
- Total Return: {metrics['total_return']:.2%}
- Benchmark Return: {metrics['benchmark_return']:.2%}
- Excess Return: {metrics['excess_return']:.2%}
- Annualized Return: {metrics['annualized_return']:.2%}

RISK METRICS:
- Volatility: {metrics['volatility']:.2%}
- Sharpe Ratio: {metrics['sharpe_ratio']:.3f}
- Maximum Drawdown: {metrics['max_drawdown']:.2%}
- Calmar Ratio: {metrics['calmar_ratio']:.3f}
- Sortino Ratio: {metrics['sortino_ratio']:.3f}

TRADING METRICS:
- Win Rate: {metrics['win_rate']:.2%}
- Profit Factor: {metrics['profit_factor']:.2f}
- Average Win: {metrics['avg_win']:.4f}
- Average Loss: {metrics['avg_loss']:.4f}

RISK ANALYSIS:
- Beta vs Benchmark: {metrics['beta']:.3f}
- Correlation vs Benchmark: {metrics['correlation']:.3f}
- Value at Risk (95%): {metrics['var_95']:.4f}
- Conditional VaR (95%): {metrics['cvar_95']:.4f}

TRANSACTION COSTS:
- Total Costs: ${metrics['total_costs']:.2f}
- Cost Impact: {metrics['cost_impact']:.2%}
"""

        return report


def main():
    """
    Example usage of RegimeBacktester
    """
    # This would typically be used with real strategy and data
    backtester = RegimeBacktester(initial_capital=100000)

    print("RegimeBacktester initialized successfully")
    print("Use backtester.run_backtest(strategy, data) to run backtests")

    return backtester


if __name__ == "__main__":
    main()