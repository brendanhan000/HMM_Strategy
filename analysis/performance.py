"""
Advanced performance analytics and metrics calculation
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from scipy import stats
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns

from config.settings import PERFORMANCE_CONFIG, REGIME_LABELS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerformanceAnalyzer:
    """
    Advanced performance analytics for regime trading strategies
    """

    def __init__(self, risk_free_rate: float = None, periods_per_year: int = None):
        """
        Initialize performance analyzer

        Parameters:
        -----------
        risk_free_rate : float, annual risk-free rate
        periods_per_year : int, trading periods per year
        """
        self.risk_free_rate = risk_free_rate or 0.02
        self.periods_per_year = periods_per_year or PERFORMANCE_CONFIG['periods_per_year']

    def analyze_performance(self, portfolio: pd.DataFrame,
                          benchmark_returns: pd.Series = None) -> Dict:
        """
        Comprehensive performance analysis

        Parameters:
        -----------
        portfolio : DataFrame with portfolio performance data
        benchmark_returns : Series of benchmark returns

        Returns:
        --------
        analysis : Dict with comprehensive performance metrics
        """
        try:
            logger.info("Starting comprehensive performance analysis...")

            analysis = {}

            # Extract returns
            strategy_returns = portfolio['net_returns']
            if benchmark_returns is None:
                benchmark_returns = portfolio.get('benchmark_returns', portfolio['returns'])

            # Basic performance metrics
            analysis['basic_metrics'] = self._calculate_basic_metrics(
                strategy_returns, benchmark_returns, portfolio
            )

            # Risk metrics
            analysis['risk_metrics'] = self._calculate_risk_metrics(
                strategy_returns, benchmark_returns
            )

            # Regime-specific analysis
            analysis['regime_analysis'] = self._analyze_regime_performance(portfolio)

            # Time-series analysis
            analysis['time_series'] = self._analyze_time_series(portfolio)

            # Factor analysis
            analysis['factor_analysis'] = self._perform_factor_analysis(
                strategy_returns, benchmark_returns
            )

            # Rolling performance
            analysis['rolling_metrics'] = self._calculate_rolling_metrics(
                strategy_returns, benchmark_returns
            )

            # Statistical tests
            analysis['statistical_tests'] = self._perform_statistical_tests(
                strategy_returns, benchmark_returns
            )

            logger.info("Performance analysis completed successfully")

            return analysis

        except Exception as e:
            logger.error(f"Error in performance analysis: {str(e)}")
            raise

    def _calculate_basic_metrics(self, strategy_returns: pd.Series,
                                benchmark_returns: pd.Series,
                                portfolio: pd.DataFrame) -> Dict:
        """Calculate basic performance metrics"""
        cumulative_strategy = (1 + strategy_returns).cumprod()
        cumulative_benchmark = (1 + benchmark_returns).cumprod()

        n_years = len(strategy_returns) / self.periods_per_year

        metrics = {
            # Total returns
            'total_return': cumulative_strategy.iloc[-1] - 1,
            'benchmark_return': cumulative_benchmark.iloc[-1] - 1,
            'excess_return': (cumulative_strategy.iloc[-1] - 1) - (cumulative_benchmark.iloc[-1] - 1),

            # Annualized returns
            'annualized_return': (1 + (cumulative_strategy.iloc[-1] - 1)) ** (1/n_years) - 1,
            'annualized_benchmark': (1 + (cumulative_benchmark.iloc[-1] - 1)) ** (1/n_years) - 1,

            # Volatility
            'volatility': strategy_returns.std() * np.sqrt(self.periods_per_year),
            'benchmark_volatility': benchmark_returns.std() * np.sqrt(self.periods_per_year),

            # Count metrics
            'total_periods': len(strategy_returns),
            'trading_days': len(strategy_returns[strategy_returns != 0]),
            'years_analyzed': n_years,
        }

        # Add excess metrics
        metrics['annualized_excess'] = metrics['annualized_return'] - metrics['annualized_benchmark']
        metrics['volatility_ratio'] = metrics['volatility'] / metrics['benchmark_volatility']

        return metrics

    def _calculate_risk_metrics(self, strategy_returns: pd.Series,
                               benchmark_returns: pd.Series) -> Dict:
        """Calculate comprehensive risk metrics"""
        metrics = {}

        # Sharpe ratio
        excess_return = strategy_returns.mean() - self.risk_free_rate / self.periods_per_year
        metrics['sharpe_ratio'] = (excess_return / strategy_returns.std() *
                                 np.sqrt(self.periods_per_year) if strategy_returns.std() > 0 else 0)

        # Information ratio
        active_returns = strategy_returns - benchmark_returns
        metrics['information_ratio'] = (active_returns.mean() / active_returns.std() *
                                      np.sqrt(self.periods_per_year) if active_returns.std() > 0 else 0)

        # Drawdown metrics
        cumulative = (1 + strategy_returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdowns = (cumulative - rolling_max) / rolling_max

        metrics['max_drawdown'] = drawdowns.min()
        metrics['avg_drawdown'] = drawdowns[drawdowns < 0].mean() if (drawdowns < 0).any() else 0
        metrics['drawdown_duration'] = self._calculate_drawdown_duration(drawdowns)

        # Calmar ratio
        annualized_return = (1 + (cumulative.iloc[-1] - 1)) ** (self.periods_per_year/len(strategy_returns)) - 1
        metrics['calmar_ratio'] = annualized_return / abs(metrics['max_drawdown']) if metrics['max_drawdown'] < 0 else 0

        # Sortino ratio
        downside_returns = strategy_returns[strategy_returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(self.periods_per_year)
        metrics['sortino_ratio'] = ((annualized_return - self.risk_free_rate) /
                                  downside_deviation if downside_deviation > 0 else 0)

        # Value at Risk
        metrics['var_95'] = np.percentile(strategy_returns, 5)
        metrics['var_99'] = np.percentile(strategy_returns, 1)
        metrics['cvar_95'] = strategy_returns[strategy_returns <= metrics['var_95']].mean()
        metrics['cvar_99'] = strategy_returns[strategy_returns <= metrics['var_99']].mean()

        # Skewness and kurtosis
        metrics['skewness'] = strategy_returns.skew()
        metrics['kurtosis'] = strategy_returns.kurtosis()

        # Beta and correlation
        if benchmark_returns.var() > 0:
            metrics['beta'] = np.cov(strategy_returns, benchmark_returns)[0, 1] / benchmark_returns.var()
        else:
            metrics['beta'] = 0

        metrics['correlation'] = strategy_returns.corr(benchmark_returns)

        # Treynor ratio
        metrics['treynor_ratio'] = ((annualized_return - self.risk_free_rate) /
                                  metrics['beta'] if metrics['beta'] != 0 else 0)

        # Tracking error
        metrics['tracking_error'] = active_returns.std() * np.sqrt(self.periods_per_year)

        return metrics

    def _calculate_drawdown_duration(self, drawdowns: pd.Series) -> Dict:
        """Calculate drawdown duration statistics"""
        durations = []
        current_duration = 0

        for dd in drawdowns:
            if dd < 0:
                current_duration += 1
            else:
                if current_duration > 0:
                    durations.append(current_duration)
                current_duration = 0

        if current_duration > 0:
            durations.append(current_duration)

        if durations:
            return {
                'max_duration': max(durations),
                'avg_duration': np.mean(durations),
                'current_duration': current_duration
            }
        else:
            return {'max_duration': 0, 'avg_duration': 0, 'current_duration': 0}

    def _analyze_regime_performance(self, portfolio: pd.DataFrame) -> Dict:
        """Analyze performance by regime"""
        regime_analysis = {}

        if 'regime_label' not in portfolio.columns:
            return regime_analysis

        for regime_label in portfolio['regime_label'].unique():
            regime_mask = portfolio['regime_label'] == regime_label
            regime_data = portfolio[regime_mask]

            if len(regime_data) > 0:
                regime_returns = regime_data['net_returns']
                regime_positions = regime_data['position']

                regime_analysis[regime_label] = {
                    # Basic metrics
                    'frequency': len(regime_data) / len(portfolio),
                    'avg_return': regime_returns.mean(),
                    'total_return': regime_returns.sum(),
                    'volatility': regime_returns.std(),
                    'avg_position': regime_positions.mean(),

                    # Risk metrics
                    'sharpe_ratio': self._calculate_sharpe_ratio(regime_returns),
                    'max_drawdown': self._calculate_max_drawdown_simple(regime_returns),
                    'win_rate': (regime_returns > 0).mean(),
                    'profit_factor': self._calculate_profit_factor(regime_returns),

                    # Distribution metrics
                    'skewness': regime_returns.skew(),
                    'kurtosis': regime_returns.kurtosis(),
                    'var_95': np.percentile(regime_returns, 5),

                    # Transition analysis
                    'entry_returns': self._analyze_regime_transitions(portfolio, regime_label)
                }

        return regime_analysis

    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio for given returns"""
        if returns.std() == 0:
            return 0
        excess_return = returns.mean() - self.risk_free_rate / self.periods_per_year
        return excess_return / returns.std() * np.sqrt(self.periods_per_year)

    def _calculate_max_drawdown_simple(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown from returns"""
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        return drawdown.min()

    def _calculate_profit_factor(self, returns: pd.Series) -> float:
        """Calculate profit factor"""
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]

        if len(negative_returns) == 0:
            return float('inf')
        if len(positive_returns) == 0:
            return 0

        return abs(positive_returns.sum() / negative_returns.sum())

    def _analyze_regime_transitions(self, portfolio: pd.DataFrame, regime_label: str) -> Dict:
        """Analyze returns around regime transitions"""
        regime_mask = portfolio['regime_label'] == regime_label
        transitions = {}

        # Find regime entry points
        regime_entries = regime_mask & (~regime_mask.shift(1).fillna(False))
        entry_indices = portfolio.index[regime_entries]

        if len(entry_indices) > 0:
            # Analyze returns after regime entry (next 5 periods)
            post_entry_returns = []
            for entry_idx in entry_indices:
                entry_pos = portfolio.index.get_loc(entry_idx)
                if entry_pos + 5 < len(portfolio):
                    post_returns = portfolio['net_returns'].iloc[entry_pos:entry_pos+5]
                    post_entry_returns.append(post_returns.sum())

            transitions['post_entry_returns'] = {
                'mean': np.mean(post_entry_returns) if post_entry_returns else 0,
                'std': np.std(post_entry_returns) if post_entry_returns else 0,
                'win_rate': np.mean([r > 0 for r in post_entry_returns]) if post_entry_returns else 0
            }

        return transitions

    def _analyze_time_series(self, portfolio: pd.DataFrame) -> Dict:
        """Analyze time series properties"""
        returns = portfolio['net_returns']

        analysis = {
            # Monthly analysis
            'monthly_performance': self._analyze_monthly_performance(portfolio),

            # Yearly analysis
            'yearly_performance': self._analyze_yearly_performance(portfolio),

            # Seasonality
            'seasonality': self._analyze_seasonality(portfolio),

            # Autocorrelation
            'autocorrelation': self._analyze_autocorrelation(returns),

            # Regime persistence
            'regime_persistence': self._analyze_regime_persistence(portfolio)
        }

        return analysis

    def _analyze_monthly_performance(self, portfolio: pd.DataFrame) -> Dict:
        """Analyze monthly performance patterns"""
        portfolio_monthly = portfolio.resample('M')['cumulative_net'].last()
        monthly_returns = portfolio_monthly.pct_change().dropna()

        return {
            'avg_monthly_return': monthly_returns.mean(),
            'monthly_volatility': monthly_returns.std(),
            'best_month': monthly_returns.max(),
            'worst_month': monthly_returns.min(),
            'positive_months': (monthly_returns > 0).mean(),
            'monthly_sharpe': self._calculate_sharpe_ratio(monthly_returns * 12)  # Annualized
        }

    def _analyze_yearly_performance(self, portfolio: pd.DataFrame) -> Dict:
        """Analyze yearly performance patterns"""
        portfolio_yearly = portfolio.resample('Y')['cumulative_net'].last()
        yearly_returns = portfolio_yearly.pct_change().dropna()

        if len(yearly_returns) > 0:
            return {
                'avg_yearly_return': yearly_returns.mean(),
                'yearly_volatility': yearly_returns.std(),
                'best_year': yearly_returns.max(),
                'worst_year': yearly_returns.min(),
                'positive_years': (yearly_returns > 0).mean(),
                'years_analyzed': len(yearly_returns)
            }
        else:
            return {'years_analyzed': 0}

    def _analyze_seasonality(self, portfolio: pd.DataFrame) -> Dict:
        """Analyze seasonal patterns"""
        returns = portfolio['net_returns']

        seasonality = {
            'by_month': returns.groupby(returns.index.month).mean().to_dict(),
            'by_quarter': returns.groupby(returns.index.quarter).mean().to_dict(),
            'by_day_of_week': returns.groupby(returns.index.dayofweek).mean().to_dict()
        }

        return seasonality

    def _analyze_autocorrelation(self, returns: pd.Series) -> Dict:
        """Analyze return autocorrelation"""
        autocorr = {}

        for lag in [1, 2, 3, 5, 10, 20]:
            if len(returns) > lag:
                autocorr[f'lag_{lag}'] = returns.autocorr(lag=lag)

        return autocorr

    def _analyze_regime_persistence(self, portfolio: pd.DataFrame) -> Dict:
        """Analyze regime persistence and transitions"""
        if 'regime' not in portfolio.columns:
            return {}

        regimes = portfolio['regime']
        transitions = {}

        # Calculate transition matrix
        transition_counts = {}
        for i in range(len(regimes) - 1):
            current = regimes.iloc[i]
            next_regime = regimes.iloc[i + 1]
            key = f"{current}_to_{next_regime}"
            transition_counts[key] = transition_counts.get(key, 0) + 1

        # Calculate average regime duration
        regime_durations = {}
        current_regime = regimes.iloc[0]
        current_duration = 1

        for i in range(1, len(regimes)):
            if regimes.iloc[i] == current_regime:
                current_duration += 1
            else:
                if current_regime not in regime_durations:
                    regime_durations[current_regime] = []
                regime_durations[current_regime].append(current_duration)
                current_regime = regimes.iloc[i]
                current_duration = 1

        # Add final duration
        if current_regime not in regime_durations:
            regime_durations[current_regime] = []
        regime_durations[current_regime].append(current_duration)

        return {
            'transition_counts': transition_counts,
            'avg_durations': {k: np.mean(v) for k, v in regime_durations.items()},
            'max_durations': {k: max(v) for k, v in regime_durations.items()}
        }

    def _perform_factor_analysis(self, strategy_returns: pd.Series,
                                benchmark_returns: pd.Series) -> Dict:
        """Perform factor analysis"""
        # Simple single-factor model (CAPM)
        excess_strategy = strategy_returns - self.risk_free_rate / self.periods_per_year
        excess_benchmark = benchmark_returns - self.risk_free_rate / self.periods_per_year

        if len(excess_strategy) > 1 and excess_benchmark.var() > 0:
            # CAPM regression
            beta = np.cov(excess_strategy, excess_benchmark)[0, 1] / excess_benchmark.var()
            alpha = excess_strategy.mean() - beta * excess_benchmark.mean()

            # R-squared
            correlation = np.corrcoef(excess_strategy, excess_benchmark)[0, 1]
            r_squared = correlation ** 2

            return {
                'alpha': alpha * self.periods_per_year,  # Annualized
                'beta': beta,
                'r_squared': r_squared,
                'correlation': correlation
            }
        else:
            return {'alpha': 0, 'beta': 0, 'r_squared': 0, 'correlation': 0}

    def _calculate_rolling_metrics(self, strategy_returns: pd.Series,
                                  benchmark_returns: pd.Series,
                                  window: int = 252) -> Dict:
        """Calculate rolling performance metrics"""
        if len(strategy_returns) < window:
            return {}

        rolling_metrics = {}

        # Rolling Sharpe ratio
        rolling_sharpe = strategy_returns.rolling(window).apply(
            lambda x: self._calculate_sharpe_ratio(x)
        )

        # Rolling correlation
        rolling_corr = strategy_returns.rolling(window).corr(benchmark_returns)

        # Rolling volatility
        rolling_vol = strategy_returns.rolling(window).std() * np.sqrt(self.periods_per_year)

        rolling_metrics = {
            'rolling_sharpe': {
                'mean': rolling_sharpe.mean(),
                'std': rolling_sharpe.std(),
                'min': rolling_sharpe.min(),
                'max': rolling_sharpe.max()
            },
            'rolling_correlation': {
                'mean': rolling_corr.mean(),
                'std': rolling_corr.std(),
                'min': rolling_corr.min(),
                'max': rolling_corr.max()
            },
            'rolling_volatility': {
                'mean': rolling_vol.mean(),
                'std': rolling_vol.std(),
                'min': rolling_vol.min(),
                'max': rolling_vol.max()
            }
        }

        return rolling_metrics

    def _perform_statistical_tests(self, strategy_returns: pd.Series,
                                  benchmark_returns: pd.Series) -> Dict:
        """Perform statistical tests"""
        tests = {}

        # Normality test (Jarque-Bera)
        jb_stat, jb_pvalue = stats.jarque_bera(strategy_returns.dropna())
        tests['normality_test'] = {
            'statistic': jb_stat,
            'p_value': jb_pvalue,
            'is_normal': jb_pvalue > 0.05
        }

        # Stationarity test (Augmented Dickey-Fuller)
        try:
            from statsmodels.tsa.stattools import adfuller
            adf_stat, adf_pvalue, _, _, _, _ = adfuller(strategy_returns.dropna())
            tests['stationarity_test'] = {
                'statistic': adf_stat,
                'p_value': adf_pvalue,
                'is_stationary': adf_pvalue < 0.05
            }
        except ImportError:
            logger.warning("statsmodels not available for stationarity test")

        # Test for difference from benchmark
        if len(strategy_returns) == len(benchmark_returns):
            t_stat, t_pvalue = stats.ttest_rel(strategy_returns, benchmark_returns)
            tests['difference_test'] = {
                'statistic': t_stat,
                'p_value': t_pvalue,
                'significantly_different': t_pvalue < 0.05
            }

        return tests

    def generate_performance_summary(self, analysis: Dict) -> str:
        """Generate a comprehensive performance summary report"""
        if not analysis:
            return "No analysis data available."

        basic = analysis.get('basic_metrics', {})
        risk = analysis.get('risk_metrics', {})
        regime = analysis.get('regime_analysis', {})

        summary = f"""
COMPREHENSIVE PERFORMANCE ANALYSIS REPORT
==========================================

BASIC PERFORMANCE METRICS:
- Total Return: {basic.get('total_return', 0):.2%}
- Annualized Return: {basic.get('annualized_return', 0):.2%}
- Benchmark Return: {basic.get('benchmark_return', 0):.2%}
- Excess Return: {basic.get('excess_return', 0):.2%}
- Volatility: {basic.get('volatility', 0):.2%}

RISK-ADJUSTED METRICS:
- Sharpe Ratio: {risk.get('sharpe_ratio', 0):.3f}
- Information Ratio: {risk.get('information_ratio', 0):.3f}
- Sortino Ratio: {risk.get('sortino_ratio', 0):.3f}
- Calmar Ratio: {risk.get('calmar_ratio', 0):.3f}
- Maximum Drawdown: {risk.get('max_drawdown', 0):.2%}

RISK METRICS:
- Beta: {risk.get('beta', 0):.3f}
- Correlation: {risk.get('correlation', 0):.3f}
- Tracking Error: {risk.get('tracking_error', 0):.2%}
- VaR (95%): {risk.get('var_95', 0):.4f}
- CVaR (95%): {risk.get('cvar_95', 0):.4f}

REGIME ANALYSIS:
"""

        for regime_name, regime_stats in regime.items():
            summary += f"\n{regime_name}:\n"
            summary += f"  - Frequency: {regime_stats.get('frequency', 0):.1%}\n"
            summary += f"  - Avg Return: {regime_stats.get('avg_return', 0):.4f}\n"
            summary += f"  - Sharpe Ratio: {regime_stats.get('sharpe_ratio', 0):.3f}\n"
            summary += f"  - Win Rate: {regime_stats.get('win_rate', 0):.1%}\n"

        return summary


def main():
    """
    Example usage of PerformanceAnalyzer
    """
    analyzer = PerformanceAnalyzer()
    print("PerformanceAnalyzer initialized successfully")
    print("Use analyzer.analyze_performance(portfolio_df) for comprehensive analysis")

    return analyzer


if __name__ == "__main__":
    main()