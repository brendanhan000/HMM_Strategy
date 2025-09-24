"""
Comprehensive visualization suite for regime detection and trading performance
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import logging
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from config.settings import VIZ_CONFIG, REGIME_COLORS, REGIME_LABELS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RegimeVisualizer:
    """
    Comprehensive visualization suite for regime detection and trading analysis
    """

    def __init__(self, style: str = None, figsize: Tuple = None, dpi: int = None):
        """
        Initialize visualizer

        Parameters:
        -----------
        style : str, matplotlib style
        figsize : tuple, default figure size
        dpi : int, figure DPI
        """
        self.style = style or VIZ_CONFIG.get('style', 'seaborn-v0_8')
        self.figsize = figsize or VIZ_CONFIG.get('figsize', (15, 10))
        self.dpi = dpi or VIZ_CONFIG.get('dpi', 300)

        # Set up matplotlib style
        plt.style.use('default')  # Reset to default first
        sns.set_palette("husl")

        # Color scheme for regimes
        self.regime_colors = REGIME_COLORS

    def plot_regime_detection(self, price_data: pd.Series, regime_sequence: np.ndarray,
                            regime_probabilities: Optional[np.ndarray] = None,
                            title: str = "Regime Detection Results") -> plt.Figure:
        """
        Plot price data with regime overlay

        Parameters:
        -----------
        price_data : Series of price data
        regime_sequence : Array of regime classifications
        regime_probabilities : Optional array of regime probabilities
        title : str, plot title

        Returns:
        --------
        fig : matplotlib Figure
        """
        try:
            # Ensure data alignment
            min_length = min(len(price_data), len(regime_sequence))
            prices = price_data.iloc[:min_length]
            regimes = regime_sequence[:min_length]

            # Create figure with subplots
            if regime_probabilities is not None:
                fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=self.figsize, dpi=self.dpi,
                                                  gridspec_kw={'height_ratios': [3, 1, 2]})
            else:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize, dpi=self.dpi,
                                             gridspec_kw={'height_ratios': [3, 1]})

            # Plot 1: Price with regime background
            ax1.plot(prices.index, prices.values, color='black', linewidth=1, alpha=0.8)

            # Add regime background colors
            self._add_regime_background(ax1, prices.index, regimes)

            ax1.set_title(title, fontsize=16, fontweight='bold')
            ax1.set_ylabel('Price ($)', fontsize=12)
            ax1.grid(True, alpha=0.3)
            ax1.legend(['Price'] + [f'Regime {i} ({REGIME_LABELS.get(i, i)})'
                                   for i in np.unique(regimes)])

            # Plot 2: Regime sequence
            regime_colors_mapped = [self.regime_colors.get(r, 'gray') for r in regimes]
            ax2.scatter(prices.index, regimes, c=regime_colors_mapped, alpha=0.7, s=10)
            ax2.set_ylabel('Regime', fontsize=12)
            ax2.set_yticks(range(len(REGIME_LABELS)))
            ax2.set_yticklabels([REGIME_LABELS.get(i, f'Regime {i}') for i in range(len(REGIME_LABELS))])
            ax2.grid(True, alpha=0.3)

            # Plot 3: Regime probabilities (if available)
            if regime_probabilities is not None:
                prob_length = min(len(prices), len(regime_probabilities))
                for i in range(regime_probabilities.shape[1]):
                    ax3.plot(prices.index[:prob_length],
                            regime_probabilities[:prob_length, i],
                            label=f'P(Regime {i})',
                            color=self.regime_colors.get(i, f'C{i}'),
                            alpha=0.8)

                ax3.set_ylabel('Probability', fontsize=12)
                ax3.set_xlabel('Date', fontsize=12)
                ax3.legend()
                ax3.grid(True, alpha=0.3)
                ax3.set_ylim(0, 1)

            plt.tight_layout()

            logger.info("Regime detection plot created successfully")
            return fig

        except Exception as e:
            logger.error(f"Error creating regime detection plot: {str(e)}")
            raise

    def plot_portfolio_performance(self, portfolio: pd.DataFrame,
                                 title: str = "Portfolio Performance Analysis") -> plt.Figure:
        """
        Plot comprehensive portfolio performance

        Parameters:
        -----------
        portfolio : DataFrame with portfolio performance data
        title : str, plot title

        Returns:
        --------
        fig : matplotlib Figure
        """
        try:
            fig, axes = plt.subplots(2, 2, figsize=self.figsize, dpi=self.dpi)
            fig.suptitle(title, fontsize=16, fontweight='bold')

            # Plot 1: Cumulative returns
            ax1 = axes[0, 0]
            ax1.plot(portfolio.index, portfolio['cumulative_net'],
                    label='Strategy', color='blue', linewidth=2)
            if 'benchmark_cumulative' in portfolio.columns:
                ax1.plot(portfolio.index, portfolio['benchmark_cumulative'],
                        label='Benchmark', color='red', linewidth=2, alpha=0.7)

            ax1.set_title('Cumulative Returns', fontweight='bold')
            ax1.set_ylabel('Cumulative Return')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Plot 2: Drawdown
            ax2 = axes[0, 1]
            cumulative = portfolio['cumulative_net']
            rolling_max = cumulative.expanding().max()
            drawdown = (cumulative - rolling_max) / rolling_max

            ax2.fill_between(portfolio.index, drawdown, 0, alpha=0.3, color='red')
            ax2.plot(portfolio.index, drawdown, color='red', linewidth=1)
            ax2.set_title('Drawdown', fontweight='bold')
            ax2.set_ylabel('Drawdown (%)')
            ax2.grid(True, alpha=0.3)

            # Plot 3: Rolling Sharpe ratio
            ax3 = axes[1, 0]
            rolling_returns = portfolio['net_returns'].rolling(window=252)
            rolling_sharpe = rolling_returns.mean() / rolling_returns.std() * np.sqrt(252)

            ax3.plot(portfolio.index, rolling_sharpe, color='green', linewidth=1)
            ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax3.set_title('Rolling Sharpe Ratio (1Y)', fontweight='bold')
            ax3.set_ylabel('Sharpe Ratio')
            ax3.grid(True, alpha=0.3)

            # Plot 4: Position and regime overlay
            ax4 = axes[1, 1]
            ax4.plot(portfolio.index, portfolio['position'],
                    color='purple', linewidth=1, alpha=0.8)

            # Add regime background if available
            if 'regime' in portfolio.columns:
                self._add_regime_background(ax4, portfolio.index, portfolio['regime'].values)

            ax4.set_title('Position Sizing', fontweight='bold')
            ax4.set_ylabel('Position')
            ax4.set_xlabel('Date')
            ax4.grid(True, alpha=0.3)

            plt.tight_layout()

            logger.info("Portfolio performance plot created successfully")
            return fig

        except Exception as e:
            logger.error(f"Error creating portfolio performance plot: {str(e)}")
            raise

    def plot_regime_analysis(self, regime_analysis: Dict,
                           title: str = "Regime Performance Analysis") -> plt.Figure:
        """
        Plot regime-specific performance analysis

        Parameters:
        -----------
        regime_analysis : Dict with regime performance metrics
        title : str, plot title

        Returns:
        --------
        fig : matplotlib Figure
        """
        try:
            fig, axes = plt.subplots(2, 2, figsize=self.figsize, dpi=self.dpi)
            fig.suptitle(title, fontsize=16, fontweight='bold')

            regimes = list(regime_analysis.keys())

            # Plot 1: Average returns by regime
            ax1 = axes[0, 0]
            avg_returns = [regime_analysis[r].get('avg_return', 0) for r in regimes]
            colors = [self.regime_colors.get(i, f'C{i}') for i in range(len(regimes))]

            bars1 = ax1.bar(regimes, avg_returns, color=colors, alpha=0.7)
            ax1.set_title('Average Returns by Regime', fontweight='bold')
            ax1.set_ylabel('Average Return')
            ax1.grid(True, alpha=0.3, axis='y')

            # Add value labels on bars
            for bar, value in zip(bars1, avg_returns):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + value*0.1,
                        f'{value:.4f}', ha='center', va='bottom')

            # Plot 2: Sharpe ratios by regime
            ax2 = axes[0, 1]
            sharpe_ratios = [regime_analysis[r].get('sharpe_ratio', 0) for r in regimes]

            bars2 = ax2.bar(regimes, sharpe_ratios, color=colors, alpha=0.7)
            ax2.set_title('Sharpe Ratios by Regime', fontweight='bold')
            ax2.set_ylabel('Sharpe Ratio')
            ax2.grid(True, alpha=0.3, axis='y')
            ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)

            # Plot 3: Win rates by regime
            ax3 = axes[1, 0]
            win_rates = [regime_analysis[r].get('win_rate', 0) for r in regimes]

            bars3 = ax3.bar(regimes, win_rates, color=colors, alpha=0.7)
            ax3.set_title('Win Rates by Regime', fontweight='bold')
            ax3.set_ylabel('Win Rate')
            ax3.set_ylim(0, 1)
            ax3.grid(True, alpha=0.3, axis='y')

            # Plot 4: Frequency of regimes
            ax4 = axes[1, 1]
            frequencies = [regime_analysis[r].get('frequency', 0) for r in regimes]

            ax4.pie(frequencies, labels=regimes, colors=colors, autopct='%1.1f%%',
                   startangle=90)
            ax4.set_title('Regime Frequency Distribution', fontweight='bold')

            plt.tight_layout()

            logger.info("Regime analysis plot created successfully")
            return fig

        except Exception as e:
            logger.error(f"Error creating regime analysis plot: {str(e)}")
            raise

    def plot_risk_metrics(self, performance_metrics: Dict,
                         title: str = "Risk Metrics Dashboard") -> plt.Figure:
        """
        Plot comprehensive risk metrics dashboard

        Parameters:
        -----------
        performance_metrics : Dict with performance metrics
        title : str, plot title

        Returns:
        --------
        fig : matplotlib Figure
        """
        try:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12), dpi=self.dpi)
            fig.suptitle(title, fontsize=16, fontweight='bold')

            risk_metrics = performance_metrics.get('risk_metrics', {})
            basic_metrics = performance_metrics.get('basic_metrics', {})

            # Plot 1: Risk-Return Scatter
            ax1 = axes[0, 0]
            strategy_return = basic_metrics.get('annualized_return', 0)
            strategy_vol = basic_metrics.get('volatility', 0)
            benchmark_return = basic_metrics.get('annualized_benchmark', 0)
            benchmark_vol = basic_metrics.get('benchmark_volatility', 0)

            ax1.scatter(strategy_vol, strategy_return, color='blue', s=100, label='Strategy')
            ax1.scatter(benchmark_vol, benchmark_return, color='red', s=100, label='Benchmark')
            ax1.set_xlabel('Volatility')
            ax1.set_ylabel('Annualized Return')
            ax1.set_title('Risk-Return Profile', fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Plot 2: Sharpe and Sortino ratios
            ax2 = axes[0, 1]
            ratios = ['Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio', 'Information Ratio']
            values = [
                risk_metrics.get('sharpe_ratio', 0),
                risk_metrics.get('sortino_ratio', 0),
                risk_metrics.get('calmar_ratio', 0),
                risk_metrics.get('information_ratio', 0)
            ]

            bars = ax2.bar(ratios, values, alpha=0.7)
            ax2.set_title('Risk-Adjusted Returns', fontweight='bold')
            ax2.set_ylabel('Ratio')
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(True, alpha=0.3, axis='y')
            ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)

            # Plot 3: VaR metrics
            ax3 = axes[0, 2]
            var_metrics = ['VaR 95%', 'VaR 99%', 'CVaR 95%', 'CVaR 99%']
            var_values = [
                risk_metrics.get('var_95', 0),
                risk_metrics.get('var_99', 0),
                risk_metrics.get('cvar_95', 0),
                risk_metrics.get('cvar_99', 0)
            ]

            colors = ['red', 'darkred', 'orange', 'darkorange']
            bars = ax3.bar(var_metrics, var_values, color=colors, alpha=0.7)
            ax3.set_title('Value at Risk Metrics', fontweight='bold')
            ax3.set_ylabel('VaR Value')
            ax3.tick_params(axis='x', rotation=45)
            ax3.grid(True, alpha=0.3, axis='y')

            # Plot 4: Distribution analysis
            ax4 = axes[1, 0]
            skewness = risk_metrics.get('skewness', 0)
            kurtosis = risk_metrics.get('kurtosis', 0)

            ax4.bar(['Skewness', 'Excess Kurtosis'], [skewness, kurtosis],
                   color=['purple', 'orange'], alpha=0.7)
            ax4.set_title('Return Distribution', fontweight='bold')
            ax4.set_ylabel('Value')
            ax4.grid(True, alpha=0.3, axis='y')
            ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)

            # Plot 5: Beta and correlation
            ax5 = axes[1, 1]
            beta = risk_metrics.get('beta', 0)
            correlation = risk_metrics.get('correlation', 0)

            ax5.bar(['Beta', 'Correlation'], [beta, correlation],
                   color=['green', 'blue'], alpha=0.7)
            ax5.set_title('Market Relationship', fontweight='bold')
            ax5.set_ylabel('Value')
            ax5.grid(True, alpha=0.3, axis='y')
            ax5.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax5.axhline(y=1, color='black', linestyle='--', alpha=0.5)

            # Plot 6: Drawdown statistics
            ax6 = axes[1, 2]
            max_dd = risk_metrics.get('max_drawdown', 0)
            avg_dd = risk_metrics.get('avg_drawdown', 0)

            ax6.bar(['Max Drawdown', 'Avg Drawdown'], [max_dd, avg_dd],
                   color=['red', 'orange'], alpha=0.7)
            ax6.set_title('Drawdown Analysis', fontweight='bold')
            ax6.set_ylabel('Drawdown')
            ax6.grid(True, alpha=0.3, axis='y')

            plt.tight_layout()

            logger.info("Risk metrics plot created successfully")
            return fig

        except Exception as e:
            logger.error(f"Error creating risk metrics plot: {str(e)}")
            raise

    def plot_regime_forecast(self, forecast_results: Dict,
                           title: str = "Regime Forecast") -> plt.Figure:
        """
        Plot regime forecasting results

        Parameters:
        -----------
        forecast_results : Dict with forecast results
        title : str, plot title

        Returns:
        --------
        fig : matplotlib Figure
        """
        try:
            forecast_probs = forecast_results.get('forecast_probabilities', np.array([]))
            most_likely = forecast_results.get('most_likely_regimes', np.array([]))
            horizon = forecast_results.get('horizon', 20)

            if len(forecast_probs) == 0:
                raise ValueError("No forecast data available")

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize, dpi=self.dpi)
            fig.suptitle(title, fontsize=16, fontweight='bold')

            # Create future dates for x-axis
            future_dates = pd.date_range(start=pd.Timestamp.now(), periods=horizon, freq='D')

            # Plot 1: Regime probabilities
            for i in range(forecast_probs.shape[1]):
                ax1.plot(future_dates, forecast_probs[:, i],
                        label=f'P(Regime {i}: {REGIME_LABELS.get(i, i)})',
                        color=self.regime_colors.get(i, f'C{i}'),
                        linewidth=2, alpha=0.8)

            ax1.fill_between(future_dates, 0, 1, alpha=0.1, color='gray')
            ax1.set_title('Regime Probability Forecasts', fontweight='bold')
            ax1.set_ylabel('Probability')
            ax1.set_ylim(0, 1)
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Plot 2: Most likely regime sequence
            regime_colors_mapped = [self.regime_colors.get(r, 'gray') for r in most_likely]
            ax2.scatter(future_dates, most_likely, c=regime_colors_mapped, s=50, alpha=0.8)
            ax2.plot(future_dates, most_likely, color='black', alpha=0.5, linewidth=1)

            ax2.set_title('Most Likely Regime Sequence', fontweight='bold')
            ax2.set_ylabel('Regime')
            ax2.set_xlabel('Date')
            ax2.set_yticks(range(len(REGIME_LABELS)))
            ax2.set_yticklabels([REGIME_LABELS.get(i, f'Regime {i}')
                               for i in range(len(REGIME_LABELS))])
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()

            logger.info("Regime forecast plot created successfully")
            return fig

        except Exception as e:
            logger.error(f"Error creating regime forecast plot: {str(e)}")
            raise

    def create_interactive_dashboard(self, portfolio: pd.DataFrame,
                                   regime_sequence: np.ndarray,
                                   performance_metrics: Dict) -> go.Figure:
        """
        Create interactive Plotly dashboard

        Parameters:
        -----------
        portfolio : DataFrame with portfolio data
        regime_sequence : Array of regime classifications
        performance_metrics : Dict with performance metrics

        Returns:
        --------
        fig : Plotly Figure
        """
        try:
            # Create subplots
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=('Price & Regimes', 'Cumulative Returns',
                              'Positions', 'Drawdown',
                              'Rolling Sharpe', 'Returns Distribution'),
                specs=[[{"secondary_y": True}, {"secondary_y": False}],
                      [{"secondary_y": False}, {"secondary_y": False}],
                      [{"secondary_y": False}, {"secondary_y": False}]]
            )

            # Ensure data alignment
            min_length = min(len(portfolio), len(regime_sequence))
            portfolio_aligned = portfolio.iloc[:min_length]
            regimes_aligned = regime_sequence[:min_length]

            # Plot 1: Price with regime overlay
            fig.add_trace(
                go.Scatter(x=portfolio_aligned.index, y=portfolio_aligned['price'],
                          name='Price', line=dict(color='black')),
                row=1, col=1
            )

            # Add regime colors as background (simplified)
            for i, regime in enumerate(regimes_aligned):
                if i > 0:
                    fig.add_shape(
                        type="rect",
                        x0=portfolio_aligned.index[i-1], x1=portfolio_aligned.index[i],
                        y0=0, y1=1, yref="y domain",
                        fillcolor=self.regime_colors.get(regime, 'gray'),
                        opacity=0.2, layer="below", line_width=0,
                        row=1, col=1
                    )

            # Plot 2: Cumulative returns
            fig.add_trace(
                go.Scatter(x=portfolio_aligned.index, y=portfolio_aligned['cumulative_net'],
                          name='Strategy', line=dict(color='blue')),
                row=1, col=2
            )

            if 'benchmark_cumulative' in portfolio_aligned.columns:
                fig.add_trace(
                    go.Scatter(x=portfolio_aligned.index, y=portfolio_aligned['benchmark_cumulative'],
                              name='Benchmark', line=dict(color='red')),
                    row=1, col=2
                )

            # Plot 3: Positions
            fig.add_trace(
                go.Scatter(x=portfolio_aligned.index, y=portfolio_aligned['position'],
                          name='Position', line=dict(color='purple')),
                row=2, col=1
            )

            # Plot 4: Drawdown
            cumulative = portfolio_aligned['cumulative_net']
            rolling_max = cumulative.expanding().max()
            drawdown = (cumulative - rolling_max) / rolling_max

            fig.add_trace(
                go.Scatter(x=portfolio_aligned.index, y=drawdown,
                          name='Drawdown', fill='tonexty',
                          line=dict(color='red')),
                row=2, col=2
            )

            # Plot 5: Rolling Sharpe
            if len(portfolio_aligned) > 252:
                rolling_returns = portfolio_aligned['net_returns'].rolling(window=252)
                rolling_sharpe = rolling_returns.mean() / rolling_returns.std() * np.sqrt(252)

                fig.add_trace(
                    go.Scatter(x=portfolio_aligned.index, y=rolling_sharpe,
                              name='Rolling Sharpe', line=dict(color='green')),
                    row=3, col=1
                )

            # Plot 6: Returns distribution
            fig.add_trace(
                go.Histogram(x=portfolio_aligned['net_returns'],
                           name='Returns Distribution', nbinsx=50),
                row=3, col=2
            )

            # Update layout
            fig.update_layout(
                title="Interactive Regime Trading Dashboard",
                height=1000,
                showlegend=True
            )

            logger.info("Interactive dashboard created successfully")
            return fig

        except Exception as e:
            logger.error(f"Error creating interactive dashboard: {str(e)}")
            raise

    def _add_regime_background(self, ax, dates, regimes):
        """Add regime background colors to plot"""
        for i in range(len(regimes) - 1):
            regime = regimes[i]
            color = self.regime_colors.get(regime, 'gray')
            ax.axvspan(dates[i], dates[i + 1], alpha=0.2, color=color)

    def save_plots(self, figures: List[plt.Figure], prefix: str = "regime_trading"):
        """
        Save all plots to files

        Parameters:
        -----------
        figures : List of matplotlib figures
        prefix : str, filename prefix
        """
        try:
            for i, fig in enumerate(figures):
                filename = f"{prefix}_plot_{i+1}.png"
                fig.savefig(filename, dpi=self.dpi, bbox_inches='tight')
                logger.info(f"Saved plot to {filename}")

        except Exception as e:
            logger.error(f"Error saving plots: {str(e)}")


def main():
    """
    Example usage of RegimeVisualizer
    """
    visualizer = RegimeVisualizer()
    print("RegimeVisualizer initialized successfully")
    print("Use visualizer methods to create various plots and dashboards")

    return visualizer


if __name__ == "__main__":
    main()