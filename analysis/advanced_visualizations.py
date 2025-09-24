"""
Advanced visualization suite for HMM regime trading system
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('default')
sns.set_palette("husl")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedVisualizer:
    """
    Advanced visualization suite for regime detection and predictions
    """

    def __init__(self, figsize: Tuple = (16, 10), dpi: int = 300):
        self.figsize = figsize
        self.dpi = dpi
        self.regime_colors = {
            'HighProfit': '#00FF00',    # Bright green
            'MediumProfit': '#FFD700',  # Gold
            'LowProfit': '#FF4500',     # Red-orange
            'Bull': '#00FF00',
            'Bear': '#FF0000',
            'Sideways': '#FFA500'
        }

    def create_prediction_dashboard(self, predictions: Dict, historical_data: pd.DataFrame,
                                  performance_results: Dict) -> go.Figure:
        """
        Create comprehensive interactive prediction dashboard
        """
        try:
            logger.info("🎨 Creating interactive prediction dashboard...")

            # Create subplots
            fig = make_subplots(
                rows=4, cols=2,
                subplot_titles=[
                    'Price Predictions with Confidence Bands',
                    'Regime Probability Forecasts',
                    'Trading Signals & Position Sizing',
                    'Volatility Forecasts',
                    'Historical Performance by Regime',
                    'Risk-Return Profile',
                    'Prediction Confidence Over Time',
                    'Expected Returns Distribution'
                ],
                specs=[
                    [{"secondary_y": True}, {"type": "scatter"}],
                    [{"type": "bar"}, {"type": "scatter"}],
                    [{"type": "scatter"}, {"type": "scatter"}],
                    [{"type": "histogram"}, {"type": "scatter"}]
                ],
                vertical_spacing=0.08,
                horizontal_spacing=0.1
            )

            # Extract prediction data
            forecast_dates = predictions['forecast_dates']
            current_state = predictions['current_state']
            regime_forecasts = predictions['regime_forecasts']
            price_forecasts = predictions['price_forecasts']
            trading_signals = predictions['trading_signals']
            confidence_intervals = predictions['confidence_intervals']

            # 1. Price Predictions with Confidence Bands
            self._add_price_predictions(fig, historical_data, price_forecasts, forecast_dates)

            # 2. Regime Probability Forecasts
            self._add_regime_forecasts(fig, regime_forecasts, forecast_dates)

            # 3. Trading Signals & Position Sizing
            self._add_trading_signals(fig, trading_signals)

            # 4. Volatility Forecasts
            self._add_volatility_forecasts(fig, predictions['volatility_forecasts'], forecast_dates)

            # 5. Historical Performance by Regime
            self._add_regime_performance(fig, performance_results)

            # 6. Risk-Return Profile
            self._add_risk_return_profile(fig, performance_results)

            # 7. Prediction Confidence
            self._add_prediction_confidence(fig, confidence_intervals, forecast_dates)

            # 8. Expected Returns Distribution
            self._add_returns_distribution(fig, trading_signals)

            # Update layout
            fig.update_layout(
                title=dict(
                    text="🎯 HMM Regime Trading System - Prediction Dashboard",
                    x=0.5,
                    font=dict(size=20, color='darkblue')
                ),
                height=1200,
                showlegend=True,
                template="plotly_white",
                font=dict(size=10),
                margin=dict(t=80, b=40, l=40, r=40)
            )

            logger.info("✅ Interactive dashboard created successfully")
            return fig

        except Exception as e:
            logger.error(f"❌ Error creating dashboard: {str(e)}")
            raise

    def _add_price_predictions(self, fig, historical_data: pd.DataFrame,
                             price_forecasts: Dict, forecast_dates: List):
        """Add price predictions with confidence bands"""
        # Historical prices (last 60 days)
        recent_data = historical_data.tail(60)
        fig.add_trace(
            go.Scatter(
                x=recent_data.index,
                y=recent_data['Close'],
                mode='lines',
                name='Historical Price',
                line=dict(color='black', width=2),
                showlegend=True
            ),
            row=1, col=1
        )

        # Predicted prices
        fig.add_trace(
            go.Scatter(
                x=forecast_dates,
                y=price_forecasts['mean'],
                mode='lines',
                name='Price Forecast',
                line=dict(color='blue', width=2, dash='dash'),
                showlegend=True
            ),
            row=1, col=1
        )

        # Confidence bands
        fig.add_trace(
            go.Scatter(
                x=forecast_dates + forecast_dates[::-1],
                y=list(price_forecasts['percentiles']['95']) + list(price_forecasts['percentiles']['5'][::-1]),
                fill='toself',
                fillcolor='rgba(0,100,80,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='95% Confidence',
                showlegend=True
            ),
            row=1, col=1
        )

        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)

    def _add_regime_forecasts(self, fig, regime_forecasts: pd.DataFrame, forecast_dates: List):
        """Add regime probability forecasts"""
        for regime in regime_forecasts.columns:
            color = self.regime_colors.get(regime, 'gray')
            fig.add_trace(
                go.Scatter(
                    x=forecast_dates,
                    y=regime_forecasts[regime],
                    mode='lines+markers',
                    name=f'{regime} Probability',
                    line=dict(color=color, width=2),
                    showlegend=True
                ),
                row=1, col=2
            )

        fig.update_xaxes(title_text="Date", row=1, col=2)
        fig.update_yaxes(title_text="Probability", row=1, col=2)

    def _add_trading_signals(self, fig, trading_signals: pd.DataFrame):
        """Add trading signals and position sizing"""
        # Position sizes
        colors = ['red' if pos < 0 else 'green' if pos > 0 else 'gray' for pos in trading_signals['position']]

        fig.add_trace(
            go.Bar(
                x=trading_signals['date'],
                y=trading_signals['position'],
                marker_color=colors,
                name='Position Size',
                showlegend=True
            ),
            row=2, col=1
        )

        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Position", row=2, col=1)

    def _add_volatility_forecasts(self, fig, volatility_forecasts: np.ndarray, forecast_dates: List):
        """Add volatility forecasts"""
        fig.add_trace(
            go.Scatter(
                x=forecast_dates,
                y=volatility_forecasts,
                mode='lines+markers',
                name='Volatility Forecast',
                line=dict(color='purple', width=2),
                showlegend=True
            ),
            row=2, col=2
        )

        fig.update_xaxes(title_text="Date", row=2, col=2)
        fig.update_yaxes(title_text="Volatility", row=2, col=2)

    def _add_regime_performance(self, fig, performance_results: Dict):
        """Add historical regime performance"""
        if 'portfolio' not in performance_results or performance_results['portfolio'].empty:
            return

        portfolio = performance_results['portfolio']
        if 'regime' not in portfolio.columns:
            return

        regime_stats = []
        for regime in portfolio['regime'].unique():
            regime_data = portfolio[portfolio['regime'] == regime]
            if len(regime_data) > 0:
                avg_return = regime_data['net_returns'].mean() * 252  # Annualized
                volatility = regime_data['net_returns'].std() * np.sqrt(252)
                sharpe = avg_return / volatility if volatility > 0 else 0

                regime_stats.append({
                    'regime': regime,
                    'return': avg_return,
                    'volatility': volatility,
                    'sharpe': sharpe
                })

        if regime_stats:
            regime_df = pd.DataFrame(regime_stats)

            fig.add_trace(
                go.Bar(
                    x=regime_df['regime'],
                    y=regime_df['return'],
                    marker_color=[self.regime_colors.get(r, 'gray') for r in regime_df['regime']],
                    name='Annualized Return',
                    showlegend=True
                ),
                row=3, col=1
            )

        fig.update_xaxes(title_text="Regime", row=3, col=1)
        fig.update_yaxes(title_text="Return", row=3, col=1)

    def _add_risk_return_profile(self, fig, performance_results: Dict):
        """Add risk-return scatter plot"""
        if 'portfolio' not in performance_results or performance_results['portfolio'].empty:
            return

        portfolio = performance_results['portfolio']
        if 'regime' not in portfolio.columns:
            return

        for regime in portfolio['regime'].unique():
            regime_data = portfolio[portfolio['regime'] == regime]
            if len(regime_data) > 0:
                returns = regime_data['net_returns'] * 252
                volatility = regime_data['net_returns'].std() * np.sqrt(252)

                fig.add_trace(
                    go.Scatter(
                        x=[volatility],
                        y=[returns.mean()],
                        mode='markers',
                        marker=dict(
                            color=self.regime_colors.get(regime, 'gray'),
                            size=15
                        ),
                        name=f'{regime} Risk-Return',
                        showlegend=True
                    ),
                    row=3, col=2
                )

        fig.update_xaxes(title_text="Risk (Volatility)", row=3, col=2)
        fig.update_yaxes(title_text="Return", row=3, col=2)

    def _add_prediction_confidence(self, fig, confidence_intervals: Dict, forecast_dates: List):
        """Add prediction confidence over time"""
        if 'regime_confidence' in confidence_intervals:
            fig.add_trace(
                go.Scatter(
                    x=forecast_dates,
                    y=confidence_intervals['regime_confidence'],
                    mode='lines+markers',
                    name='Prediction Confidence',
                    line=dict(color='orange', width=2),
                    showlegend=True
                ),
                row=4, col=1
            )

        fig.update_xaxes(title_text="Date", row=4, col=1)
        fig.update_yaxes(title_text="Confidence", row=4, col=1)

    def _add_returns_distribution(self, fig, trading_signals: pd.DataFrame):
        """Add expected returns distribution"""
        expected_returns = trading_signals['expected_return'] * 100  # Convert to percentage

        fig.add_trace(
            go.Histogram(
                x=expected_returns,
                nbinsx=20,
                name='Expected Returns Distribution',
                marker_color='lightblue',
                showlegend=True
            ),
            row=4, col=2
        )

        fig.update_xaxes(title_text="Expected Return (%)", row=4, col=2)
        fig.update_yaxes(title_text="Frequency", row=4, col=2)

    def create_regime_analysis_plot(self, historical_data: pd.DataFrame,
                                  regime_predictions: Dict) -> plt.Figure:
        """
        Create detailed regime analysis plot
        """
        try:
            logger.info("📊 Creating regime analysis plot...")

            fig, axes = plt.subplots(3, 1, figsize=(16, 12), dpi=self.dpi)
            fig.suptitle('HMM Regime Detection Analysis', fontsize=16, fontweight='bold')

            # Align data
            regime_sequence = regime_predictions['regime_sequence']
            min_length = min(len(historical_data), len(regime_sequence))

            dates = historical_data.index[:min_length]
            prices = historical_data['Close'].iloc[:min_length]
            regimes = regime_sequence[:min_length]

            # Plot 1: Price with regime background
            ax1 = axes[0]
            ax1.plot(dates, prices, color='black', linewidth=1.5, alpha=0.8, label='SPY Price')

            # Add regime background colors
            self._add_regime_background(ax1, dates, regimes, prices)

            ax1.set_title('SPY Price with Regime Detection', fontweight='bold')
            ax1.set_ylabel('Price ($)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Plot 2: Regime sequence
            ax2 = axes[1]
            regime_numeric = []
            regime_names = []
            unique_regimes = list(set(regimes))

            for regime in regimes:
                if regime in unique_regimes:
                    regime_numeric.append(unique_regimes.index(regime))

            colors = [self.regime_colors.get(regime, 'gray') for regime in regimes]
            ax2.scatter(dates, regime_numeric, c=colors, alpha=0.7, s=5)

            ax2.set_title('Regime Sequence Over Time', fontweight='bold')
            ax2.set_ylabel('Regime')
            ax2.set_yticks(range(len(unique_regimes)))
            ax2.set_yticklabels(unique_regimes)
            ax2.grid(True, alpha=0.3)

            # Plot 3: Regime probabilities (if available)
            ax3 = axes[2]
            if 'regime_probabilities' in regime_predictions:
                regime_probs = regime_predictions['regime_probabilities']
                if not regime_probs.empty and len(regime_probs) >= min_length:
                    for regime in regime_probs.columns:
                        color = self.regime_colors.get(regime, 'gray')
                        ax3.plot(dates, regime_probs[regime].iloc[:min_length],
                               label=f'{regime} Probability', color=color, alpha=0.8)

                    ax3.set_title('Regime Probabilities Over Time', fontweight='bold')
                    ax3.set_ylabel('Probability')
                    ax3.set_xlabel('Date')
                    ax3.legend()
                    ax3.grid(True, alpha=0.3)
                    ax3.set_ylim(0, 1)
            else:
                ax3.text(0.5, 0.5, 'Regime probabilities not available',
                        ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title('Regime Probabilities', fontweight='bold')

            plt.tight_layout()
            logger.info("✅ Regime analysis plot created")

            return fig

        except Exception as e:
            logger.error(f"❌ Error creating regime analysis plot: {str(e)}")
            raise

    def _add_regime_background(self, ax, dates, regimes, prices):
        """Add regime background colors to plot"""
        y_min, y_max = ax.get_ylim() if ax.get_ylim() != (0, 1) else (prices.min(), prices.max())

        current_regime = regimes[0] if len(regimes) > 0 else None
        start_idx = 0

        for i, regime in enumerate(regimes[1:], 1):
            if regime != current_regime:
                # Add background for previous regime
                color = self.regime_colors.get(current_regime, 'gray')
                ax.axvspan(dates[start_idx], dates[i], alpha=0.2, color=color,
                          label=f'{current_regime}' if current_regime not in ax.get_legend_handles_labels()[1] else "")

                current_regime = regime
                start_idx = i

        # Add final regime
        if current_regime:
            color = self.regime_colors.get(current_regime, 'gray')
            ax.axvspan(dates[start_idx], dates[-1], alpha=0.2, color=color,
                      label=f'{current_regime}' if current_regime not in ax.get_legend_handles_labels()[1] else "")

    def create_prediction_summary_plot(self, predictions: Dict) -> plt.Figure:
        """
        Create prediction summary visualization
        """
        try:
            logger.info("🔮 Creating prediction summary plot...")

            fig, axes = plt.subplots(2, 2, figsize=(16, 10), dpi=self.dpi)
            fig.suptitle('30-Day Market Predictions Summary', fontsize=16, fontweight='bold')

            trading_signals = predictions['trading_signals']
            forecast_dates = predictions['forecast_dates']
            price_forecasts = predictions['price_forecasts']

            # Plot 1: Expected Returns
            ax1 = axes[0, 0]
            returns = trading_signals['expected_return'] * 100
            colors = ['green' if r > 0 else 'red' if r < 0 else 'gray' for r in returns]

            ax1.bar(range(len(returns)), returns, color=colors, alpha=0.7)
            ax1.set_title('Expected Daily Returns (%)', fontweight='bold')
            ax1.set_ylabel('Expected Return (%)')
            ax1.set_xlabel('Days Ahead')
            ax1.grid(True, alpha=0.3)
            ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)

            # Plot 2: Position Sizes
            ax2 = axes[0, 1]
            positions = trading_signals['position']
            pos_colors = ['green' if p > 0 else 'red' if p < 0 else 'gray' for p in positions]

            ax2.bar(range(len(positions)), positions, color=pos_colors, alpha=0.7)
            ax2.set_title('Recommended Position Sizes', fontweight='bold')
            ax2.set_ylabel('Position')
            ax2.set_xlabel('Days Ahead')
            ax2.grid(True, alpha=0.3)
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)

            # Plot 3: Signal Strength
            ax3 = axes[1, 0]
            strength_mapping = {'Very Weak': 1, 'Weak': 2, 'Moderate': 3, 'Strong': 4, 'Very Strong': 5}
            strengths = [strength_mapping.get(s, 0) for s in trading_signals['signal_strength']]

            scatter = ax3.scatter(range(len(strengths)), strengths,
                                c=trading_signals['regime_confidence'], cmap='viridis',
                                s=100, alpha=0.7)
            ax3.set_title('Signal Strength & Confidence', fontweight='bold')
            ax3.set_ylabel('Signal Strength')
            ax3.set_xlabel('Days Ahead')
            ax3.set_yticks(range(1, 6))
            ax3.set_yticklabels(['Very Weak', 'Weak', 'Moderate', 'Strong', 'Very Strong'])
            ax3.grid(True, alpha=0.3)

            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax3)
            cbar.set_label('Regime Confidence')

            # Plot 4: Price Forecast with Uncertainty
            ax4 = axes[1, 1]
            days = range(len(price_forecasts['mean']))

            ax4.plot(days, price_forecasts['mean'], 'b-', linewidth=2, label='Expected Price')
            ax4.fill_between(days,
                           price_forecasts['percentiles']['25'],
                           price_forecasts['percentiles']['75'],
                           alpha=0.3, color='blue', label='50% Confidence')
            ax4.fill_between(days,
                           price_forecasts['percentiles']['5'],
                           price_forecasts['percentiles']['95'],
                           alpha=0.2, color='lightblue', label='90% Confidence')

            ax4.set_title('Price Forecasts with Uncertainty', fontweight='bold')
            ax4.set_ylabel('Price ($)')
            ax4.set_xlabel('Days Ahead')
            ax4.legend()
            ax4.grid(True, alpha=0.3)

            plt.tight_layout()
            logger.info("✅ Prediction summary plot created")

            return fig

        except Exception as e:
            logger.error(f"❌ Error creating prediction summary: {str(e)}")
            raise

    def save_all_visualizations(self, figures: List, predictions: Dict,
                               prefix: str = "hmm_predictions") -> List[str]:
        """
        Save all visualizations to files
        """
        try:
            saved_files = []
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            for i, fig in enumerate(figures):
                if isinstance(fig, go.Figure):
                    # Plotly figure
                    filename = f"{prefix}_{timestamp}_interactive_{i+1}.html"
                    fig.write_html(filename)
                    saved_files.append(filename)
                else:
                    # Matplotlib figure
                    filename = f"{prefix}_{timestamp}_static_{i+1}.png"
                    fig.savefig(filename, dpi=self.dpi, bbox_inches='tight')
                    saved_files.append(filename)

            # Save prediction summary as text
            if predictions:
                from .future_predictions import AdvancedPredictor
                predictor = AdvancedPredictor()
                summary_text = predictor.generate_prediction_summary(predictions)

                text_filename = f"{prefix}_{timestamp}_summary.txt"
                with open(text_filename, 'w') as f:
                    f.write(summary_text)
                saved_files.append(text_filename)

            logger.info(f"✅ Saved {len(saved_files)} visualization files")
            return saved_files

        except Exception as e:
            logger.error(f"❌ Error saving visualizations: {str(e)}")
            return []


def main():
    """Test the advanced visualizer"""
    visualizer = AdvancedVisualizer()
    print("Advanced Visualizer initialized successfully")
    print("Use visualizer methods to create comprehensive visualizations")

    return visualizer


if __name__ == "__main__":
    main()