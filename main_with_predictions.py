"""
Enhanced HMM Trading System with Future Predictions and Advanced Visualizations
"""

import sys
import os
import numpy as np
import pandas as pd
import logging
import warnings
from datetime import datetime
from typing import Dict, List

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.data_loader import DataLoader
from models.enhanced_garch_model import EnhancedGARCHModel
from models.performance_optimized_hmm import PerformanceOptimizedHMM
from trading.performance_optimized_strategy import PerformanceOptimizedStrategy
from analysis.future_predictions import AdvancedPredictor
from analysis.advanced_visualizations import AdvancedVisualizer

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)

class PredictiveHMMTradingSystem:
    """
    Complete HMM Trading System with Predictions and Visualizations
    """

    def __init__(self):
        self.data_loader = DataLoader()
        self.garch_model = EnhancedGARCHModel()
        self.hmm_model = PerformanceOptimizedHMM()
        self.strategy = PerformanceOptimizedStrategy()
        self.predictor = AdvancedPredictor(forecast_horizon=30)
        self.visualizer = AdvancedVisualizer()

        # Results storage
        self.historical_data = None
        self.performance_results = None
        self.predictions = None
        self.visualizations = []

    def run_complete_analysis(self) -> Dict:
        """
        Run complete analysis with predictions and visualizations
        """
        try:
            logger.info("🚀 PREDICTIVE HMM TRADING SYSTEM - COMPLETE ANALYSIS")
            logger.info("🎯 Includes: Performance Analysis + Future Predictions + Advanced Visualizations")

            print("\n" + "="*80)
            print("🎯 PREDICTIVE HMM REGIME TRADING SYSTEM")
            print("="*80)

            # Phase 1: Historical Analysis
            logger.info("📊 Phase 1: Historical Performance Analysis")
            performance_results = self._run_historical_analysis()

            # Phase 2: Future Predictions
            logger.info("🔮 Phase 2: Generating Future Predictions")
            predictions = self._generate_future_predictions()

            # Phase 3: Advanced Visualizations
            logger.info("🎨 Phase 3: Creating Advanced Visualizations")
            visualizations = self._create_visualizations()

            # Phase 4: Comprehensive Report
            logger.info("📋 Phase 4: Generating Comprehensive Report")
            self._generate_complete_report(performance_results, predictions)

            results = {
                'performance_results': performance_results,
                'predictions': predictions,
                'visualizations': visualizations,
                'historical_data': self.historical_data
            }

            logger.info("🎉 Complete analysis finished successfully!")

            return results

        except Exception as e:
            logger.error(f"❌ Error in complete analysis: {str(e)}")
            raise

    def _run_historical_analysis(self) -> Dict:
        """Run historical performance analysis"""
        try:
            # Load data
            logger.info("📈 Loading historical SPY data...")
            self.historical_data = self.data_loader.load_and_process()
            logger.info(f"✅ Loaded {len(self.historical_data)} observations "
                       f"({self.historical_data.index[0].date()} to {self.historical_data.index[-1].date()})")

            # Train models
            logger.info("🧠 Training GARCH and HMM models...")
            self.garch_model.fit(self.historical_data['Returns'])
            features = self.garch_model.extract_regime_features(self.historical_data['Returns'])
            self.hmm_model.fit(features, self.historical_data['Returns'])

            # Generate regime predictions
            regime_predictions = self.hmm_model.predict_regimes(features, self.historical_data['Returns'])
            regime_positions = self.hmm_model.get_regime_positions()

            logger.info("✅ Models trained successfully")

            # Run strategy
            logger.info("⚡ Running optimized trading strategy...")
            performance_results = self.strategy.run_optimized_backtest(
                regime_predictions, self.historical_data['Close'], regime_positions
            )

            # Store for later use
            performance_results['regime_predictions'] = regime_predictions
            self.performance_results = performance_results

            return performance_results

        except Exception as e:
            logger.error(f"❌ Error in historical analysis: {str(e)}")
            raise

    def _generate_future_predictions(self) -> Dict:
        """Generate future predictions"""
        try:
            # Generate predictions
            predictions = self.predictor.generate_future_predictions(
                self.hmm_model,
                self.garch_model,
                self.historical_data.tail(252),  # Last year of data
                self.historical_data['Close'].iloc[-1]
            )

            self.predictions = predictions
            return predictions

        except Exception as e:
            logger.error(f"❌ Error generating predictions: {str(e)}")
            raise

    def _create_visualizations(self) -> List:
        """Create comprehensive visualizations"""
        try:
            visualizations = []

            # 1. Interactive Prediction Dashboard
            dashboard = self.visualizer.create_prediction_dashboard(
                self.predictions, self.historical_data, self.performance_results
            )
            visualizations.append(('Interactive Dashboard', dashboard))

            # 2. Regime Analysis Plot
            regime_plot = self.visualizer.create_regime_analysis_plot(
                self.historical_data, self.performance_results['regime_predictions']
            )
            visualizations.append(('Regime Analysis', regime_plot))

            # 3. Prediction Summary Plot
            prediction_plot = self.visualizer.create_prediction_summary_plot(self.predictions)
            visualizations.append(('Prediction Summary', prediction_plot))

            # Save all visualizations
            figures = [viz[1] for viz in visualizations]
            saved_files = self.visualizer.save_all_visualizations(figures, self.predictions)

            logger.info(f"✅ Created {len(visualizations)} visualizations")
            logger.info(f"📁 Saved files: {', '.join(saved_files)}")

            self.visualizations = visualizations
            return visualizations

        except Exception as e:
            logger.error(f"❌ Error creating visualizations: {str(e)}")
            raise

    def _generate_complete_report(self, performance_results: Dict, predictions: Dict):
        """Generate comprehensive report"""
        try:
            print("\n" + "="*90)
            print("📊 COMPREHENSIVE TRADING SYSTEM REPORT")
            print("="*90)

            # Historical Performance Section
            metrics = performance_results.get('metrics', {})
            if metrics:
                targets = metrics.get('targets_met', {})

                print(f"\n🏆 HISTORICAL PERFORMANCE (BACKTESTING):")
                print(f"   Total Return:      {metrics['total_return']:.1%} {'✅' if targets.get('return_target') else '❌'}")
                print(f"   Annual Return:     {metrics['annual_return']:.1%}")
                print(f"   Sharpe Ratio:      {metrics['sharpe_ratio']:.2f} {'✅' if targets.get('sharpe_target') else '❌'}")
                print(f"   Max Drawdown:      {metrics['max_drawdown']:.1%}")
                print(f"   Win Rate:          {metrics['win_rate']:.1%}")
                print(f"   Information Ratio: {metrics['information_ratio']:.2f}")

                # Regime performance
                portfolio = performance_results.get('portfolio', pd.DataFrame())
                if not portfolio.empty and 'regime' in portfolio.columns:
                    print(f"\n🎭 REGIME PERFORMANCE BREAKDOWN:")
                    for regime in portfolio['regime'].unique():
                        regime_data = portfolio[portfolio['regime'] == regime]
                        if len(regime_data) > 0:
                            regime_returns = regime_data['net_returns']
                            freq = len(regime_data) / len(portfolio)
                            avg_return = regime_returns.mean() * 252
                            regime_sharpe = avg_return / (regime_returns.std() * np.sqrt(252)) if regime_returns.std() > 0 else 0
                            win_rate = (regime_returns > 0).mean()

                            print(f"   {regime:>12}: Frequency={freq:>5.1%}, "
                                  f"Ann.Return={avg_return:>6.1%}, "
                                  f"Sharpe={regime_sharpe:>5.2f}, "
                                  f"Win Rate={win_rate:>5.1%}")

            # Future Predictions Section
            if predictions:
                current_state = predictions['current_state']
                trading_signals = predictions['trading_signals']

                print(f"\n🔮 FUTURE MARKET PREDICTIONS:")
                print(f"   Current Regime:    {current_state['current_regime']}")
                print(f"   Confidence:        {current_state['confidence']:.1%}")
                print(f"   Current Price:     ${current_state['current_price']:.2f}")
                print(f"   Volatility:        {current_state['current_volatility']:.1%}")

                # Next 5 days detailed predictions
                print(f"\n📅 NEXT 5 TRADING DAYS DETAILED OUTLOOK:")
                next_5_days = trading_signals.head(5)

                for i, (_, signal) in enumerate(next_5_days.iterrows(), 1):
                    print(f"\n   Day {i} ({signal['date'].strftime('%Y-%m-%d')}):")
                    print(f"      Action:         {signal['action']} ({signal['signal_strength']})")
                    print(f"      Regime:         {signal['regime']} (Confidence: {signal['regime_confidence']:.1%})")
                    print(f"      Target Price:   ${signal['target_price']:.2f}")
                    print(f"      Expected Return: {signal['expected_return']:.1%}")
                    print(f"      Risk Level:     {signal['risk_level']}")
                    print(f"      Position Size:  {signal['position']:.1f}x")
                    print(f"      Stop Loss:      ${signal['stop_loss']:.2f}")
                    print(f"      Take Profit:    ${signal['take_profit']:.2f}")

                # Overall outlook
                strong_signals = trading_signals[trading_signals['signal_strength'].isin(['Strong', 'Very Strong'])]
                buy_signals = len(trading_signals[trading_signals['action'] == 'BUY'])
                sell_signals = len(trading_signals[trading_signals['action'] == 'SELL'])

                print(f"\n📊 30-DAY STRATEGIC OUTLOOK:")
                print(f"   Strong Signals:    {len(strong_signals)}/{len(trading_signals)}")
                print(f"   Buy Signals:       {buy_signals}")
                print(f"   Sell Signals:      {sell_signals}")
                print(f"   Hold Periods:      {len(trading_signals) - buy_signals - sell_signals}")
                print(f"   Overall Bias:      {'BULLISH' if buy_signals > sell_signals else 'BEARISH' if sell_signals > buy_signals else 'NEUTRAL'}")

                # Risk warnings
                high_risk_days = len(trading_signals[trading_signals['risk_level'].isin(['High', 'Very High'])])
                if high_risk_days > 0:
                    print(f"\n⚠️  RISK WARNINGS:")
                    print(f"   • {high_risk_days} high-risk trading days identified")

                low_confidence_days = len(trading_signals[trading_signals['regime_confidence'] < 0.6])
                if low_confidence_days > 0:
                    print(f"   • {low_confidence_days} low-confidence prediction days")

            # Visualization Files
            if self.visualizations:
                print(f"\n📊 VISUALIZATION FILES CREATED:")
                for name, _ in self.visualizations:
                    print(f"   • {name}")

                print(f"\n💡 HOW TO USE:")
                print(f"   • Open HTML files in web browser for interactive charts")
                print(f"   • PNG files can be used for reports and presentations")
                print(f"   • Text summary contains detailed prediction analysis")

            # Success Summary
            success_indicators = []
            if metrics and metrics.get('targets_met', {}).get('return_target'):
                success_indicators.append("Return Target Achieved")
            if metrics and metrics.get('targets_met', {}).get('sharpe_target'):
                success_indicators.append("Sharpe Target Achieved")
            if predictions:
                success_indicators.append("Future Predictions Generated")
            if self.visualizations:
                success_indicators.append("Advanced Visualizations Created")

            print(f"\n🎯 SYSTEM SUCCESS SUMMARY:")
            for indicator in success_indicators:
                print(f"   ✅ {indicator}")

            if len(success_indicators) >= 3:
                print(f"\n🎉 SYSTEM PERFORMANCE: EXCELLENT")
            elif len(success_indicators) >= 2:
                print(f"\n✅ SYSTEM PERFORMANCE: GOOD")
            else:
                print(f"\n⚠️  SYSTEM PERFORMANCE: NEEDS IMPROVEMENT")

            print(f"\n🕒 Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("="*90)

        except Exception as e:
            logger.error(f"❌ Error generating report: {str(e)}")


def main():
    """Main execution function"""
    print("""
╔════════════════════════════════════════════════════════════════════════════════╗
║                   🔮 PREDICTIVE HMM TRADING SYSTEM                             ║
║                                                                                ║
║  🚀 Complete System Features:                                                  ║
║  • Historical Performance Analysis (>30% return, >1.5 Sharpe targets)          ║
║  • 30-Day Future Market Predictions                                            ║
║  • Advanced Interactive Visualizations                                         ║
║  • Actionable Trading Signals with Entry/Exit Points                           ║
║                                                                                ║
║  📊 Outputs:                                                                   ║
║  • Interactive HTML dashboards                                                 ║
║  • Static PNG charts for reports                                               ║
║  • Detailed prediction summaries                                               ║
║  • Trading signals with risk management                                        ║
║                                                                                ║
║  🤖 Generated with Claude Code                                                 ║
╚════════════════════════════════════════════════════════════════════════════════╝
    """)

    try:
        system = PredictiveHMMTradingSystem()
        results = system.run_complete_analysis()

        print(f"\n🎉 SUCCESS! Complete analysis finished.")
        print(f"📁 Check current directory for visualization files:")
        print(f"   • Interactive dashboards (.html)")
        print(f"   • Static charts (.png)")
        print(f"   • Prediction summaries (.txt)")

        return results

    except Exception as e:
        logger.error(f"💥 SYSTEM ERROR: {str(e)}")
        print(f"\n❌ Analysis failed: {str(e)}")
        return None


if __name__ == "__main__":
    results = main()