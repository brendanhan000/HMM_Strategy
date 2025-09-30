"""
Enhanced HMM Trading System with Future Predictions and Advanced Visualizations
"""

import sys
import os
import numpy as np
import pandas as pd
import logging
import warnings
import argparse
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

    def __init__(self, symbol: str = 'SPY'):
        self.symbol = symbol
        self.data_loader = DataLoader(symbol=symbol)
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
            logger.info(f"📈 Loading historical {self.symbol} data...")
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
                None,  # Let the predictor fetch current price
                self.symbol  # Pass the symbol for current price lookup
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

    def _generate_exit_signals(self, trading_signals: pd.DataFrame, current_state: Dict) -> pd.DataFrame:
        """
        Generate comprehensive exit signals and conditions
        """
        try:
            logger.info("🚪 Generating enhanced exit signals...")

            exit_signals = []

            for i, (_, signal) in enumerate(trading_signals.iterrows()):
                # Base exit conditions
                exit_info = {
                    'date': signal['date'],
                    'regime': signal['regime'],
                    'action': signal['action']
                }

                # 1. Regime Change Risk Assessment
                if i > 0:
                    prev_regime = trading_signals.iloc[i-1]['regime']
                    if signal['regime'] != prev_regime:
                        exit_info['regime_change_risk'] = 'HIGH - Regime transition detected'
                        exit_info['exit_trigger'] = 'REGIME_CHANGE'
                    elif signal['regime_confidence'] < 0.6:
                        exit_info['regime_change_risk'] = 'MEDIUM - Low regime confidence'
                    else:
                        exit_info['regime_change_risk'] = 'LOW - Stable regime'

                # 2. Position-specific exit triggers
                position = signal['position']
                if abs(position) > 2.0:  # High leverage
                    exit_info['exit_trigger'] = 'HIGH_LEVERAGE_REVIEW'
                    exit_info['hold_duration'] = 3  # Max 3 days for high leverage
                elif abs(position) > 1.0:  # Medium leverage
                    exit_info['hold_duration'] = 7  # Max 7 days
                else:
                    exit_info['hold_duration'] = 14  # Max 14 days for low leverage

                # 3. Volatility-based exits
                vol_forecast = signal['volatility_forecast']
                if vol_forecast > 0.3:  # High volatility
                    exit_info['volatility_exit'] = 'EXIT if vol > 35%'
                    exit_info['exit_trigger'] = 'VOLATILITY_SPIKE'
                elif vol_forecast > 0.25:
                    exit_info['volatility_exit'] = 'REDUCE position if vol > 30%'
                else:
                    exit_info['volatility_exit'] = 'Normal volatility conditions'

                # 4. Risk-based exits
                risk_level = signal['risk_level']
                if risk_level in ['Very High', 'High']:
                    exit_info['exit_trigger'] = 'HIGH_RISK'
                    exit_info['hold_duration'] = min(exit_info.get('hold_duration', 14), 2)

                # 5. Trailing stop calculation
                target_price = signal['target_price']
                if position > 0:  # Long position
                    # Trailing stop 3% below current high
                    exit_info['trailing_stop'] = target_price * 0.97
                    exit_info['trailing_stop_pct'] = '-3%'
                elif position < 0:  # Short position
                    # Trailing stop 3% above current low
                    exit_info['trailing_stop'] = target_price * 1.03
                    exit_info['trailing_stop_pct'] = '+3%'
                else:  # No position
                    exit_info['trailing_stop'] = target_price
                    exit_info['trailing_stop_pct'] = 'N/A'

                # 6. Signal strength-based exits
                signal_strength = signal['signal_strength']
                if signal_strength in ['Very Weak', 'Weak']:
                    exit_info['exit_trigger'] = 'WEAK_SIGNAL'
                    exit_info['hold_duration'] = 1  # Exit quickly on weak signals

                # 7. Expected return-based exits
                expected_return = signal['expected_return']
                if abs(expected_return) < 0.005:  # Less than 0.5% expected
                    exit_info['exit_trigger'] = 'LOW_EXPECTED_RETURN'
                elif expected_return < -0.02:  # Expecting significant loss
                    exit_info['exit_trigger'] = 'NEGATIVE_OUTLOOK'

                # 8. Time-based exit rules
                # Weekend approach (if approaching Friday)
                if signal['date'].weekday() == 4:  # Friday
                    exit_info['weekend_rule'] = 'CONSIDER_EXIT - Weekend approach'

                # Month-end effects
                if signal['date'].day >= 28:
                    exit_info['month_end_rule'] = 'MONTH_END - Potential volatility'

                exit_signals.append(exit_info)

            exit_df = pd.DataFrame(exit_signals)

            # Add comprehensive exit recommendations
            exit_df['exit_recommendation'] = exit_df.apply(self._calculate_exit_recommendation, axis=1)

            logger.info(f"✅ Generated exit signals for {len(exit_df)} trading days")

            return exit_df

        except Exception as e:
            logger.error(f"❌ Error generating exit signals: {str(e)}")
            return pd.DataFrame()

    def _calculate_exit_recommendation(self, row) -> str:
        """Calculate comprehensive exit recommendation"""
        exit_triggers = []

        # Collect all exit triggers
        if row.get('exit_trigger'):
            exit_triggers.append(row['exit_trigger'])

        regime_risk = str(row.get('regime_change_risk', ''))
        if regime_risk.startswith('HIGH'):
            exit_triggers.append('REGIME_RISK')

        volatility_exit = str(row.get('volatility_exit', ''))
        if volatility_exit.startswith('EXIT'):
            exit_triggers.append('VOL_SPIKE')

        # Generate recommendation
        if len(exit_triggers) >= 3:
            return "🚨 IMMEDIATE EXIT RECOMMENDED"
        elif len(exit_triggers) >= 2:
            return "⚠️ CONSIDER EXIT - Multiple warnings"
        elif len(exit_triggers) >= 1:
            return "👀 MONITOR CLOSELY - Exit trigger present"
        else:
            return "✅ HOLD - No exit signals"

    def _analyze_exit_recommendations(self, exit_signals: pd.DataFrame) -> Dict:
        """Analyze exit recommendations and provide summary statistics"""
        try:
            analysis = {}

            if exit_signals.empty:
                return {
                    'immediate_exits': 0,
                    'consider_exits': 0,
                    'monitor_days': 0,
                    'safe_hold_days': 0,
                    'exit_trigger_rate': 0.0,
                    'high_risk_periods': []
                }

            # Count recommendation types
            recommendations = exit_signals['exit_recommendation'].value_counts()
            total_days = len(exit_signals)

            analysis['immediate_exits'] = sum(1 for rec in exit_signals['exit_recommendation'] if '🚨 IMMEDIATE EXIT' in str(rec))
            analysis['consider_exits'] = sum(1 for rec in exit_signals['exit_recommendation'] if '⚠️ CONSIDER EXIT' in str(rec))
            analysis['monitor_days'] = sum(1 for rec in exit_signals['exit_recommendation'] if '👀 MONITOR CLOSELY' in str(rec))
            analysis['safe_hold_days'] = sum(1 for rec in exit_signals['exit_recommendation'] if '✅ HOLD' in str(rec))

            # Calculate exit trigger rate
            days_with_triggers = total_days - analysis['safe_hold_days']
            analysis['exit_trigger_rate'] = days_with_triggers / total_days if total_days > 0 else 0

            # Identify high-risk periods
            high_risk_periods = []
            for _, row in exit_signals.iterrows():
                if '🚨 IMMEDIATE EXIT' in str(row['exit_recommendation']):
                    reason_parts = []
                    if row.get('exit_trigger'):
                        reason_parts.append(row['exit_trigger'])

                    regime_risk = str(row.get('regime_change_risk', ''))
                    if regime_risk.startswith('HIGH'):
                        reason_parts.append('Regime Change')

                    volatility_exit = str(row.get('volatility_exit', ''))
                    if volatility_exit.startswith('EXIT'):
                        reason_parts.append('Volatility Spike')

                    high_risk_periods.append({
                        'date': row['date'],
                        'reason': ', '.join(reason_parts) if reason_parts else 'Multiple risk factors'
                    })

            analysis['high_risk_periods'] = high_risk_periods

            return analysis

        except Exception as e:
            logger.error(f"❌ Error analyzing exit recommendations: {str(e)}")
            return {
                'immediate_exits': 0,
                'consider_exits': 0,
                'monitor_days': 0,
                'safe_hold_days': 0,
                'exit_trigger_rate': 0.0,
                'high_risk_periods': []
            }

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

                # Enhanced exit analysis
                exit_signals = self._generate_exit_signals(trading_signals, current_state)

                # Next 5 days detailed predictions with exit conditions
                print(f"\n📅 NEXT 5 TRADING DAYS DETAILED OUTLOOK:")
                next_5_days = trading_signals.head(5)

                for i, (_, signal) in enumerate(next_5_days.iterrows(), 1):
                    exit_info = exit_signals.iloc[i-1] if i <= len(exit_signals) else {}

                    print(f"\n   Day {i} ({signal['date'].strftime('%Y-%m-%d')}):")
                    print(f"      Action:         {signal['action']} ({signal['signal_strength']})")
                    print(f"      Regime:         {signal['regime']} (Confidence: {signal['regime_confidence']:.1%})")
                    print(f"      Target Price:   ${signal['target_price']:.2f}")
                    print(f"      Expected Return: {signal['expected_return']:.1%}")
                    print(f"      Risk Level:     {signal['risk_level']}")
                    print(f"      Position Size:  {signal['position']:.1f}x")
                    print(f"      Stop Loss:      ${signal['stop_loss']:.2f}")
                    print(f"      Take Profit:    ${signal['take_profit']:.2f}")

                    # Add exit conditions
                    if exit_info:
                        print(f"      EXIT CONDITIONS:")
                        if exit_info.get('exit_recommendation'):
                            print(f"         • Overall Recommendation: {exit_info['exit_recommendation']}")
                        if exit_info.get('regime_change_risk'):
                            print(f"         • Regime Change Risk: {exit_info['regime_change_risk']}")
                        if exit_info.get('exit_trigger'):
                            print(f"         • Exit Trigger: {exit_info['exit_trigger']}")
                        if exit_info.get('hold_duration'):
                            print(f"         • Max Hold Duration: {exit_info['hold_duration']} days")
                        if exit_info.get('trailing_stop'):
                            print(f"         • Trailing Stop: ${exit_info['trailing_stop']:.2f}")
                        if exit_info.get('volatility_exit'):
                            print(f"         • Vol-based Exit: {exit_info['volatility_exit']}")
                        if exit_info.get('weekend_rule'):
                            print(f"         • Weekend Rule: {exit_info['weekend_rule']}")
                        if exit_info.get('month_end_rule'):
                            print(f"         • Month-End Rule: {exit_info['month_end_rule']}")

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

                # Exit Strategy Summary
                exit_analysis = self._analyze_exit_recommendations(exit_signals)
                print(f"\n🚪 EXIT STRATEGY OVERVIEW:")
                print(f"   Immediate Exits:     {exit_analysis['immediate_exits']} days")
                print(f"   Consider Exits:      {exit_analysis['consider_exits']} days")
                print(f"   Monitor Closely:     {exit_analysis['monitor_days']} days")
                print(f"   Safe Hold Days:      {exit_analysis['safe_hold_days']} days")
                print(f"   Exit Trigger Rate:   {exit_analysis['exit_trigger_rate']:.1%}")

                if exit_analysis['high_risk_periods']:
                    print(f"\n🚨 HIGH-RISK EXIT PERIODS:")
                    for period in exit_analysis['high_risk_periods'][:3]:  # Show top 3
                        print(f"   • {period['date'].strftime('%Y-%m-%d')}: {period['reason']}")

            # Visualization Files
            if self.visualizations:
                print(f"\n📊 VISUALIZATION FILES CREATED:")
                for name, _ in self.visualizations:
                    print(f"   • {name}")

                print(f"\n💡 HOW TO USE:")
                print(f"   • Open HTML files in web browser for interactive charts")
                print(f"   • PNG files can be used for reports and presentations")
                print(f"   • Text summary contains detailed prediction analysis")

                # Exit Strategy Guide
                print(f"\n🚪 COMPREHENSIVE EXIT STRATEGY GUIDE:")
                print(f"   📋 Exit Signal Priority (Immediate Action Required):")
                print(f"      1. 🚨 IMMEDIATE EXIT: Multiple risk factors present")
                print(f"      2. ⚠️  CONSIDER EXIT: 2+ warnings detected")
                print(f"      3. 👀 MONITOR CLOSELY: 1 exit trigger active")
                print(f"      4. ✅ HOLD: No exit signals detected")

                print(f"\n   🎯 Exit Trigger Types:")
                print(f"      • REGIME_CHANGE: Market regime transition detected")
                print(f"      • VOLATILITY_SPIKE: Volatility exceeds 30-35%")
                print(f"      • HIGH_LEVERAGE_REVIEW: Position size > 2x")
                print(f"      • HIGH_RISK: Risk level marked as High/Very High")
                print(f"      • WEAK_SIGNAL: Signal strength is Weak/Very Weak")
                print(f"      • LOW_EXPECTED_RETURN: Expected return < 0.5%")
                print(f"      • NEGATIVE_OUTLOOK: Expecting significant losses")

                print(f"\n   ⏰ Time-Based Rules:")
                print(f"      • High Leverage (>2x): Hold max 3 days")
                print(f"      • Medium Leverage (1-2x): Hold max 7 days")
                print(f"      • Low Leverage (<1x): Hold max 14 days")
                print(f"      • Weekend Approach: Consider exit on Fridays")
                print(f"      • Month-End: Increased volatility expected")

                print(f"\n   💰 Risk Management:")
                print(f"      • Trailing Stop: 3% from current high/low")
                print(f"      • Static Stop Loss: Individual per position")
                print(f"      • Take Profit: 2x expected return or 3% minimum")
                print(f"      • Position Sizing: Adjusted by volatility & confidence")

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


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='🔮 Predictive HMM Trading System - Advanced Market Regime Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main_with_predictions.py                    # Default SPY analysis
  python main_with_predictions.py --symbol PLTR      # Palantir analysis
  python main_with_predictions.py --symbol AAPL      # Apple analysis
  python main_with_predictions.py --symbol QQQ       # QQQ ETF analysis
        """
    )

    parser.add_argument(
        '--symbol', '-s',
        type=str,
        default='SPY',
        help='Stock symbol to analyze (default: SPY)'
    )

    parser.add_argument(
        '--forecast-horizon', '-f',
        type=int,
        default=30,
        help='Number of days to forecast (default: 30)'
    )

    return parser.parse_args()

def main():
    """Main execution function"""
    # Parse command line arguments
    args = parse_arguments()

    print(f"""
╔════════════════════════════════════════════════════════════════════════════════╗
║                   🔮 PREDICTIVE HMM TRADING SYSTEM                           ║
║                                                                                ║
║  📈 Analyzing Symbol: {args.symbol:<10}                                        ║
║  📅 Forecast Horizon: {args.forecast_horizon} days                                        ║
║                                                                                ║
║  🚀 Complete System Features:                                                 ║
║  • Historical Performance Analysis (>30% return, >1.5 Sharpe targets)        ║
║  • {args.forecast_horizon}-Day Future Market Predictions                                           ║
║  • Advanced Interactive Visualizations                                        ║
║  • Actionable Trading Signals with Entry/Exit Points                         ║
║                                                                                ║
║  📊 Outputs:                                                                  ║
║  • Interactive HTML dashboards                                                ║
║  • Static PNG charts for reports                                              ║
║  • Detailed prediction summaries                                              ║
║  • Trading signals with risk management                                       ║
║                                                                                ║
║  🤖 Generated with Claude Code                                                 ║
╚════════════════════════════════════════════════════════════════════════════════╝
    """)

    try:
        system = PredictiveHMMTradingSystem(symbol=args.symbol)
        system.predictor = AdvancedPredictor(forecast_horizon=args.forecast_horizon)
        results = system.run_complete_analysis()

        print(f"\n🎉 SUCCESS! Complete analysis finished for {args.symbol}.")
        print(f"📁 Check current directory for visualization files:")
        print(f"   • Interactive dashboards (.html)")
        print(f"   • Static charts (.png)")
        print(f"   • Prediction summaries (.txt)")

        return results

    except Exception as e:
        logger.error(f"💥 SYSTEM ERROR: {str(e)}")
        print(f"\n❌ Analysis failed for {args.symbol}: {str(e)}")
        return None


if __name__ == "__main__":
    results = main()