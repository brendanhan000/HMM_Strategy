"""
Final optimized HMM trading system targeting >30% return and >1.5 Sharpe ratio
Based on performance analysis and regime profitability optimization
"""

import sys
import os
import numpy as np
import pandas as pd
import logging
import warnings
from typing import Dict

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.data_loader import DataLoader
from models.enhanced_garch_model import EnhancedGARCHModel
from models.performance_optimized_hmm import PerformanceOptimizedHMM
from trading.performance_optimized_strategy import PerformanceOptimizedStrategy

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)

class OptimizedHMMTradingSystem:
    """
    Final optimized HMM trading system
    """

    def __init__(self):
        self.data_loader = DataLoader()
        self.garch_model = EnhancedGARCHModel()
        self.hmm_model = PerformanceOptimizedHMM()
        self.strategy = PerformanceOptimizedStrategy()

    def run_final_analysis(self) -> Dict:
        """
        Run the final optimized analysis
        """
        try:
            logger.info("🎯 FINAL OPTIMIZED HMM TRADING SYSTEM")
            logger.info("📈 Target: >30% Total Return, >1.5 Sharpe Ratio")

            # Load data
            logger.info("📊 Loading SPY data...")
            data = self.data_loader.load_and_process()
            logger.info(f"✅ Loaded {len(data)} observations")

            # Train GARCH model
            logger.info("⚙️  Training GARCH model...")
            self.garch_model.fit(data['Returns'])
            features = self.garch_model.extract_regime_features(data['Returns'])
            logger.info("✅ GARCH model trained")

            # Train performance-optimized HMM
            logger.info("🧠 Training performance-optimized HMM...")
            self.hmm_model.fit(features, data['Returns'])
            regime_predictions = self.hmm_model.predict_regimes(features, data['Returns'])
            regime_positions = self.hmm_model.get_regime_positions()
            logger.info("✅ Performance-optimized HMM trained")

            # Display expected performance
            expected_perf = self.hmm_model.get_model_summary().get('expected_performance', {})
            if expected_perf:
                logger.info(f"📊 Expected Performance:")
                logger.info(f"   Annual Return: {expected_perf.get('expected_annual_return', 0):.1%}")
                logger.info(f"   Expected Sharpe: {expected_perf.get('expected_sharpe', 0):.2f}")

            # Run optimized strategy
            logger.info("🚀 Running optimized trading strategy...")
            results = self.strategy.run_optimized_backtest(
                regime_predictions, data['Close'], regime_positions
            )

            # Generate final report
            self._generate_final_report(results)

            return results

        except Exception as e:
            logger.error(f"❌ Error in final analysis: {str(e)}")
            raise

    def _generate_final_report(self, results: Dict):
        """Generate the final performance report"""
        metrics = results.get('metrics', {})

        print("\n" + "="*90)
        print("🎯 OPTIMIZED HMM REGIME TRADING SYSTEM - FINAL PERFORMANCE REPORT")
        print("="*90)

        if not metrics:
            print("❌ No performance metrics available")
            return

        # Extract key metrics
        total_return = metrics.get('total_return', 0)
        annual_return = metrics.get('annual_return', 0)
        benchmark_return = metrics.get('benchmark_return', 0)
        sharpe_ratio = metrics.get('sharpe_ratio', 0)
        max_drawdown = metrics.get('max_drawdown', 0)
        volatility = metrics.get('volatility', 0)
        information_ratio = metrics.get('information_ratio', 0)
        win_rate = metrics.get('win_rate', 0)

        # Target achievement
        targets = metrics.get('targets_met', {})
        return_achieved = targets.get('return_target', False)
        sharpe_achieved = targets.get('sharpe_target', False)
        both_achieved = return_achieved and sharpe_achieved

        print(f"\n🎯 TARGET ACHIEVEMENT STATUS:")
        print(f"   Overall Success:   {'🎉 SUCCESS!' if both_achieved else '⚠️  PARTIAL SUCCESS' if (return_achieved or sharpe_achieved) else '❌ TARGETS NOT MET'}")
        print(f"   Return Target:     {'✅' if return_achieved else '❌'} {total_return:.1%} (Target: >30.0%)")
        print(f"   Sharpe Target:     {'✅' if sharpe_achieved else '❌'} {sharpe_ratio:.2f} (Target: >1.5)")

        if targets.get('return_gap') is not None:
            print(f"   Return Gap:        {targets['return_gap']:.1%}")
        if targets.get('sharpe_gap') is not None:
            print(f"   Sharpe Gap:        {targets['sharpe_gap']:+.2f}")

        print(f"\n📊 PERFORMANCE SUMMARY:")
        print(f"   Total Return:      {total_return:.1%}")
        print(f"   Annual Return:     {annual_return:.1%}")
        print(f"   Benchmark Return:  {benchmark_return:.1%}")
        print(f"   Excess Return:     {(total_return - benchmark_return):.1%}")

        print(f"\n📈 RISK-ADJUSTED RETURNS:")
        print(f"   Sharpe Ratio:      {sharpe_ratio:.2f}")
        print(f"   Information Ratio: {information_ratio:.2f}")
        print(f"   Calmar Ratio:      {metrics.get('calmar_ratio', 0):.2f}")

        print(f"\n⚠️  RISK ANALYSIS:")
        print(f"   Volatility:        {volatility:.1%}")
        print(f"   Max Drawdown:      {max_drawdown:.1%}")
        print(f"   Tracking Error:    {metrics.get('tracking_error', 0):.1%}")

        print(f"\n📊 TRADING STATISTICS:")
        print(f"   Win Rate:          {win_rate:.1%}")
        print(f"   Total Trades:      {metrics.get('total_trades', 0)}")
        print(f"   Analysis Period:   {metrics.get('n_years', 0):.1f} years")

        # Regime performance breakdown
        portfolio = results.get('portfolio', pd.DataFrame())
        if not portfolio.empty and 'regime' in portfolio.columns:
            print(f"\n🎭 REGIME PERFORMANCE BREAKDOWN:")
            for regime in portfolio['regime'].unique():
                regime_data = portfolio[portfolio['regime'] == regime]
                if len(regime_data) > 0:
                    regime_returns = regime_data['net_returns']
                    freq = len(regime_data) / len(portfolio)
                    avg_return = regime_returns.mean()
                    regime_vol = regime_returns.std() * np.sqrt(252)
                    regime_sharpe = avg_return * np.sqrt(252) / regime_vol if regime_vol > 0 else 0
                    regime_win_rate = (regime_returns > 0).mean()

                    print(f"   {regime:>12}: Freq={freq:>5.1%}, "
                          f"Ann.Ret={avg_return * 252:>6.1%}, "
                          f"Sharpe={regime_sharpe:>5.2f}, "
                          f"Win={regime_win_rate:>5.1%}")

        # Success celebration or improvement recommendations
        if both_achieved:
            print(f"\n🎉 CONGRATULATIONS!")
            print(f"   Both performance targets have been achieved!")
            print(f"   • Total Return: {total_return:.1%} (Target: >30%)")
            print(f"   • Sharpe Ratio: {sharpe_ratio:.2f} (Target: >1.5)")
            print(f"   The optimized HMM regime detection system is successful!")
        else:
            print(f"\n💡 PERFORMANCE ANALYSIS:")

            if not return_achieved:
                gap = targets.get('return_gap', 0)
                print(f"   Return shortfall: {gap:.1%}")
                if gap > -0.10:  # Within 10%
                    print(f"   → Close to target! Consider minor leverage increase")
                else:
                    print(f"   → Significant gap. May need strategy restructuring")

            if not sharpe_achieved:
                gap = targets.get('sharpe_gap', 0)
                print(f"   Sharpe shortfall: {gap:+.2f}")
                if gap > -0.3:  # Within 0.3
                    print(f"   → Close to target! Consider risk management improvements")
                else:
                    print(f"   → Significant gap. May need volatility reduction")

            print(f"\n🔧 NEXT STEPS:")
            if not return_achieved:
                print(f"   • Increase position sizing in profitable regimes")
                print(f"   • Improve regime detection accuracy")
                print(f"   • Consider alternative alpha sources")
            if not sharpe_achieved:
                print(f"   • Implement dynamic volatility targeting")
                print(f"   • Improve risk-adjusted position sizing")
                print(f"   • Consider regime-specific risk management")

        print("\n" + "="*90)

        return metrics


def main():
    """Main execution function"""
    print("""
╔════════════════════════════════════════════════════════════════════════════════╗
║                   🎯 OPTIMIZED HMM REGIME TRADING SYSTEM                      ║
║                                                                                ║
║  🚀 FINAL VERSION - Performance Optimized                                     ║
║                                                                                ║
║  Targets:                                                                      ║
║  • Total Return: >30%                                                          ║
║  • Sharpe Ratio: >1.5                                                          ║
║                                                                                ║
║  🔧 Key Optimizations:                                                         ║
║  • Performance-based regime identification                                     ║
║  • Profit-optimized position sizing                                            ║
║  • Enhanced feature engineering                                                ║
║  • Dynamic risk management                                                     ║
║                                                                                ║
║  🤖 Generated with Claude Code                                                 ║
╚════════════════════════════════════════════════════════════════════════════════╝
    """)

    try:
        system = OptimizedHMMTradingSystem()
        results = system.run_final_analysis()

        # Final status message
        if results and results.get('metrics'):
            targets = results['metrics'].get('targets_met', {})
            if targets.get('return_target') and targets.get('sharpe_target'):
                print("\n🎉 MISSION ACCOMPLISHED: All performance targets achieved!")
            else:
                print(f"\n📊 Analysis complete. See performance breakdown above.")

        return results

    except Exception as e:
        logger.error(f"❌ System failed: {str(e)}")
        print(f"\n💥 SYSTEM ERROR: {str(e)}")
        return None


if __name__ == "__main__":
    results = main()