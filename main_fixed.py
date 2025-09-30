"""
Fixed main execution script targeting >30% return and >1.5 Sharpe ratio
With robust data handling and error-free calculations
"""

import sys
import os
import numpy as np
import pandas as pd
import logging
import warnings
from datetime import datetime
from typing import Dict

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import modules
from data.data_loader import DataLoader
from models.enhanced_garch_model import EnhancedGARCHModel
from models.enhanced_hmm_model import EnhancedRegimeHMM
from trading.fixed_strategy import FixedRegimeTradingStrategy

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class FixedHMMTradingSystem:
    """
    Fixed HMM Trading System targeting performance goals
    """

    def __init__(self):
        self.data_loader = DataLoader()
        self.garch_model = EnhancedGARCHModel()
        self.hmm_model = EnhancedRegimeHMM()
        self.strategy = FixedRegimeTradingStrategy()

    def run_analysis(self) -> Dict:
        """
        Run complete analysis with fixed components
        """
        try:
            logger.info("🚀 Starting Fixed HMM Trading System...")
            logger.info("🎯 Target: >30% Total Return, >1.5 Sharpe Ratio")

            # Step 1: Load data
            logger.info("📊 Loading SPY data...")
            data = self.data_loader.load_and_process()
            logger.info(f"✅ Loaded {len(data)} observations from {data.index[0].date()} to {data.index[-1].date()}")

            # Step 2: Train models
            logger.info("🧠 Training GARCH model...")
            self.garch_model.fit(data['Returns'])

            logger.info("🔍 Extracting regime features...")
            features = self.garch_model.extract_regime_features(data['Returns'])

            logger.info("🎭 Training HMM model...")
            self.hmm_model.fit(features, data['Returns'])

            # Generate predictions
            regime_predictions = self.hmm_model.predict_regimes(features)
            regime_positions = self.hmm_model.get_regime_positions()

            logger.info("✅ Models trained successfully")
            logger.info(f"Regime mapping: {self.hmm_model.regime_mapping}")

            # Step 3: Run strategy
            logger.info("⚡ Running trading strategy...")
            results = self.strategy.run_backtest(
                regime_predictions,
                data['Close'],
                regime_positions
            )

            # Step 4: Generate report
            self._generate_report(results)

            return results

        except Exception as e:
            logger.error(f"❌ Error in analysis: {str(e)}")
            raise

    def _generate_report(self, results: Dict):
        """Generate comprehensive report"""
        metrics = results['metrics']
        targets = results['targets_met']

        print("\n" + "="*80)
        print("🎯 FIXED HMM REGIME TRADING SYSTEM - PERFORMANCE REPORT")
        print("="*80)

        print(f"\n📊 OVERALL PERFORMANCE:")
        print(f"   Total Return:      {metrics['total_return']:.1%} {'✅' if targets['return_target'] else '❌'} (Target: >30%)")
        print(f"   Annual Return:     {metrics['annual_return']:.1%}")
        print(f"   Benchmark Return:  {metrics['benchmark_return']:.1%}")
        print(f"   Excess Return:     {metrics['excess_return']:.1%}")

        print(f"\n📈 RISK-ADJUSTED METRICS:")
        print(f"   Sharpe Ratio:      {metrics['sharpe_ratio']:.2f} {'✅' if targets['sharpe_target'] else '❌'} (Target: >1.5)")
        print(f"   Benchmark Sharpe:  {metrics['benchmark_sharpe']:.2f}")
        print(f"   Information Ratio: {metrics['information_ratio']:.2f}")
        print(f"   Calmar Ratio:      {metrics['calmar_ratio']:.2f}")

        print(f"\n⚠️  RISK METRICS:")
        print(f"   Volatility:        {metrics['volatility']:.1%}")
        print(f"   Max Drawdown:      {metrics['max_drawdown']:.1%}")
        print(f"   Tracking Error:    {metrics['tracking_error']:.1%}")

        print(f"\n📊 TRADING STATISTICS:")
        print(f"   Win Rate:          {metrics['win_rate']:.1%}")
        print(f"   Total Trades:      {metrics['total_trades']}")
        print(f"   Analysis Period:   {metrics['n_years']:.1f} years")
        print(f"   Observations:      {metrics['n_observations']}")

        print(f"\n🎯 TARGET ACHIEVEMENT:")
        if targets['both_targets']:
            print(f"   🎉 SUCCESS: ALL TARGETS ACHIEVED!")
            print(f"   ✅ Return Target (>30%): {metrics['total_return']:.1%}")
            print(f"   ✅ Sharpe Target (>1.5): {metrics['sharpe_ratio']:.2f}")
        else:
            print(f"   ⚠️  TARGETS NOT FULLY MET")
            print(f"   {'✅' if targets['return_target'] else '❌'} Return Target: {metrics['total_return']:.1%} {'(ACHIEVED)' if targets['return_target'] else '(NOT MET)'}")
            print(f"   {'✅' if targets['sharpe_target'] else '❌'} Sharpe Target: {metrics['sharpe_ratio']:.2f} {'(ACHIEVED)' if targets['sharpe_target'] else '(NOT MET)'}")

            if not targets['return_target'] or not targets['sharpe_target']:
                print(f"\n💡 IMPROVEMENT SUGGESTIONS:")
                if not targets['return_target']:
                    print(f"   • Increase position sizing or leverage")
                    print(f"   • Improve regime detection timing")
                    print(f"   • Consider more aggressive position allocation")
                if not targets['sharpe_target']:
                    print(f"   • Implement better risk management")
                    print(f"   • Consider volatility targeting")
                    print(f"   • Optimize position sizing based on volatility")

        # Regime-specific analysis
        portfolio = results['portfolio']
        if 'regime' in portfolio.columns:
            print(f"\n🎭 REGIME PERFORMANCE BREAKDOWN:")
            for regime in portfolio['regime'].unique():
                regime_data = portfolio[portfolio['regime'] == regime]
                if len(regime_data) > 0:
                    regime_returns = regime_data['net_returns']
                    freq = len(regime_data) / len(portfolio)
                    avg_return = regime_returns.mean()
                    regime_sharpe = avg_return / regime_returns.std() * np.sqrt(252) if regime_returns.std() > 0 else 0
                    win_rate = (regime_returns > 0).mean()

                    print(f"   {regime:>8}: Frequency={freq:>5.1%}, "
                          f"Avg Return={avg_return:>8.4f}, "
                          f"Sharpe={regime_sharpe:>5.2f}, "
                          f"Win Rate={win_rate:>5.1%}")

        print("\n" + "="*80)

        return results


def main():
    """Main execution function"""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                   🎯 FIXED HMM REGIME TRADING SYSTEM                        ║
║                                                                              ║
║  Performance Targets:                                                        ║
║  • Total Return: >30%                                                        ║
║  • Sharpe Ratio: >1.5                                                        ║
║                                                                              ║
║  🔧 Fixed Issues:                                                            ║
║  • Robust data alignment                                                     ║
║  • NaN-free calculations                                                     ║
║  • Enhanced position sizing                                                  ║
║  • Improved regime detection                                                 ║
║                                                                              ║
║  🤖 Generated with Claude Code                                               ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)

    try:
        system = FixedHMMTradingSystem()
        results = system.run_analysis()

        if results['targets_met']['both_targets']:
            print("\n🎉 MISSION ACCOMPLISHED: All performance targets achieved!")
        else:
            total_return = results['metrics']['total_return']
            sharpe_ratio = results['metrics']['sharpe_ratio']
            print(f"\n📊 CURRENT PERFORMANCE:")
            print(f"   Total Return: {total_return:.1%} {'✅' if total_return >= 0.30 else '❌'}")
            print(f"   Sharpe Ratio: {sharpe_ratio:.2f} {'✅' if sharpe_ratio >= 1.5 else '❌'}")

        return results

    except Exception as e:
        logger.error(f"❌ Fatal error: {str(e)}")
        return None


if __name__ == "__main__":
    results = main()