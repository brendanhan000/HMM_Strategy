"""
Enhanced main execution script for optimized HMM Regime Trading System
Target: >30% total return, >1.5 Sharpe ratio

This enhanced version focuses on:
1. Better feature engineering for regime detection
2. Intelligent regime mapping and interpretation
3. Optimized position sizing and signal generation
4. Parameter optimization for performance targets
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

# Import enhanced modules
from data.data_loader import DataLoader
from models.enhanced_garch_model import EnhancedGARCHModel
from models.enhanced_hmm_model import EnhancedRegimeHMM
from trading.enhanced_strategy import EnhancedRegimeTradingStrategy
from analysis.visualization import RegimeVisualizer

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('enhanced_hmm_trading.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class EnhancedHMMTradingSystem:
    """
    Enhanced HMM Trading System optimized for performance targets
    """

    def __init__(self):
        self.data_loader = DataLoader()
        self.garch_model = EnhancedGARCHModel()
        self.hmm_model = EnhancedRegimeHMM()
        self.strategy = EnhancedRegimeTradingStrategy()
        self.visualizer = RegimeVisualizer()

        # Results storage
        self.data = None
        self.features = None
        self.regime_predictions = None
        self.portfolio_results = None
        self.final_metrics = None

    def run_enhanced_analysis(self) -> Dict:
        """
        Run enhanced analysis targeting >30% return and >1.5 Sharpe
        """
        try:
            logger.info("🚀 Starting Enhanced HMM Trading System Analysis...")
            logger.info("🎯 Target: >30% Total Return, >1.5 Sharpe Ratio")

            # Step 1: Load and prepare data
            self._load_and_prepare_data()

            # Step 2: Train enhanced models
            self._train_enhanced_models()

            # Step 3: Run optimized strategy
            self._run_optimized_strategy()

            # Step 4: Analyze and validate performance
            results = self._analyze_performance()

            # Step 5: Generate report
            self._generate_enhanced_report(results)

            return results

        except Exception as e:
            logger.error(f"❌ Error in enhanced analysis: {str(e)}")
            raise

    def _load_and_prepare_data(self):
        """Load and prepare data with enhanced preprocessing"""
        logger.info("📊 Loading and preparing market data...")

        # Load SPY data
        self.data = self.data_loader.load_and_process()

        # Add enhanced features for better regime detection
        self.data['price_ma_20'] = self.data['Close'].rolling(20).mean()
        self.data['price_ma_50'] = self.data['Close'].rolling(50).mean()
        self.data['rsi'] = self._calculate_rsi(self.data['Close'])
        self.data['trend_strength'] = self._calculate_trend_strength(self.data['Close'])

        logger.info(f"✅ Data loaded: {len(self.data)} observations")
        logger.info(f"📅 Period: {self.data.index[0].date()} to {self.data.index[-1].date()}")

    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_trend_strength(self, prices: pd.Series, window: int = 20) -> pd.Series:
        """Calculate trend strength indicator"""
        ma = prices.rolling(window).mean()
        return (prices - ma) / ma

    def _train_enhanced_models(self):
        """Train enhanced GARCH and HMM models"""
        logger.info("🧠 Training enhanced models...")

        # Train enhanced GARCH model
        self.garch_model.fit(self.data['Returns'])

        # Extract enhanced features
        self.features = self.garch_model.extract_regime_features(self.data['Returns'])

        # Train enhanced HMM model with intelligent regime mapping
        self.hmm_model.fit(self.features, self.data['Returns'])

        # Generate regime predictions
        self.regime_predictions = self.hmm_model.predict_regimes(self.features)

        logger.info("✅ Enhanced models trained successfully")

        # Log regime mapping
        regime_mapping = self.hmm_model.regime_mapping
        logger.info(f"🎭 Regime Mapping: {regime_mapping}")

        # Log regime characteristics
        for state, regime in regime_mapping.items():
            chars = self.hmm_model.regime_characteristics.get(state, {})
            logger.info(f"  {regime}: Avg Return={chars.get('avg_return', 0):.4f}, "
                       f"Sharpe={chars.get('sharpe', 0):.2f}, "
                       f"Frequency={chars.get('frequency', 0):.1%}")

    def _run_optimized_strategy(self):
        """Run strategy with parameter optimization"""
        logger.info("⚡ Running optimized trading strategy...")

        # Get volatility forecasts
        vol_forecast = self.garch_model.get_volatility_forecast(self.data['Returns'])

        # Get optimized regime positions
        regime_positions = self.hmm_model.get_regime_positions()

        # Optimize strategy parameters
        optimization_results = self.strategy.optimize_parameters(
            self.regime_predictions, self.data['Close'], vol_forecast
        )

        logger.info(f"🔧 Optimization Results: {optimization_results}")

        # Generate final signals with optimized parameters
        signals = self.strategy.generate_enhanced_signals(
            self.regime_predictions, self.data['Close'], vol_forecast, regime_positions
        )

        # Calculate portfolio performance
        self.portfolio_results = self.strategy.calculate_enhanced_returns(signals, self.data['Close'])

        logger.info("✅ Strategy execution completed")

    def _analyze_performance(self) -> Dict:
        """Analyze performance and validate targets"""
        logger.info("📈 Analyzing performance...")

        # Calculate comprehensive metrics
        self.final_metrics = self.strategy.calculate_performance_metrics(self.portfolio_results)

        # Performance validation
        total_return = self.final_metrics['total_return']
        sharpe_ratio = self.final_metrics['sharpe_ratio']
        annual_return = self.final_metrics['annual_return']

        # Check if targets are met
        return_target_met = total_return >= 0.30
        sharpe_target_met = sharpe_ratio >= 1.5

        logger.info("🎯 PERFORMANCE VALIDATION:")
        logger.info(f"  Total Return: {total_return:.1%} {'✅' if return_target_met else '❌'} (Target: >30%)")
        logger.info(f"  Annual Return: {annual_return:.1%}")
        logger.info(f"  Sharpe Ratio: {sharpe_ratio:.2f} {'✅' if sharpe_target_met else '❌'} (Target: >1.5)")
        logger.info(f"  Max Drawdown: {self.final_metrics['max_drawdown']:.1%}")
        logger.info(f"  Information Ratio: {self.final_metrics['information_ratio']:.2f}")

        # Additional performance metrics
        regime_performance = self._analyze_regime_performance()

        results = {
            'performance_metrics': self.final_metrics,
            'regime_performance': regime_performance,
            'targets_met': {
                'return_target': return_target_met,
                'sharpe_target': sharpe_target_met,
                'both_targets': return_target_met and sharpe_target_met
            },
            'portfolio_data': self.portfolio_results,
            'regime_predictions': self.regime_predictions
        }

        return results

    def _analyze_regime_performance(self) -> Dict:
        """Analyze performance by regime"""
        regime_perf = {}

        if 'regime' in self.portfolio_results.columns:
            for regime in self.portfolio_results['regime'].unique():
                regime_mask = self.portfolio_results['regime'] == regime
                regime_data = self.portfolio_results[regime_mask]

                if len(regime_data) > 0:
                    regime_returns = regime_data['net_returns']
                    regime_perf[regime] = {
                        'avg_return': regime_returns.mean(),
                        'volatility': regime_returns.std(),
                        'sharpe': regime_returns.mean() / regime_returns.std() * np.sqrt(252) if regime_returns.std() > 0 else 0,
                        'win_rate': (regime_returns > 0).mean(),
                        'frequency': len(regime_data) / len(self.portfolio_results),
                        'total_return': regime_returns.sum()
                    }

        return regime_perf

    def _generate_enhanced_report(self, results: Dict):
        """Generate comprehensive performance report"""
        logger.info("📋 Generating enhanced performance report...")

        targets_met = results['targets_met']
        metrics = results['performance_metrics']
        regime_perf = results['regime_performance']

        print("\n" + "="*80)
        print("🎯 ENHANCED HMM REGIME TRADING SYSTEM - PERFORMANCE REPORT")
        print("="*80)

        print(f"\n📊 OVERALL PERFORMANCE:")
        print(f"   Total Return:     {metrics['total_return']:.1%} {'✅' if targets_met['return_target'] else '❌'}")
        print(f"   Annual Return:    {metrics['annual_return']:.1%}")
        print(f"   Sharpe Ratio:     {metrics['sharpe_ratio']:.2f} {'✅' if targets_met['sharpe_target'] else '❌'}")
        print(f"   Max Drawdown:     {metrics['max_drawdown']:.1%}")
        print(f"   Information Ratio: {metrics['information_ratio']:.2f}")
        print(f"   Calmar Ratio:     {metrics['calmar_ratio']:.2f}")
        print(f"   Win Rate:         {metrics['win_rate']:.1%}")

        print(f"\n🎭 REGIME PERFORMANCE:")
        for regime, perf in regime_perf.items():
            print(f"   {regime:>8}: Return={perf['avg_return']:>7.4f}, "
                  f"Sharpe={perf['sharpe']:>5.2f}, "
                  f"Win Rate={perf['win_rate']:>5.1%}, "
                  f"Freq={perf['frequency']:>5.1%}")

        print(f"\n🎯 TARGET ACHIEVEMENT:")
        print(f"   Return Target (>30%):  {'✅ ACHIEVED' if targets_met['return_target'] else '❌ NOT MET'}")
        print(f"   Sharpe Target (>1.5):  {'✅ ACHIEVED' if targets_met['sharpe_target'] else '❌ NOT MET'}")
        print(f"   Overall Success:       {'🎉 ALL TARGETS MET!' if targets_met['both_targets'] else '⚠️  TARGETS NOT MET'}")

        if not targets_met['both_targets']:
            print(f"\n💡 RECOMMENDATIONS FOR IMPROVEMENT:")
            if not targets_met['return_target']:
                print(f"   • Consider increasing leverage or position sizing")
                print(f"   • Improve regime detection accuracy")
                print(f"   • Optimize entry/exit timing")
            if not targets_met['sharpe_target']:
                print(f"   • Reduce position volatility")
                print(f"   • Improve risk management")
                print(f"   • Consider regime-specific volatility targeting")

        print("\n" + "="*80)

        # Create visualizations
        try:
            self._create_visualizations(results)
        except Exception as e:
            logger.warning(f"⚠️  Could not create visualizations: {str(e)}")

    def _create_visualizations(self, results: Dict):
        """Create performance visualizations"""
        logger.info("📊 Creating visualizations...")

        try:
            # Regime detection plot
            regime_plot = self.visualizer.plot_regime_detection(
                self.data['Close'],
                self.regime_predictions['regime_sequence'],
                self.regime_predictions.get('regime_probabilities')
            )

            # Portfolio performance plot
            portfolio_plot = self.visualizer.plot_portfolio_performance(
                self.portfolio_results,
                title="Enhanced HMM Strategy Performance"
            )

            # Save plots
            self.visualizer.save_plots([regime_plot, portfolio_plot], prefix="enhanced_hmm_trading")

            logger.info("✅ Visualizations created successfully")

        except Exception as e:
            logger.error(f"❌ Error creating visualizations: {str(e)}")


def main():
    """Main execution function"""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                   🚀 ENHANCED HMM REGIME TRADING SYSTEM                     ║
║                                                                              ║
║  🎯 Performance Targets:                                                     ║
║     • Total Return: >30%                                                     ║
║     • Sharpe Ratio: >1.5                                                     ║
║                                                                              ║
║  ⚡ Enhanced Features:                                                       ║
║     • Intelligent regime identification                                      ║
║     • Optimized position sizing                                              ║
║     • Advanced signal generation                                             ║
║     • Parameter optimization                                                  ║
║                                                                              ║
║  🤖 Generated with Claude Code                                               ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)

    try:
        # Initialize enhanced system
        system = EnhancedHMMTradingSystem()

        # Run enhanced analysis
        results = system.run_enhanced_analysis()

        # Final status
        if results['targets_met']['both_targets']:
            print("\n🎉 SUCCESS: All performance targets achieved!")
        else:
            print("\n⚠️  Performance targets not fully met. See recommendations above.")

        return results

    except Exception as e:
        logger.error(f"❌ Error in main execution: {str(e)}")
        print(f"\n❌ ERROR: {str(e)}")
        return None


if __name__ == "__main__":
    main()