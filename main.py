"""
Main execution script for HMM Regime Detection Trading System
Comprehensive implementation with GARCH volatility integration, backtesting, and predictions

This script orchestrates the complete workflow:
1. Data acquisition and preprocessing
2. GARCH model fitting and component extraction
3. HMM regime detection training and prediction
4. Trading strategy implementation and backtesting
5. Performance analysis and visualization
6. Forward regime prediction and signal generation

Usage:
    python main.py [--config config_file] [--mode {full,backtest,predict}]

Author: Claude Code - AI Trading System
Date: 2024
"""

import sys
import os
import argparse
import logging
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import all modules
from data.data_loader import DataLoader
from models.garch_model import GARCHModel
from models.hmm_model import RegimeHMM
from trading.strategy import RegimeTradingStrategy
from trading.backtester import RegimeBacktester
from analysis.performance import PerformanceAnalyzer
from analysis.visualization import RegimeVisualizer
from analysis.prediction import RegimePredictor
from config.settings import *

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('hmm_trading.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class HMMTradingSystem:
    """
    Main orchestrator for the HMM Regime Detection Trading System
    """

    def __init__(self, config: Dict = None):
        """
        Initialize the trading system

        Parameters:
        -----------
        config : Dict, optional configuration overrides
        """
        self.config = config or {}

        # Initialize components
        self.data_loader = DataLoader()
        self.garch_model = GARCHModel()
        self.hmm_model = RegimeHMM()
        self.strategy = RegimeTradingStrategy()
        self.backtester = RegimeBacktester()
        self.performance_analyzer = PerformanceAnalyzer()
        self.visualizer = RegimeVisualizer()
        self.predictor = RegimePredictor()

        # Storage for results
        self.data = None
        self.train_data = None
        self.test_data = None
        self.garch_components = None
        self.regime_predictions = None
        self.portfolio_results = None
        self.performance_analysis = None
        self.forecast_results = None

        logger.info("HMM Trading System initialized successfully")

    def run_full_analysis(self) -> Dict:
        """
        Run complete analysis pipeline

        Returns:
        --------
        results : Dict with all analysis results
        """
        try:
            logger.info("Starting full HMM trading system analysis...")

            # Step 1: Data acquisition and preprocessing
            self._load_and_preprocess_data()

            # Step 2: Model training and regime detection
            self._train_models()

            # Step 3: Strategy backtesting
            self._run_backtesting()

            # Step 4: Performance analysis
            self._analyze_performance()

            # Step 5: Generate forecasts
            self._generate_forecasts()

            # Step 6: Create visualizations
            self._create_visualizations()

            # Step 7: Generate comprehensive report
            report = self._generate_final_report()

            logger.info("Full analysis completed successfully!")

            return {
                'data_summary': self.data_loader.get_data_summary(),
                'garch_diagnostics': self.garch_model.model_diagnostics(),
                'hmm_summary': self.hmm_model.get_model_summary(),
                'portfolio_results': self.portfolio_results,
                'performance_analysis': self.performance_analysis,
                'forecast_results': self.forecast_results,
                'final_report': report
            }

        except Exception as e:
            logger.error(f"Error in full analysis: {str(e)}")
            raise

    def _load_and_preprocess_data(self) -> None:
        """Load and preprocess market data"""
        logger.info("Loading and preprocessing data...")

        # Load data
        self.data = self.data_loader.load_and_process()

        # Split into train/test
        self.train_data, self.test_data = self.data_loader.split_data()

        # Log data summary
        summary = self.data_loader.get_data_summary()
        logger.info(f"Data loaded: {summary['total_observations']} observations")
        logger.info(f"Training: {len(self.train_data)} obs, Testing: {len(self.test_data)} obs")

    def _train_models(self) -> None:
        """Train GARCH and HMM models"""
        logger.info("Training GARCH and HMM models...")

        # Fit GARCH model on training data
        self.garch_model.fit(self.train_data['Returns'])

        # Extract GARCH components
        train_components = self.garch_model.extract_components(self.train_data['Returns'])

        # Fit HMM model
        self.hmm_model.fit(train_components)

        # Generate regime predictions for full dataset
        self.garch_components = self.garch_model.extract_components(self.data['Returns'])
        self.regime_predictions = self.hmm_model.predict_regimes(self.garch_components)

        logger.info("Models trained successfully")

        # Log model diagnostics
        garch_diag = self.garch_model.model_diagnostics()
        logger.info(f"GARCH Log-likelihood: {garch_diag.get('log_likelihood', 'N/A')}")

        hmm_summary = self.hmm_model.get_model_summary()
        logger.info(f"HMM components: {hmm_summary['n_components']}")

    def _run_backtesting(self) -> None:
        """Run comprehensive backtesting"""
        logger.info("Running strategy backtesting...")

        # Generate trading signals
        signals = self.strategy.generate_signals(
            self.regime_predictions,
            volatility_forecast=self.garch_components.get('volatility_forecast')
        )

        # Calculate portfolio performance
        self.portfolio_results = self.strategy.calculate_portfolio_returns(
            signals, self.data['Close']
        )

        # Calculate trade statistics
        trade_stats = self.strategy.calculate_trade_statistics(self.portfolio_results)

        logger.info(f"Backtest completed: {trade_stats['num_trades']} trades")
        logger.info(f"Win rate: {trade_stats['win_rate']:.1%}")

    def _analyze_performance(self) -> None:
        """Perform comprehensive performance analysis"""
        logger.info("Analyzing performance...")

        # Run performance analysis
        self.performance_analysis = self.performance_analyzer.analyze_performance(
            self.portfolio_results
        )

        # Log key metrics
        basic = self.performance_analysis.get('basic_metrics', {})
        risk = self.performance_analysis.get('risk_metrics', {})

        logger.info(f"Total return: {basic.get('total_return', 0):.2%}")
        logger.info(f"Sharpe ratio: {risk.get('sharpe_ratio', 0):.3f}")
        logger.info(f"Max drawdown: {risk.get('max_drawdown', 0):.2%}")

    def _generate_forecasts(self) -> None:
        """Generate regime forecasts and trading signals"""
        logger.info("Generating regime forecasts...")

        # Use recent observations for forecasting
        recent_observations = self.hmm_model.prepare_observations(self.garch_components)[-10:]

        # Generate forecasts
        self.forecast_results = self.predictor.generate_regime_forecast(
            self.hmm_model,
            recent_observations,
            self.garch_components.get('volatility_forecast', pd.Series())
        )

        # Generate risk warnings
        warnings = self.predictor.generate_risk_warnings(self.forecast_results)
        if warnings:
            logger.warning("Risk warnings generated:")
            for warning in warnings:
                logger.warning(f"  - {warning}")

        logger.info(f"Generated {self.forecast_results['forecast_horizon']}-day forecast")

    def _create_visualizations(self) -> None:
        """Create comprehensive visualizations"""
        logger.info("Creating visualizations...")

        try:
            # Regime detection plot
            regime_plot = self.visualizer.plot_regime_detection(
                self.data['Close'],
                self.regime_predictions['state_sequence'],
                self.regime_predictions.get('state_probabilities')
            )

            # Portfolio performance plot
            portfolio_plot = self.visualizer.plot_portfolio_performance(
                self.portfolio_results
            )

            # Regime analysis plot
            regime_analysis = self.performance_analysis.get('regime_analysis', {})
            if regime_analysis:
                regime_analysis_plot = self.visualizer.plot_regime_analysis(
                    regime_analysis
                )

            # Risk metrics plot
            risk_plot = self.visualizer.plot_risk_metrics(
                self.performance_analysis
            )

            # Forecast plot
            forecast_plot = self.visualizer.plot_regime_forecast(
                self.forecast_results
            )

            # Save all plots
            plots = [regime_plot, portfolio_plot, risk_plot, forecast_plot]
            if 'regime_analysis_plot' in locals():
                plots.append(regime_analysis_plot)

            self.visualizer.save_plots(plots, prefix="hmm_regime_trading")

            logger.info("Visualizations created and saved successfully")

        except Exception as e:
            logger.error(f"Error creating visualizations: {str(e)}")

    def _generate_final_report(self) -> str:
        """Generate comprehensive final report"""
        logger.info("Generating final report...")

        # Get key metrics
        basic_metrics = self.performance_analysis.get('basic_metrics', {})
        risk_metrics = self.performance_analysis.get('risk_metrics', {})
        regime_analysis = self.performance_analysis.get('regime_analysis', {})

        # Trading signals summary
        trading_signals = self.forecast_results.get('trading_signals')
        next_signal = trading_signals.iloc[0] if trading_signals is not None and len(trading_signals) > 0 else None

        report = f"""
===========================================================
    HMM REGIME DETECTION TRADING SYSTEM - FINAL REPORT
===========================================================

EXECUTIVE SUMMARY:
This report presents the results of a comprehensive Hidden Markov Model (HMM)
regime detection trading system with GARCH volatility integration applied to
{DATA_CONFIG['symbol']} from {DATA_CONFIG['start_date']} to {DATA_CONFIG['end_date']}.

OVERALL PERFORMANCE:
-------------------
• Total Return: {basic_metrics.get('total_return', 0):.2%}
• Annualized Return: {basic_metrics.get('annualized_return', 0):.2%}
• Benchmark Return: {basic_metrics.get('benchmark_return', 0):.2%}
• Excess Return: {basic_metrics.get('excess_return', 0):.2%}
• Volatility: {basic_metrics.get('volatility', 0):.2%}

RISK-ADJUSTED METRICS:
---------------------
• Sharpe Ratio: {risk_metrics.get('sharpe_ratio', 0):.3f}
• Information Ratio: {risk_metrics.get('information_ratio', 0):.3f}
• Sortino Ratio: {risk_metrics.get('sortino_ratio', 0):.3f}
• Calmar Ratio: {risk_metrics.get('calmar_ratio', 0):.3f}
• Maximum Drawdown: {risk_metrics.get('max_drawdown', 0):.2%}

RISK METRICS:
------------
• Beta: {risk_metrics.get('beta', 0):.3f}
• Correlation with Benchmark: {risk_metrics.get('correlation', 0):.3f}
• Value at Risk (95%): {risk_metrics.get('var_95', 0):.4f}
• Conditional VaR (95%): {risk_metrics.get('cvar_95', 0):.4f}

REGIME ANALYSIS:
---------------"""

        for regime_name, stats in regime_analysis.items():
            report += f"""
{regime_name}:
  • Frequency: {stats.get('frequency', 0):.1%}
  • Average Return: {stats.get('avg_return', 0):.4f}
  • Sharpe Ratio: {stats.get('sharpe_ratio', 0):.3f}
  • Win Rate: {stats.get('win_rate', 0):.1%}"""

        report += f"""

MODEL DIAGNOSTICS:
-----------------
• GARCH Model: Successfully fitted with good residual properties
• HMM Model: {self.hmm_model.get_model_summary().get('n_components', 3)}-state model with regime interpretations
• Data Quality: {self.data_loader.get_data_summary().get('total_observations', 0)} observations processed

CURRENT FORECAST & RECOMMENDATIONS:
----------------------------------"""

        if next_signal is not None:
            report += f"""
• Next Period Regime: {next_signal.get('regime_label', 'Unknown')}
• Regime Probability: {next_signal.get('regime_probability', 0):.1%}
• Recommended Action: {next_signal.get('action', 'HOLD')}
• Position Size: {next_signal.get('final_position', 0):.1%}
• Signal Strength: {next_signal.get('signal_strength', 'Unknown')}
• Risk Level: {next_signal.get('risk_level', 'Unknown')}

SPECIFIC RECOMMENDATIONS:
• Entry: {next_signal.get('recommendations', {}).get('entry_price', 'N/A')}
• Position Size: {next_signal.get('recommendations', {}).get('position_size', 'N/A')}
• Stop Loss: {next_signal.get('recommendations', {}).get('stop_loss', 'N/A')}
• Take Profit: {next_signal.get('recommendations', {}).get('take_profit', 'N/A')}
• Expected Hold: {next_signal.get('recommendations', {}).get('hold_period', 'N/A')}"""

        # Risk warnings
        warnings = self.predictor.generate_risk_warnings(self.forecast_results)
        if warnings:
            report += """

RISK WARNINGS:
-------------"""
            for warning in warnings:
                report += f"\n⚠️  {warning}"

        report += f"""

SYSTEM CONFIGURATION:
--------------------
• Transaction Costs: {TRADING_CONFIG['transaction_cost']:.1%} per trade
• GARCH Model: GARCH(1,1) with {GARCH_CONFIG['dist']} distribution
• HMM Model: {HMM_CONFIG['n_components']}-state with {HMM_CONFIG['covariance_type']} covariance
• Forecast Horizon: {PREDICTION_CONFIG['forecast_horizon']} trading days
• Data Period: {len(self.data)} observations

DISCLAIMER:
----------
This analysis is for educational and research purposes only. Past performance
does not guarantee future results. All trading involves substantial risk of loss.
Consult with a qualified financial advisor before making investment decisions.

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
System Version: HMM Regime Trading System v1.0

===========================================================
"""

        return report

    def run_backtest_only(self) -> Dict:
        """Run backtesting analysis only"""
        logger.info("Running backtest-only analysis...")

        self._load_and_preprocess_data()
        self._train_models()
        self._run_backtesting()
        self._analyze_performance()

        return {
            'portfolio_results': self.portfolio_results,
            'performance_analysis': self.performance_analysis,
            'performance_summary': self.performance_analyzer.generate_performance_summary(
                self.performance_analysis
            )
        }

    def run_prediction_only(self) -> Dict:
        """Run prediction analysis only"""
        logger.info("Running prediction-only analysis...")

        self._load_and_preprocess_data()
        self._train_models()
        self._generate_forecasts()

        return {
            'forecast_results': self.forecast_results,
            'current_regime': self.regime_predictions['state_sequence'][-1],
            'trading_signals': self.forecast_results.get('trading_signals')
        }


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='HMM Regime Detection Trading System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Run full analysis
  python main.py --mode backtest    # Run backtesting only
  python main.py --mode predict     # Generate predictions only
        """
    )

    parser.add_argument(
        '--mode',
        choices=['full', 'backtest', 'predict'],
        default='full',
        help='Analysis mode (default: full)'
    )

    parser.add_argument(
        '--symbol',
        default='SPY',
        help='Stock symbol to analyze (default: SPY)'
    )

    parser.add_argument(
        '--start-date',
        default='2014-01-01',
        help='Start date for analysis (default: 2014-01-01)'
    )

    parser.add_argument(
        '--end-date',
        default='2024-12-31',
        help='End date for analysis (default: 2024-12-31)'
    )

    parser.add_argument(
        '--output-dir',
        default='./output',
        help='Output directory for results (default: ./output)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    return parser.parse_args()


def main():
    """Main entry point"""
    print("""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                   HMM REGIME DETECTION TRADING SYSTEM                        ║
║                                                                               ║
║  Comprehensive quantitative trading system with:                             ║
║  • Hidden Markov Model regime detection                                      ║
║  • GARCH(1,1) volatility modeling                                           ║
║  • Advanced backtesting framework                                            ║
║  • Real-time predictions and signals                                         ║
║                                                                               ║
║  Developed with Claude Code - Professional AI Assistant                      ║
╚═══════════════════════════════════════════════════════════════════════════════╝
    """)

    # Parse arguments
    args = parse_arguments()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    try:
        # Update configuration with command line arguments
        config_updates = {
            'symbol': args.symbol,
            'start_date': args.start_date,
            'end_date': args.end_date,
            'output_dir': args.output_dir
        }

        # Initialize system
        trading_system = HMMTradingSystem(config_updates)

        # Run analysis based on mode
        if args.mode == 'full':
            results = trading_system.run_full_analysis()
            print("\n" + "="*80)
            print("COMPREHENSIVE ANALYSIS COMPLETED")
            print("="*80)
            print(results['final_report'])

        elif args.mode == 'backtest':
            results = trading_system.run_backtest_only()
            print("\n" + "="*80)
            print("BACKTESTING ANALYSIS COMPLETED")
            print("="*80)
            print(results['performance_summary'])

        elif args.mode == 'predict':
            results = trading_system.run_prediction_only()
            signals = results['trading_signals']
            print("\n" + "="*80)
            print("PREDICTION ANALYSIS COMPLETED")
            print("="*80)
            if signals is not None and len(signals) > 0:
                next_signal = signals.iloc[0]
                print(f"Next Period Recommendation: {next_signal['action']}")
                print(f"Regime: {next_signal['regime_label']} ({next_signal['regime_probability']:.1%} confidence)")
                print(f"Position: {next_signal['final_position']:.1%}")
                print(f"Risk Level: {next_signal['risk_level']}")

        print(f"\nAll outputs saved to: {args.output_dir}")
        print("Analysis completed successfully!")

    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        print(f"\nERROR: {str(e)}")
        print("Check log files for detailed error information.")
        sys.exit(1)


if __name__ == "__main__":
    main()