"""
Multi-Stock HMM Regime Detection System
Run the regime trading model on any stock or portfolio of stocks
"""

import sys
import os
import numpy as np
import pandas as pd
import logging
import warnings
from datetime import datetime
from typing import Dict, List, Optional
import argparse

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

class MultiStockHMMAnalyzer:
    """
    Multi-stock HMM regime detection and trading analysis
    """

    def __init__(self, symbols: List[str], start_date: str = '2014-01-01',
                 end_date: str = '2024-12-31'):
        self.symbols = symbols if isinstance(symbols, list) else [symbols]
        self.start_date = start_date
        self.end_date = end_date
        self.results = {}

    def analyze_single_stock(self, symbol: str, generate_predictions: bool = True,
                           create_visualizations: bool = True) -> Dict:
        """
        Analyze a single stock with HMM regime detection
        """
        try:
            logger.info(f"🎯 Analyzing {symbol}...")

            # Initialize components
            data_loader = DataLoader(symbol=symbol, start_date=self.start_date, end_date=self.end_date)
            garch_model = EnhancedGARCHModel()
            hmm_model = PerformanceOptimizedHMM()
            strategy = PerformanceOptimizedStrategy()

            # Load and process data
            logger.info(f"📊 Loading {symbol} data...")
            historical_data = data_loader.load_and_process()

            if len(historical_data) < 252:
                logger.warning(f"⚠️  {symbol}: Insufficient data ({len(historical_data)} days)")
                return {'error': 'Insufficient data', 'symbol': symbol}

            logger.info(f"✅ {symbol}: Loaded {len(historical_data)} observations")

            # Train models
            logger.info(f"🧠 Training models for {symbol}...")
            garch_model.fit(historical_data['Returns'])
            features = garch_model.extract_regime_features(historical_data['Returns'])
            hmm_model.fit(features, historical_data['Returns'])

            # Generate regime predictions
            regime_predictions = hmm_model.predict_regimes(features, historical_data['Returns'])
            regime_positions = hmm_model.get_regime_positions()

            # Run trading strategy
            logger.info(f"⚡ Running strategy for {symbol}...")
            performance_results = strategy.run_optimized_backtest(
                regime_predictions, historical_data['Close'], regime_positions
            )

            # Store regime predictions for visualization
            performance_results['regime_predictions'] = regime_predictions

            # Generate predictions if requested
            predictions = None
            if generate_predictions:
                try:
                    logger.info(f"🔮 Generating predictions for {symbol}...")
                    predictor = AdvancedPredictor(forecast_horizon=30)
                    predictions = predictor.generate_future_predictions(
                        hmm_model, garch_model, historical_data.tail(252),
                        historical_data['Close'].iloc[-1]
                    )
                except Exception as e:
                    logger.warning(f"⚠️  Could not generate predictions for {symbol}: {str(e)}")

            # Create visualizations if requested
            visualizations = []
            if create_visualizations and predictions:
                try:
                    logger.info(f"🎨 Creating visualizations for {symbol}...")
                    visualizer = AdvancedVisualizer()

                    # Regime analysis plot
                    regime_plot = visualizer.create_regime_analysis_plot(
                        historical_data, regime_predictions
                    )
                    visualizations.append(('Regime Analysis', regime_plot))

                    # Prediction summary plot
                    prediction_plot = visualizer.create_prediction_summary_plot(predictions)
                    visualizations.append(('Predictions', prediction_plot))

                    # Save visualizations
                    figures = [viz[1] for viz in visualizations]
                    saved_files = visualizer.save_all_visualizations(
                        figures, predictions, prefix=f"{symbol.lower()}_hmm"
                    )
                    logger.info(f"📁 {symbol}: Saved {len(saved_files)} visualization files")

                except Exception as e:
                    logger.warning(f"⚠️  Could not create visualizations for {symbol}: {str(e)}")

            result = {
                'symbol': symbol,
                'historical_data': historical_data,
                'performance_results': performance_results,
                'predictions': predictions,
                'visualizations': visualizations,
                'regime_mapping': hmm_model.regime_mapping,
                'optimal_positions': hmm_model.get_regime_positions()
            }

            logger.info(f"✅ {symbol} analysis completed successfully")
            return result

        except Exception as e:
            logger.error(f"❌ Error analyzing {symbol}: {str(e)}")
            return {'error': str(e), 'symbol': symbol}

    def analyze_multiple_stocks(self, generate_predictions: bool = True,
                              create_visualizations: bool = True) -> Dict:
        """
        Analyze multiple stocks and compare results
        """
        try:
            logger.info(f"🎯 Multi-Stock Analysis: {', '.join(self.symbols)}")

            results = {}
            performance_summary = []

            for symbol in self.symbols:
                result = self.analyze_single_stock(
                    symbol, generate_predictions, create_visualizations
                )
                results[symbol] = result

                # Collect performance metrics for comparison
                if 'performance_results' in result and 'metrics' in result['performance_results']:
                    metrics = result['performance_results']['metrics']
                    summary = {
                        'Symbol': symbol,
                        'Total_Return': metrics.get('total_return', 0),
                        'Sharpe_Ratio': metrics.get('sharpe_ratio', 0),
                        'Max_Drawdown': metrics.get('max_drawdown', 0),
                        'Win_Rate': metrics.get('win_rate', 0),
                        'Volatility': metrics.get('volatility', 0),
                        'Information_Ratio': metrics.get('information_ratio', 0)
                    }
                    performance_summary.append(summary)

            # Create comparison analysis
            if performance_summary:
                comparison_df = pd.DataFrame(performance_summary)
                comparison_analysis = self._create_comparison_analysis(comparison_df)
                results['comparison'] = comparison_analysis

            self.results = results
            self._generate_multi_stock_report()

            return results

        except Exception as e:
            logger.error(f"❌ Error in multi-stock analysis: {str(e)}")
            raise

    def _create_comparison_analysis(self, comparison_df: pd.DataFrame) -> Dict:
        """Create comparative analysis of multiple stocks"""
        try:
            # Rank stocks by different metrics
            rankings = {
                'Best_Total_Return': comparison_df.nlargest(3, 'Total_Return')['Symbol'].tolist(),
                'Best_Sharpe_Ratio': comparison_df.nlargest(3, 'Sharpe_Ratio')['Symbol'].tolist(),
                'Lowest_Drawdown': comparison_df.nsmallest(3, 'Max_Drawdown')['Symbol'].tolist(),
                'Highest_Win_Rate': comparison_df.nlargest(3, 'Win_Rate')['Symbol'].tolist()
            }

            # Calculate statistics
            stats = {
                'Mean_Return': comparison_df['Total_Return'].mean(),
                'Mean_Sharpe': comparison_df['Sharpe_Ratio'].mean(),
                'Mean_Drawdown': comparison_df['Max_Drawdown'].mean(),
                'Best_Performer': comparison_df.loc[comparison_df['Sharpe_Ratio'].idxmax(), 'Symbol'],
                'Worst_Performer': comparison_df.loc[comparison_df['Sharpe_Ratio'].idxmin(), 'Symbol']
            }

            # Target achievement
            target_achievers = {
                'Return_Target_30pct': comparison_df[comparison_df['Total_Return'] >= 0.30]['Symbol'].tolist(),
                'Sharpe_Target_1_5': comparison_df[comparison_df['Sharpe_Ratio'] >= 1.5]['Symbol'].tolist(),
                'Both_Targets': comparison_df[
                    (comparison_df['Total_Return'] >= 0.30) &
                    (comparison_df['Sharpe_Ratio'] >= 1.5)
                ]['Symbol'].tolist()
            }

            return {
                'performance_df': comparison_df,
                'rankings': rankings,
                'stats': stats,
                'target_achievers': target_achievers
            }

        except Exception as e:
            logger.error(f"Error creating comparison: {str(e)}")
            return {}

    def _generate_multi_stock_report(self):
        """Generate comprehensive multi-stock report"""
        try:
            print("\n" + "="*100)
            print("🎯 MULTI-STOCK HMM REGIME TRADING ANALYSIS REPORT")
            print("="*100)

            if not self.results:
                print("❌ No results available")
                return

            # Individual stock performance
            print(f"\n📊 INDIVIDUAL STOCK PERFORMANCE:")
            print(f"{'Stock':<8} {'Total Return':<12} {'Sharpe Ratio':<12} {'Max Drawdown':<12} {'Win Rate':<10} {'Status':<15}")
            print("-" * 80)

            for symbol, result in self.results.items():
                if symbol == 'comparison':
                    continue

                if 'error' in result:
                    print(f"{symbol:<8} {'ERROR':<12} {result['error'][:40]:<40}")
                    continue

                metrics = result.get('performance_results', {}).get('metrics', {})
                if metrics:
                    total_return = metrics.get('total_return', 0)
                    sharpe_ratio = metrics.get('sharpe_ratio', 0)
                    max_drawdown = metrics.get('max_drawdown', 0)
                    win_rate = metrics.get('win_rate', 0)

                    # Status based on targets
                    return_ok = total_return >= 0.30
                    sharpe_ok = sharpe_ratio >= 1.5

                    if return_ok and sharpe_ok:
                        status = "🎉 EXCELLENT"
                    elif return_ok or sharpe_ok:
                        status = "✅ GOOD"
                    else:
                        status = "⚠️  NEEDS WORK"

                    print(f"{symbol:<8} {total_return:>10.1%} {sharpe_ratio:>10.2f} "
                          f"{max_drawdown:>10.1%} {win_rate:>8.1%} {status:<15}")

            # Comparison analysis
            if 'comparison' in self.results:
                comparison = self.results['comparison']
                rankings = comparison.get('rankings', {})
                stats = comparison.get('stats', {})
                achievers = comparison.get('target_achievers', {})

                print(f"\n🏆 COMPARATIVE RANKINGS:")
                print(f"   Best Total Return:     {', '.join(rankings.get('Best_Total_Return', []))}")
                print(f"   Best Sharpe Ratio:     {', '.join(rankings.get('Best_Sharpe_Ratio', []))}")
                print(f"   Lowest Drawdown:       {', '.join(rankings.get('Lowest_Drawdown', []))}")
                print(f"   Highest Win Rate:      {', '.join(rankings.get('Highest_Win_Rate', []))}")

                print(f"\n📈 PORTFOLIO STATISTICS:")
                print(f"   Average Return:        {stats.get('Mean_Return', 0):.1%}")
                print(f"   Average Sharpe:        {stats.get('Mean_Sharpe', 0):.2f}")
                print(f"   Average Drawdown:      {stats.get('Mean_Drawdown', 0):.1%}")
                print(f"   Best Overall:          {stats.get('Best_Performer', 'N/A')}")
                print(f"   Worst Overall:         {stats.get('Worst_Performer', 'N/A')}")

                print(f"\n🎯 TARGET ACHIEVEMENT:")
                print(f"   30%+ Return:           {', '.join(achievers.get('Return_Target_30pct', ['None']))}")
                print(f"   1.5+ Sharpe:           {', '.join(achievers.get('Sharpe_Target_1_5', ['None']))}")
                print(f"   Both Targets:          {', '.join(achievers.get('Both_Targets', ['None']))}")

                if achievers.get('Both_Targets'):
                    print(f"\n🎉 SUCCESS: {len(achievers['Both_Targets'])} stocks achieved both targets!")
                else:
                    print(f"\n💡 RECOMMENDATION: Consider portfolio optimization or strategy refinement")

            # Current predictions summary
            print(f"\n🔮 CURRENT MARKET PREDICTIONS:")
            for symbol, result in self.results.items():
                if symbol == 'comparison' or 'error' in result:
                    continue

                predictions = result.get('predictions')
                if predictions:
                    current_state = predictions.get('current_state', {})
                    signals = predictions.get('trading_signals')

                    if signals is not None and len(signals) > 0:
                        next_signal = signals.iloc[0]
                        print(f"   {symbol:<8}: {current_state.get('current_regime', 'Unknown')} "
                              f"({current_state.get('confidence', 0):.1%}) → "
                              f"{next_signal.get('action', 'HOLD')} "
                              f"({next_signal.get('signal_strength', 'Unknown')})")

            print(f"\n📁 FILES GENERATED:")
            total_files = 0
            for symbol, result in self.results.items():
                if 'visualizations' in result:
                    num_viz = len(result['visualizations'])
                    if num_viz > 0:
                        print(f"   {symbol}: {num_viz} visualization files")
                        total_files += num_viz

            print(f"   Total: {total_files} files created")
            print(f"\n🕒 Analysis completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("="*100)

        except Exception as e:
            logger.error(f"Error generating multi-stock report: {str(e)}")

    def get_best_performers(self, metric: str = 'sharpe_ratio', top_n: int = 3) -> List[str]:
        """Get top performing stocks by specified metric"""
        if not self.results or 'comparison' not in self.results:
            return []

        comparison = self.results['comparison']
        performance_df = comparison.get('performance_df')

        if performance_df is None or performance_df.empty:
            return []

        # Map metric names
        metric_map = {
            'sharpe_ratio': 'Sharpe_Ratio',
            'total_return': 'Total_Return',
            'win_rate': 'Win_Rate',
            'information_ratio': 'Information_Ratio'
        }

        column = metric_map.get(metric, 'Sharpe_Ratio')

        if metric == 'max_drawdown':  # Lower is better for drawdown
            top_stocks = performance_df.nsmallest(top_n, 'Max_Drawdown')
        else:
            top_stocks = performance_df.nlargest(top_n, column)

        return top_stocks['Symbol'].tolist()


def main():
    """Command line interface for multi-stock analysis"""
    parser = argparse.ArgumentParser(description='Multi-Stock HMM Regime Trading Analysis')

    parser.add_argument('--symbols', '-s', nargs='+', default=['SPY'],
                       help='Stock symbols to analyze (e.g., --symbols AAPL MSFT GOOGL)')

    parser.add_argument('--start-date', default='2014-01-01',
                       help='Start date for analysis (YYYY-MM-DD)')

    parser.add_argument('--end-date', default='2024-12-31',
                       help='End date for analysis (YYYY-MM-DD)')

    parser.add_argument('--no-predictions', action='store_true',
                       help='Skip future predictions generation')

    parser.add_argument('--no-visuals', action='store_true',
                       help='Skip visualization creation')

    parser.add_argument('--single', action='store_true',
                       help='Analyze stocks individually without comparison')

    args = parser.parse_args()

    print(f"""
╔════════════════════════════════════════════════════════════════════════════════╗
║                   📊 MULTI-STOCK HMM REGIME ANALYZER                         ║
║                                                                                ║
║  Analyzing: {', '.join(args.symbols):<60} ║
║  Period: {args.start_date} to {args.end_date}                                      ║
║  Predictions: {'No' if args.no_predictions else 'Yes':<55} ║
║  Visualizations: {'No' if args.no_visuals else 'Yes':<50} ║
╚════════════════════════════════════════════════════════════════════════════════╝
    """)

    try:
        analyzer = MultiStockHMMAnalyzer(
            symbols=args.symbols,
            start_date=args.start_date,
            end_date=args.end_date
        )

        if args.single or len(args.symbols) == 1:
            # Single stock analysis
            for symbol in args.symbols:
                result = analyzer.analyze_single_stock(
                    symbol,
                    generate_predictions=not args.no_predictions,
                    create_visualizations=not args.no_visuals
                )

                if 'error' not in result:
                    metrics = result['performance_results']['metrics']
                    print(f"\n🎯 {symbol} Results:")
                    print(f"   Total Return: {metrics['total_return']:.1%}")
                    print(f"   Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
                    print(f"   Max Drawdown: {metrics['max_drawdown']:.1%}")
        else:
            # Multi-stock analysis with comparison
            results = analyzer.analyze_multiple_stocks(
                generate_predictions=not args.no_predictions,
                create_visualizations=not args.no_visuals
            )

            # Show best performers
            best_sharpe = analyzer.get_best_performers('sharpe_ratio', 3)
            best_return = analyzer.get_best_performers('total_return', 3)

            print(f"\n🏆 TOP PERFORMERS:")
            print(f"   Best Sharpe Ratios: {', '.join(best_sharpe)}")
            print(f"   Best Total Returns: {', '.join(best_return)}")

    except Exception as e:
        print(f"\n❌ Analysis failed: {str(e)}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())