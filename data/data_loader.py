"""
Data acquisition and preprocessing for HMM regime detection
"""

import numpy as np
import pandas as pd
import yfinance as yf
import logging
from datetime import datetime, timedelta
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from config.settings import DATA_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    """
    Handles data acquisition, preprocessing, and validation for SPY data
    """

    def __init__(self, symbol: str = None, start_date: str = None, end_date: str = None):
        self.symbol = symbol or DATA_CONFIG['symbol']
        self.start_date = start_date or DATA_CONFIG['start_date']
        self.end_date = end_date or DATA_CONFIG['end_date']
        self.raw_data = None
        self.processed_data = None

    def download_data(self) -> pd.DataFrame:
        """
        Download price data from Yahoo Finance
        """
        try:
            logger.info(f"Downloading {self.symbol} data from {self.start_date} to {self.end_date}")
            ticker = yf.Ticker(self.symbol)
            data = ticker.history(start=self.start_date, end=self.end_date, auto_adjust=True)

            if data.empty:
                raise ValueError(f"No data found for symbol {self.symbol}")

            logger.info(f"Downloaded {len(data)} trading days of data")
            self.raw_data = data
            return data

        except Exception as e:
            logger.error(f"Error downloading data: {str(e)}")
            raise

    def preprocess_data(self, data: pd.DataFrame = None) -> pd.DataFrame:
        """
        Preprocess raw price data for analysis
        """
        if data is None:
            data = self.raw_data

        if data is None:
            raise ValueError("No data available. Run download_data() first.")

        try:
            # Create a copy to avoid modifying original data
            df = data.copy()

            # Ensure we have required columns
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")

            # Handle missing values
            df = df.dropna()

            # Calculate returns
            df['Returns'] = df['Close'].pct_change()
            df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))

            # Calculate additional features
            df['High_Low_Ratio'] = df['High'] / df['Low']
            df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
            df['Price_MA_20'] = df['Close'].rolling(window=20).mean()
            df['Price_MA_50'] = df['Close'].rolling(window=50).mean()
            df['Volatility_20'] = df['Returns'].rolling(window=20).std() * np.sqrt(252)

            # Remove first row with NaN returns
            df = df.dropna()

            # Validate data quality
            self._validate_data(df)

            logger.info(f"Processed data: {len(df)} observations")
            logger.info(f"Date range: {df.index[0]} to {df.index[-1]}")

            self.processed_data = df
            return df

        except Exception as e:
            logger.error(f"Error preprocessing data: {str(e)}")
            raise

    def _validate_data(self, df: pd.DataFrame) -> None:
        """
        Validate data quality and consistency
        """
        # Check for sufficient data
        if len(df) < 252:  # Less than 1 year of data
            logger.warning("Less than 1 year of data available")

        # Check for extreme returns (likely data errors)
        extreme_returns = df['Returns'].abs() > 0.5  # 50% daily return
        if extreme_returns.any():
            logger.warning(f"Found {extreme_returns.sum()} extreme return observations")

        # Check for zero volume days
        zero_volume = df['Volume'] == 0
        if zero_volume.any():
            logger.warning(f"Found {zero_volume.sum()} zero volume days")

        # Check for price consistency
        price_inconsistent = (df['High'] < df['Low']) | (df['Close'] > df['High']) | (df['Close'] < df['Low'])
        if price_inconsistent.any():
            logger.warning(f"Found {price_inconsistent.sum()} price inconsistencies")

    def split_data(self, train_ratio: float = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into training and testing sets
        """
        if self.processed_data is None:
            raise ValueError("No processed data available. Run preprocess_data() first.")

        train_ratio = train_ratio or DATA_CONFIG['train_ratio']

        n_train = int(len(self.processed_data) * train_ratio)

        train_data = self.processed_data.iloc[:n_train].copy()
        test_data = self.processed_data.iloc[n_train:].copy()

        logger.info(f"Training data: {len(train_data)} observations ({train_data.index[0]} to {train_data.index[-1]})")
        logger.info(f"Testing data: {len(test_data)} observations ({test_data.index[0]} to {test_data.index[-1]})")

        return train_data, test_data

    def get_data_summary(self) -> dict:
        """
        Get summary statistics of the processed data
        """
        if self.processed_data is None:
            raise ValueError("No processed data available. Run preprocess_data() first.")

        df = self.processed_data

        summary = {
            'symbol': self.symbol,
            'start_date': df.index[0],
            'end_date': df.index[-1],
            'total_observations': len(df),
            'mean_return': df['Returns'].mean(),
            'volatility': df['Returns'].std() * np.sqrt(252),
            'sharpe_ratio': (df['Returns'].mean() * 252) / (df['Returns'].std() * np.sqrt(252)),
            'max_drawdown': self._calculate_max_drawdown(df['Close']),
            'skewness': df['Returns'].skew(),
            'kurtosis': df['Returns'].kurtosis(),
            'min_return': df['Returns'].min(),
            'max_return': df['Returns'].max(),
        }

        return summary

    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """
        Calculate maximum drawdown from price series
        """
        cumulative = (1 + prices.pct_change()).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        return drawdown.min()

    def load_and_process(self) -> pd.DataFrame:
        """
        Convenience method to download and process data in one call
        """
        self.download_data()
        return self.preprocess_data()


def main():
    """
    Example usage of DataLoader
    """
    # Initialize data loader
    loader = DataLoader()

    # Load and process data
    data = loader.load_and_process()

    # Get summary
    summary = loader.get_data_summary()
    print("Data Summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")

    # Split data
    train_data, test_data = loader.split_data()

    return data, train_data, test_data


if __name__ == "__main__":
    main()