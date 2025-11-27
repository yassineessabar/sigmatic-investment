#!/usr/bin/env python3

"""
Paper Trading Script for Neutral Pairs Strategy
Runs live paper trading with real-time data from Binance.
"""

import sys
import os
import time
import logging
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / 'src'))

# Load environment variables
load_dotenv()

from data.binance_client import BinanceDataClient
from src.execution.paper_trader import PaperTradingEngine
from src.utils.config import ConfigManager
from src.utils.logger import setup_logging

class LivePaperTrader:
    def __init__(self, config_path="config/vps_live_config.yaml"):
        # Load configuration
        self.config_manager = ConfigManager()
        self.config = self.config_manager.load_config(config_path)

        # Setup logging
        setup_logging(self.config.get('logging', {}))
        self.logger = logging.getLogger(__name__)

        # Initialize Binance client
        api_key = os.getenv('BINANCE_API_KEY')
        api_secret = os.getenv('BINANCE_API_SECRET')
        testnet = self.config.get('binance', {}).get('testnet', True)

        self.binance_client = BinanceDataClient(api_key, api_secret, testnet=testnet)

        # Initialize paper trading engine
        initial_balance = self.config.get('backtest', {}).get('initial_capital', 100000)
        self.paper_trader = PaperTradingEngine(initial_balance, self.config)

        # Trading pairs from config
        self.pairs = self.config.get('pairs', [])

        self.logger.info(f"Initialized paper trader with ${initial_balance} starting balance")
        self.logger.info(f"Monitoring {len(self.pairs)} trading pairs")

    def get_current_prices(self):
        """Get current prices for all symbols"""
        prices = {}
        try:
            for pair in self.pairs:
                base_symbol = pair['base']
                hedge_symbol = pair['hedge']

                # Get current prices
                base_data = self.binance_client.get_historical_klines(
                    base_symbol, '1m', limit=1
                )
                hedge_data = self.binance_client.get_historical_klines(
                    hedge_symbol, '1m', limit=1
                )

                if not base_data.empty and not hedge_data.empty:
                    prices[base_symbol] = float(base_data['close'].iloc[-1])
                    prices[hedge_symbol] = float(hedge_data['close'].iloc[-1])

        except Exception as e:
            self.logger.error(f"Error getting prices: {e}")

        return prices

    def calculate_z_score(self, pair, lookback_periods=100):
        """Calculate z-score for pair ratio"""
        try:
            base_symbol = pair['base']
            hedge_symbol = pair['hedge']

            # Get historical data
            base_data = self.binance_client.get_historical_klines(
                base_symbol, '1h', limit=lookback_periods
            )
            hedge_data = self.binance_client.get_historical_klines(
                hedge_symbol, '1h', limit=lookback_periods
            )

            if len(base_data) < lookback_periods or len(hedge_data) < lookback_periods:
                self.logger.warning(f"Insufficient data for {base_symbol}/{hedge_symbol}")
                return None

            # Calculate ratio
            ratio = base_data['close'] / hedge_data['close']

            # Calculate z-score
            mean_ratio = ratio.mean()
            std_ratio = ratio.std()
            current_ratio = ratio.iloc[-1]

            if std_ratio == 0:
                return 0

            z_score = (current_ratio - mean_ratio) / std_ratio

            return float(z_score)

        except Exception as e:
            self.logger.error(f"Error calculating z-score for {pair}: {e}")
            return None

    def check_signals(self):
        """Check for trading signals"""
        signals = []

        for pair in self.pairs:
            try:
                z_score = self.calculate_z_score(pair, pair.get('lookback', 100))

                if z_score is None:
                    continue

                entry_threshold = pair.get('entry_z', 2.0)
                exit_threshold = pair.get('exit_z', 0.5)

                signal = {
                    'pair': pair,
                    'z_score': z_score,
                    'base_symbol': pair['base'],
                    'hedge_symbol': pair['hedge'],
                    'action': None
                }

                # Entry signals
                if abs(z_score) > entry_threshold:
                    if z_score > 0:
                        signal['action'] = 'short_base_long_hedge'
                    else:
                        signal['action'] = 'long_base_short_hedge'

                # Exit signals (if we have positions)
                elif abs(z_score) < exit_threshold:
                    signal['action'] = 'exit'

                signals.append(signal)

                self.logger.info(f"{pair['base']}/{pair['hedge']}: Z-score = {z_score:.3f}")

            except Exception as e:
                self.logger.error(f"Error checking signals for {pair}: {e}")

        return signals

    def execute_signals(self, signals):
        """Execute trading signals"""
        current_prices = self.get_current_prices()

        for signal in signals:
            if signal['action'] and signal['action'] != 'exit':
                try:
                    pair = signal['pair']
                    base_symbol = signal['base_symbol']
                    hedge_symbol = signal['hedge_symbol']
                    max_notional = pair.get('max_notional', 1000)

                    base_price = current_prices.get(base_symbol)
                    hedge_price = current_prices.get(hedge_symbol)

                    if not base_price or not hedge_price:
                        self.logger.warning(f"Missing prices for {base_symbol}/{hedge_symbol}")
                        continue

                    # Calculate position sizes
                    base_size = max_notional / base_price / 2  # Half allocation each
                    hedge_size = max_notional / hedge_price / 2

                    if signal['action'] == 'short_base_long_hedge':
                        # Short base, long hedge
                        self.logger.info(f"Signal: Short {base_size:.6f} {base_symbol} @ ${base_price}")
                        self.logger.info(f"Signal: Long {hedge_size:.6f} {hedge_symbol} @ ${hedge_price}")

                    elif signal['action'] == 'long_base_short_hedge':
                        # Long base, short hedge
                        self.logger.info(f"Signal: Long {base_size:.6f} {base_symbol} @ ${base_price}")
                        self.logger.info(f"Signal: Short {hedge_size:.6f} {hedge_symbol} @ ${hedge_price}")

                except Exception as e:
                    self.logger.error(f"Error executing signal: {e}")

    def run(self):
        """Main trading loop"""
        self.logger.info("🚀 Starting Paper Trading...")
        self.logger.info(f"Testnet mode: {self.config.get('binance', {}).get('testnet', True)}")

        interval = self.config.get('execution', {}).get('interval', '1h')

        # Convert interval to seconds
        if interval == '1m':
            sleep_time = 60
        elif interval == '5m':
            sleep_time = 300
        elif interval == '1h':
            sleep_time = 3600
        else:
            sleep_time = 3600  # Default to 1 hour

        try:
            while True:
                start_time = datetime.now()
                self.logger.info(f"\n--- Trading Cycle Started at {start_time} ---")

                # Check for signals
                signals = self.check_signals()

                # Execute signals
                if signals:
                    self.execute_signals(signals)
                else:
                    self.logger.info("No signals generated")

                # Show current balance
                self.logger.info(f"Current paper balance: ${self.paper_trader.cash_balance:,.2f}")

                # Sleep until next cycle
                self.logger.info(f"Sleeping for {sleep_time} seconds until next cycle...")
                time.sleep(sleep_time)

        except KeyboardInterrupt:
            self.logger.info("\n⏹️ Stopping paper trading...")
        except Exception as e:
            self.logger.error(f"Error in trading loop: {e}")

def main():
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = "config/vps_live_config.yaml"

    trader = LivePaperTrader(config_path)
    trader.run()

if __name__ == "__main__":
    main()