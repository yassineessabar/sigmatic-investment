#!/usr/bin/env python3

"""
Live Paper Trading Script for VPS Deployment
Runs the neutral pairs strategy continuously with real-time data.
"""

import sys
import os
import time
import logging
import signal
from datetime import datetime, timedelta
from pathlib import Path
import schedule
from typing import Dict, List

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.config import ConfigManager
from src.utils.logger import setup_logging
from data.binance_client import BinanceDataClient
from src.strategies.neutral_pairs import compute_signals, validate_signals
from src.execution.executor import ExecutionEngine
from src.backtest.portfolio import PortfolioState


class LiveTradingBot:
    def __init__(self, config_path: str = "config/two_week_config.yaml"):
        self.config_manager = ConfigManager()
        self.config = self.config_manager.load_config(config_path)

        # Setup logging
        setup_logging(self.config.get('logging', {}))
        self.logger = logging.getLogger(__name__)

        # Initialize components
        binance_config = self.config.get('binance', {})
        self.binance_client = BinanceDataClient(
            api_key=binance_config.get('api_key', ''),
            api_secret=binance_config.get('api_secret', ''),
            testnet=binance_config.get('testnet', True)
        )
        self.executor = ExecutionEngine(self.binance_client, self.config)

        # Portfolio tracking
        initial_capital = self.config.get('backtest', {}).get('initial_capital', 100000)
        self.portfolio = PortfolioState(initial_capital)

        # Runtime state
        self.running = True
        self.start_time = datetime.now()
        self.cycle_count = 0

        # Setup signal handlers
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

        self.logger.info(f"LiveTradingBot initialized with config: {config_path}")
        self.logger.info(f"Trading mode: {self.config.get('execution', {}).get('mode', 'paper')}")

    def _signal_handler(self, signum, frame):
        self.logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.running = False

    def run_trading_cycle(self):
        """Execute one complete trading cycle"""
        try:
            self.cycle_count += 1
            cycle_start = datetime.now()

            self.logger.info(f"Starting trading cycle #{self.cycle_count}")

            # Get market data for all symbols
            symbols = self._get_all_symbols()
            market_data = {}

            for symbol in symbols:
                try:
                    data = self.binance_client.get_historical_klines(
                        symbol=symbol,
                        interval='1h',
                        limit=200
                    )
                    market_data[symbol] = data
                except Exception as e:
                    self.logger.warning(f"Failed to get data for {symbol}: {e}")

            if not market_data:
                self.logger.warning("No market data received, skipping cycle")
                return

            # Generate signals
            signals = compute_signals(market_data, self.config)

            if signals:
                self.logger.info(f"Generated {len(signals)} trading signals")

                # Execute trades
                execution_results = self.executor.process_signals(signals, market_data)

                # Log execution results
                for result in execution_results:
                    self.logger.info(f"Trade executed: {result}")

                # Update portfolio state
                self._update_portfolio_state()

            else:
                self.logger.info("No trading signals generated")

            # Log performance metrics
            self._log_performance_metrics()

            cycle_duration = datetime.now() - cycle_start
            self.logger.info(f"Trading cycle #{self.cycle_count} completed in {cycle_duration}")

        except Exception as e:
            self.logger.error(f"Error in trading cycle: {e}", exc_info=True)

    def _get_all_symbols(self) -> List[str]:
        """Get all symbols used in pairs"""
        symbols = set()
        for pair in self.config.get('pairs', []):
            symbols.add(pair['base'])
            symbols.add(pair['hedge'])
        return list(symbols)

    def _update_portfolio_state(self):
        """Update portfolio state with current positions"""
        try:
            account_summary = self.executor.get_account_summary()
            positions = self.executor.get_all_positions()

            # Log current state
            self.logger.info(f"Account Summary: {account_summary}")
            if positions:
                self.logger.info(f"Open Positions: {positions}")

        except Exception as e:
            self.logger.error(f"Error updating portfolio state: {e}")

    def _log_performance_metrics(self):
        """Log key performance metrics"""
        try:
            runtime = datetime.now() - self.start_time
            account_summary = self.executor.get_account_summary()

            if 'total_equity' in account_summary:
                initial_capital = self.config.get('backtest', {}).get('initial_capital', 100000)
                total_return = (account_summary['total_equity'] - initial_capital) / initial_capital

                self.logger.info(f"Performance Metrics:")
                self.logger.info(f"  Runtime: {runtime}")
                self.logger.info(f"  Total Equity: ${account_summary.get('total_equity', 0):,.2f}")
                self.logger.info(f"  Cash: ${account_summary.get('cash', 0):,.2f}")
                self.logger.info(f"  Unrealized PnL: ${account_summary.get('unrealized_pnl', 0):,.2f}")
                self.logger.info(f"  Total Return: {total_return:.4f} ({total_return*100:.2f}%)")
                self.logger.info(f"  Cycles Completed: {self.cycle_count}")

        except Exception as e:
            self.logger.error(f"Error logging performance metrics: {e}")

    def run_daily_report(self):
        """Generate and log daily performance report"""
        try:
            self.logger.info("Generating daily report...")
            self._log_performance_metrics()

            # Additional daily metrics could be added here
            # e.g., daily PnL, number of trades, etc.

        except Exception as e:
            self.logger.error(f"Error generating daily report: {e}")

    def start(self):
        """Start the trading bot with scheduled execution"""
        self.logger.info("Starting Live Trading Bot...")
        self.logger.info(f"Config: {self.config}")

        # Schedule trading cycles every hour
        interval = self.config.get('execution', {}).get('interval', '1h')
        if interval == '1h':
            schedule.every().hour.at(":00").do(self.run_trading_cycle)
        elif interval == '15m':
            schedule.every(15).minutes.do(self.run_trading_cycle)
        elif interval == '5m':
            schedule.every(5).minutes.do(self.run_trading_cycle)
        else:
            self.logger.warning(f"Unknown interval {interval}, defaulting to 1h")
            schedule.every().hour.at(":00").do(self.run_trading_cycle)

        # Schedule daily reports
        if self.config.get('monitoring', {}).get('daily_reports', True):
            schedule.every().day.at("00:00").do(self.run_daily_report)

        # Run initial cycle
        self.logger.info("Running initial trading cycle...")
        self.run_trading_cycle()

        # Main execution loop
        while self.running:
            try:
                schedule.run_pending()
                time.sleep(60)  # Check every minute

                # Health check
                if self.cycle_count > 0 and self.cycle_count % 24 == 0:  # Every 24 cycles (24 hours)
                    self.logger.info(f"Health check: Bot has been running for {datetime.now() - self.start_time}")

            except KeyboardInterrupt:
                self.logger.info("Keyboard interrupt received, shutting down...")
                break
            except Exception as e:
                self.logger.error(f"Error in main loop: {e}", exc_info=True)
                time.sleep(60)  # Wait before retrying

        self.logger.info("Trading bot stopped")


def main():
    """Main entry point"""
    try:
        # Create necessary directories
        os.makedirs('logs', exist_ok=True)

        # Start the bot
        bot = LiveTradingBot()
        bot.start()

    except Exception as e:
        logging.error(f"Failed to start trading bot: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()