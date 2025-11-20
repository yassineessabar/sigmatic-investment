import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import time
import logging
from datetime import datetime
from typing import Dict, List

from src.utils.config import ConfigManager
from src.utils.logger import setup_logging
from data.data_loader import DataLoader
from data.binance_client import BinanceDataClient
from .executor import ExecutionEngine
from ..strategies.neutral_pairs import compute_signals, validate_signals

logger = logging.getLogger(__name__)


class OrderRouter:
    def __init__(self, config: Dict):
        self.config = config
        self.execution_engine = None
        self.data_loader = None
        self.setup_components()

    def setup_components(self):
        if self.config['execution']['mode'] == 'live':
            binance_client = BinanceDataClient(
                api_key=os.getenv('BINANCE_API_KEY'),
                api_secret=os.getenv('BINANCE_API_SECRET'),
                testnet=self.config['binance']['testnet']
            )
            self.execution_engine = ExecutionEngine(binance_client, self.config)
            self.data_loader = DataLoader(binance_client)
        else:  # paper mode
            self.execution_engine = ExecutionEngine(None, self.config)
            # For paper mode, we can still use a mock client for data
            mock_client = BinanceDataClient(
                api_key="mock_key",
                api_secret="mock_secret",
                testnet=True
            )
            self.data_loader = DataLoader(mock_client)

    def run_trading_loop(self):
        logger.info("Starting order routing and execution loop...")

        symbols = []
        for pair in self.config['pairs']:
            symbols.extend([pair['base'], pair['hedge']])
        symbols = list(set(symbols))

        while True:
            try:
                current_data = self.data_loader.get_live_data_slice(
                    symbols=symbols,
                    interval=self.config['execution']['interval'],
                    lookback_periods=max([p['lookback'] for p in self.config['pairs']])
                )

                aligned_data = self.data_loader.align_data(current_data)

                raw_signals = compute_signals(aligned_data, self.config)
                validated_signals = validate_signals(raw_signals, self.config)

                if validated_signals:
                    execution_results = self.execution_engine.process_signals(
                        validated_signals, aligned_data
                    )

                    logger.info(f"Executed {len(execution_results)} orders")
                    for result in execution_results:
                        logger.info(f"Order executed: {result}")

                    self._log_portfolio_status()

                interval_seconds = self._interval_to_seconds(self.config['execution']['interval'])
                time.sleep(interval_seconds)

            except KeyboardInterrupt:
                logger.info("Trading loop stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                time.sleep(30)

    def _log_portfolio_status(self):
        try:
            account_summary = self.execution_engine.get_account_summary()
            positions = self.execution_engine.get_all_positions()

            logger.info("=== Portfolio Status ===")
            logger.info(f"Account Summary: {account_summary}")

            if positions:
                logger.info("Current Positions:")
                for symbol, position in positions.items():
                    if position.get('size', 0) != 0:
                        logger.info(
                            f"  {symbol}: {position['side']} {position['size']:.4f} "
                            f"@ {position.get('current_price', 0):.4f} "
                            f"PnL: {position.get('unrealized_pnl', 0):.2f}"
                        )
            else:
                logger.info("No open positions")

            logger.info("========================")

        except Exception as e:
            logger.error(f"Error logging portfolio status: {e}")

    def process_target_positions(self, target_positions: Dict):
        try:
            current_positions = self.execution_engine.get_all_positions()

            from .executor import compute_order_diff
            orders = compute_order_diff(current_positions, target_positions)

            execution_results = []
            for order in orders:
                if self.config["execution"]["mode"] == "live":
                    if self.execution_engine.binance_client:
                        result = self.execution_engine.binance_client.place_order(
                            symbol=order['symbol'],
                            side=order['side'].upper(),
                            order_type='MARKET',
                            quantity=order['size']
                        )
                        execution_results.append(result)
                else:
                    from .executor import simulate_trade
                    result = simulate_trade(order)
                    execution_results.append(result)

            return execution_results

        except Exception as e:
            logger.error(f"Error processing target positions: {e}")
            return []

    def _interval_to_seconds(self, interval: str) -> int:
        interval_map = {
            '1m': 60,
            '5m': 300,
            '15m': 900,
            '30m': 1800,
            '1h': 3600,
            '2h': 7200,
            '4h': 14400,
            '6h': 21600,
            '8h': 28800,
            '12h': 43200,
            '1d': 86400,
        }
        return interval_map.get(interval, 3600)


def main():
    config_manager = ConfigManager()
    config = config_manager.load_config('config/neutral_pairs.yaml')

    setup_logging(config)

    order_router = OrderRouter(config)
    order_router.run_trading_loop()


if __name__ == "__main__":
    main()