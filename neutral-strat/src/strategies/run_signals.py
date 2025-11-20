import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from typing import Dict, List
import time
import logging
from datetime import datetime

from src.utils.config import ConfigManager
from src.utils.logger import setup_logging
from data.data_loader import DataLoader
from data.binance_client import BinanceDataClient
from src.strategies.neutral_pairs import compute_signals, validate_signals


def run_signal_generation():
    config_manager = ConfigManager()
    config = config_manager.load_config('config/neutral_pairs.yaml')

    setup_logging(config)
    logger = logging.getLogger(__name__)

    try:
        binance_client = BinanceDataClient(
            api_key=os.getenv('BINANCE_API_KEY'),
            api_secret=os.getenv('BINANCE_API_SECRET'),
            testnet=config['binance']['testnet']
        )

        data_loader = DataLoader(binance_client)

        symbols = []
        for pair in config['pairs']:
            symbols.extend([pair['base'], pair['hedge']])
        symbols = list(set(symbols))

        logger.info("Starting signal generation loop...")

        while True:
            try:
                current_data = data_loader.get_live_data_slice(
                    symbols=symbols,
                    interval=config['execution']['interval'],
                    lookback_periods=max([p['lookback'] for p in config['pairs']])
                )

                aligned_data = data_loader.align_data(current_data)

                raw_signals = compute_signals(aligned_data, config)
                validated_signals = validate_signals(raw_signals, config)

                logger.info(f"Generated {len(validated_signals)} validated signals at {datetime.now()}")

                for signal in validated_signals:
                    logger.info(
                        f"Signal: {signal.pair_name} | "
                        f"Base: {signal.base_signal.side} {signal.base_signal.size:.4f} {signal.base_signal.symbol} | "
                        f"Hedge: {signal.hedge_signal.side} {signal.hedge_signal.size:.4f} {signal.hedge_signal.symbol} | "
                        f"Z-score: {signal.spread_zscore:.2f} | "
                        f"Reason: {signal.entry_reason}"
                    )

                interval_seconds = _interval_to_seconds(config['execution']['interval'])
                time.sleep(interval_seconds)

            except Exception as e:
                logger.error(f"Error in signal generation loop: {e}")
                time.sleep(30)

    except KeyboardInterrupt:
        logger.info("Signal generation stopped by user")
    except Exception as e:
        logger.error(f"Fatal error in signal generation: {e}")
        raise


def _interval_to_seconds(interval: str) -> int:
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


if __name__ == "__main__":
    run_signal_generation()