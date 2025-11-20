#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, List
import time
import logging
from datetime import datetime

from src.utils.config import ConfigManager
from src.utils.logger import setup_logging
from data.data_loader import DataLoader
from data.binance_client import BinanceDataClient
from src.strategies.relative_momentum import compute_relative_momentum_signals
from src.strategies.neutral_pairs import validate_signals


def run_relative_momentum_signal_generation():
    """Run live signal generation for relative momentum strategy"""

    config_manager = ConfigManager()
    config = config_manager.load_config('config/relative_momentum.yaml')

    setup_logging(config)
    logger = logging.getLogger(__name__)

    try:
        # Initialize Binance client (uses testnet by default)
        binance_client = BinanceDataClient(
            api_key=os.getenv('BINANCE_API_KEY', 'test_key'),
            api_secret=os.getenv('BINANCE_API_SECRET', 'test_secret'),
            testnet=config['binance']['testnet']
        )

        data_loader = DataLoader(binance_client)

        # Extract symbols from pairs config
        symbols = []
        for pair in config['pairs']:
            symbols.extend([pair['base'], pair['alt']])
        symbols = list(set(symbols))

        logger.info("Starting relative momentum signal generation loop...")
        logger.info(f"Monitoring pairs: {[f\"{p['base']}/{p['alt']}\" for p in config['pairs']]}")
        logger.info(f"Trading mode: {config['execution']['mode']}")

        while True:
            try:
                # Get current market data
                current_data = data_loader.get_live_data_slice(
                    symbols=symbols,
                    interval=config['execution']['interval'],
                    lookback_periods=max([p.get('ema_window', 20) for p in config['pairs']]) + 10
                )

                aligned_data = data_loader.align_data(current_data)

                # Generate signals using relative momentum strategy
                raw_signals = compute_relative_momentum_signals(aligned_data, config)
                validated_signals = validate_signals(raw_signals, config)

                logger.info(f"Generated {len(validated_signals)} validated signals at {datetime.now()}")

                # Log signal details
                for signal in validated_signals:
                    logger.info(
                        f"Signal: {signal.pair_name} | "
                        f"Base: {signal.base_signal.side} {signal.base_signal.size:.4f} {signal.base_signal.symbol} @ {signal.base_signal.price:.2f} | "
                        f"Alt: {signal.hedge_signal.side} {signal.hedge_signal.size:.4f} {signal.hedge_signal.symbol} @ {signal.hedge_signal.price:.2f} | "
                        f"Confidence: {signal.base_signal.confidence:.2f} | "
                        f"Reason: {signal.entry_reason}"
                    )

                    # Log metadata if available
                    if signal.base_signal.metadata:
                        metadata = signal.base_signal.metadata
                        logger.info(
                            f"  → Relative ratio: {metadata.get('relative_ratio', 'N/A'):.6f} | "
                            f"EMA ratio: {metadata.get('ema_ratio', 'N/A'):.6f} | "
                            f"Weight: {metadata.get('weight', 'N/A'):.2f}"
                        )

                # In paper mode, just log the signals
                if config['execution']['mode'] == 'paper':
                    if validated_signals:
                        logger.info("📊 Paper mode: Signals logged above (no actual trades executed)")
                    else:
                        logger.info("📊 Paper mode: No signals generated")

                # Wait for next interval
                interval_seconds = _interval_to_seconds(config['execution']['interval'])
                logger.info(f"⏰ Waiting {interval_seconds}s until next signal generation...")
                time.sleep(interval_seconds)

            except Exception as e:
                logger.error(f"Error in signal generation loop: {e}")
                logger.info("⏰ Retrying in 30 seconds...")
                time.sleep(30)

    except KeyboardInterrupt:
        logger.info("Signal generation stopped by user")
    except Exception as e:
        logger.error(f"Fatal error in signal generation: {e}")
        raise


def _interval_to_seconds(interval: str) -> int:
    """Convert interval string to seconds"""
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
    run_relative_momentum_signal_generation()