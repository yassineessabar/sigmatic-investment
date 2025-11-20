#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from datetime import datetime
import logging

from src.utils.config import ConfigManager
from src.utils.logger import setup_logging
from src.backtest.engine import BacktestEngine
from src.strategies.neutral_pairs import compute_signals
from data.data_loader import DataLoader
from data.binance_client import BinanceDataClient


def run_backtest_script():
    parser = argparse.ArgumentParser(description='Run backtest for neutral pairs strategy')
    parser.add_argument('--config', default='config/neutral_pairs.yaml',
                       help='Path to configuration file')
    parser.add_argument('--start-date', type=str, required=True,
                       help='Start date for backtest (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, required=True,
                       help='End date for backtest (YYYY-MM-DD)')
    parser.add_argument('--output', type=str, default='results/backtest_results.json',
                       help='Output file for results')

    args = parser.parse_args()

    config_manager = ConfigManager()
    config = config_manager.load_config(args.config)

    setup_logging(config)
    logger = logging.getLogger(__name__)

    start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
    end_date = datetime.strptime(args.end_date, '%Y-%m-%d')

    logger.info(f"Running backtest from {start_date} to {end_date}")

    try:
        # Use mock client for backtesting (no real API calls needed for historical data)
        binance_client = BinanceDataClient(
            api_key="mock_key",
            api_secret="mock_secret",
            testnet=True
        )

        data_loader = DataLoader(binance_client)

        symbols = []
        for pair in config['pairs']:
            symbols.extend([pair['base'], pair['hedge']])
        symbols = list(set(symbols))

        engine = BacktestEngine(config)
        results = engine.run_backtest(
            data_loader=data_loader,
            strategy_fn=compute_signals,
            start_date=start_date,
            end_date=end_date,
            symbols=symbols
        )

        # Create output directory
        os.makedirs(os.path.dirname(args.output), exist_ok=True)

        # Save results
        results_dict = {
            'total_return': results.total_return,
            'sharpe_ratio': results.sharpe_ratio,
            'max_drawdown': results.max_drawdown,
            'win_rate': results.win_rate,
            'profit_factor': results.profit_factor,
            'total_trades': results.total_trades,
            'start_date': start_date.isoformat(),
            'end_date': end_date.isoformat(),
            'config': config
        }

        import json
        with open(args.output, 'w') as f:
            json.dump(results_dict, f, indent=2, default=str)

        logger.info(f"Backtest Results:")
        logger.info(f"Total Return: {results.total_return:.2%}")
        logger.info(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
        logger.info(f"Max Drawdown: {results.max_drawdown:.2%}")
        logger.info(f"Win Rate: {results.win_rate:.2%}")
        logger.info(f"Total Trades: {results.total_trades}")
        logger.info(f"Results saved to: {args.output}")

    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        raise


if __name__ == "__main__":
    run_backtest_script()