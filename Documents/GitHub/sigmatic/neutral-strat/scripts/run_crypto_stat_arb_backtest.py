#!/usr/bin/env python3

"""
Crypto Statistical Arbitrage Backtesting Script
Market-neutral statistical arbitrage for cryptocurrency pairs
Target: 2+ Sharpe ratio with controlled drawdown
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import yaml
import argparse
import logging
import time
import ccxt
from datetime import datetime
from typing import Dict, Any

from src.strategies.crypto_stat_arb import run_crypto_stat_arb_backtest
from src.utils.audit_trail import StrategyAuditTrail

def fetch_ohlcv_all(exchange, symbol, timeframe='1d', since=None, limit=1000, sleep=1):
    """Fetch all historical OHLCV data for a symbol"""
    print(f"--- Fetching historical data for {symbol} ({timeframe}) ---")
    all_ohlcv = []

    if since is None:
        since = exchange.parse8601('2022-01-01T00:00:00Z')

    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
            if not ohlcv:
                break

            all_ohlcv += ohlcv
            last_ts = ohlcv[-1][0]
            since = last_ts + 1

            if len(ohlcv) < limit:
                break

            time.sleep(sleep)
        except Exception as e:
            logging.warning(f"Error fetching data for {symbol}: {e}")
            time.sleep(5)
            continue

    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('datetime', inplace=True)
    df = df[['open', 'high', 'low', 'close', 'volume']]

    print(f"→ {len(df)} candles retrieved for {symbol}.")
    return df

def load_crypto_data(config: Dict[str, Any], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
    """Load cryptocurrency data for statistical arbitrage"""

    # Initialize Binance exchange
    exchange = ccxt.binance({
        'apiKey': '',
        'secret': '',
        'sandbox': False,
        'enableRateLimit': True,
    })

    # Extract symbols from config
    symbols = []
    for pair_config in config['pairs']:
        symbols.extend([pair_config['base'], pair_config['alt']])

    symbols = list(set(symbols))  # Remove duplicates

    try:
        # Convert start_date to timestamp
        start_ts = exchange.parse8601(start_date + 'T00:00:00Z')

        # Fetch data for all symbols
        price_data = {}
        volume_data = {}

        for symbol in symbols:
            print(f"Fetching data for {symbol}...")
            df = fetch_ohlcv_all(exchange, symbol, since=start_ts)

            if not df.empty:
                # Filter by date range
                df = df[df.index >= start_date]
                df = df[df.index <= end_date]

                price_data[symbol] = df['close']
                volume_data[symbol] = df['volume']

        # Convert to DataFrames
        prices_df = pd.DataFrame(price_data).dropna(how='all')
        volumes_df = pd.DataFrame(volume_data).dropna(how='all')

        # Align dates
        common_dates = prices_df.index.intersection(volumes_df.index)
        prices_df = prices_df.loc[common_dates]
        volumes_df = volumes_df.loc[common_dates]

        print(f"Loaded data for {len(prices_df.columns)} symbols, {len(prices_df)} days")

        return {
            'prices': prices_df,
            'volumes': volumes_df
        }

    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return {}

def run_stat_arb_backtest(start_date=None, end_date=None,
                         config_path='config/crypto_stat_arb_config.yaml',
                         create_audit_version=True,
                         description="",
                         optimization_notes=""):
    """Run crypto statistical arbitrage backtest"""

    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Use config dates if not provided
    if start_date is None:
        start_date = config['backtest']['start_date']
    if end_date is None:
        end_date = config['backtest']['end_date']

    # Configure logging
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(levelname)s - %(message)s')

    print(f"Running Crypto Statistical Arbitrage Backtest")
    print(f"Period: {start_date} to {end_date}")
    print(f"Strategy: Market-Neutral Statistical Arbitrage")
    print("-" * 60)

    # Load data
    print("Loading cryptocurrency data...")
    data = load_crypto_data(config, start_date, end_date)

    if not data or 'prices' not in data:
        print("ERROR: Unable to load cryptocurrency data")
        return None

    # Use all available symbols from data
    pairs = list(data['prices'].columns) if 'prices' in data else []
    print(f"Available symbols for stat arb: {pairs}")

    # Run backtest
    commission_bps = config['execution']['commission_bps']

    # Check for ensemble mode
    use_ensemble = config.get('strategy', {}).get('ensemble_mode', False)
    use_ml = config.get('strategy', {}).get('ml_enhancement', False)

    if use_ensemble:
        # Run ensemble strategy
        from src.strategies.ensemble_stat_arb import run_ensemble_stat_arb_backtest
        results = run_ensemble_stat_arb_backtest(data, commission_bps)
    else:
        results = run_crypto_stat_arb_backtest(data, pairs, commission_bps, enable_ml_enhancement=use_ml)

    if 'error' in results:
        print(f"ERROR: {results['error']}")
        return None

    # Display results
    print("\n📊 CRYPTO STATISTICAL ARBITRAGE RESULTS")
    print("=" * 60)

    strategy_metrics = results['strategy_metrics']
    best_strategy = results['best_strategy']

    for strategy_name, metrics in strategy_metrics.items():
        print(f"\n🎯 {strategy_name.upper()} STRATEGY:")
        print(f"   Annual Return: {metrics.get('annual_return', 0):.2%}")
        print(f"   Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
        print(f"   Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
        print(f"   Calmar Ratio: {metrics.get('calmar_ratio', 0):.2f}")
        print(f"   Sortino Ratio: {metrics.get('sortino_ratio', 0):.2f}")
        print(f"   Market Beta: {metrics.get('market_beta', 0):.3f}")
        print(f"   Win Rate: {metrics.get('win_rate', 0):.2%}")

        if strategy_name == best_strategy:
            print("   ⭐ BEST STRATEGY ⭐")

    # Performance summary
    best_metrics = strategy_metrics[best_strategy]
    total_return = best_metrics.get('total_return', 0)
    final_value = config['backtest']['initial_capital'] * (1 + total_return)

    print(f"\n💰 PERFORMANCE SUMMARY ({best_strategy}):")
    print(f"   Initial Capital: ${config['backtest']['initial_capital']:,}")
    print(f"   Final Value: ${final_value:,.0f}")
    print(f"   Total Return: {total_return:.2%}")
    print(f"   Best Sharpe: {results['best_sharpe']:.2f}")

    # Target achievement check
    target_sharpe = 2.0
    if results['best_sharpe'] >= target_sharpe:
        print(f"   ✅ TARGET ACHIEVED: Sharpe {results['best_sharpe']:.2f} >= {target_sharpe}")
    else:
        print(f"   ❌ TARGET MISSED: Sharpe {results['best_sharpe']:.2f} < {target_sharpe}")

    # Create audit version if requested
    if create_audit_version:
        try:
            audit_trail = StrategyAuditTrail()

            # Prepare comprehensive results
            audit_results = {
                'start_date': start_date,
                'end_date': end_date,
                'strategy_type': 'crypto_stat_arb',
                'best_strategy': best_strategy,
                'sharpe_ratio': results['best_sharpe'],
                'annual_return': best_metrics.get('annual_return', 0),
                'total_return': total_return,
                'max_drawdown': best_metrics.get('max_drawdown', 0),
                'calmar_ratio': best_metrics.get('calmar_ratio', 0),
                'final_value': final_value,
                'target_achieved': results['best_sharpe'] >= target_sharpe,
                'commission_bps': commission_bps,
                'num_pairs': len(pairs),
                'all_strategies': strategy_metrics
            }

            version_id = audit_trail.create_strategy_version(
                strategy_file='src/strategies/crypto_stat_arb.py',
                config_file=config_path,
                results=audit_results,
                description=description or f"Crypto Statistical Arbitrage - {best_strategy}",
                optimization_notes=optimization_notes or f"Target: 2+ Sharpe, Achieved: {results['best_sharpe']:.2f}"
            )

            print(f"\n📋 Created audit version: {version_id}")

        except Exception as e:
            print(f"Warning: Could not create audit version: {e}")

    return results

def main():
    parser = argparse.ArgumentParser(description='Run Crypto Statistical Arbitrage Backtest')
    parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--config', type=str, default='config/crypto_stat_arb_config.yaml',
                       help='Configuration file path')
    parser.add_argument('--description', type=str, default='',
                       help='Description for audit trail')
    parser.add_argument('--optimization-notes', type=str, default='',
                       help='Optimization notes for audit trail')
    parser.add_argument('--no-audit', action='store_true',
                       help='Skip creating audit trail version')

    args = parser.parse_args()

    # Validate date format
    if args.start_date:
        try:
            datetime.strptime(args.start_date, '%Y-%m-%d')
        except ValueError:
            print("Error: start-date must be in YYYY-MM-DD format")
            return

    if args.end_date:
        try:
            datetime.strptime(args.end_date, '%Y-%m-%d')
        except ValueError:
            print("Error: end-date must be in YYYY-MM-DD format")
            return

    # Run backtest
    run_stat_arb_backtest(
        start_date=args.start_date,
        end_date=args.end_date,
        config_path=args.config,
        create_audit_version=not args.no_audit,
        description=args.description,
        optimization_notes=args.optimization_notes
    )

if __name__ == '__main__':
    main()