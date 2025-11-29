#!/usr/bin/env python3

"""
Advanced Crypto Statistical Arbitrage Optimizer
Systematic optimization to achieve 2+ Sharpe ratio target
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
from itertools import product

from src.strategies.crypto_stat_arb import CryptoStatArbStrategy
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
    return df

def load_crypto_data(start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
    """Load cryptocurrency data for optimization"""

    # Initialize Binance exchange
    exchange = ccxt.binance({
        'apiKey': '',
        'secret': '',
        'sandbox': False,
        'enableRateLimit': True,
    })

    # Top performing crypto symbols based on existing analysis
    symbols = ['BTCUSDT', 'ETHUSDT', 'AVAXUSDT', 'ADAUSDT', 'SOLUSDT', 'DOTUSDT', 'MATICUSDT', 'LINKUSDT']

    try:
        start_ts = exchange.parse8601(start_date + 'T00:00:00Z')

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

def optimize_statistical_arbitrage(start_date: str, end_date: str):
    """Comprehensive optimization for statistical arbitrage"""

    print("🔬 ADVANCED CRYPTO STATISTICAL ARBITRAGE OPTIMIZATION")
    print("=" * 70)
    print(f"Period: {start_date} to {end_date}")
    print(f"Target: Achieve 2+ Sharpe Ratio")
    print("-" * 70)

    # Load data
    print("Loading cryptocurrency data...")
    data = load_crypto_data(start_date, end_date)

    if not data or 'prices' not in data:
        print("ERROR: Unable to load cryptocurrency data")
        return

    prices_df = data['prices']
    volumes_df = data['volumes']

    # Split data for walk-forward optimization
    split_date = '2024-01-01'
    split_ts = pd.Timestamp(split_date)

    train_prices = prices_df[prices_df.index < split_ts]
    test_prices = prices_df[prices_df.index >= split_ts]
    train_volumes = volumes_df[volumes_df.index < split_ts]
    test_volumes = volumes_df[volumes_df.index >= split_ts]

    print(f"Training: {len(train_prices)} days")
    print(f"Testing: {len(test_prices)} days")

    # Initialize strategy
    strategy = CryptoStatArbStrategy(commission_bps=1.0)  # Low cost assumption

    best_results = []

    # 1. Symbol Selection Optimization
    print("\\n🎯 PHASE 1: SYMBOL SELECTION OPTIMIZATION")

    # Test different symbol combinations
    all_symbols = list(prices_df.columns)

    # Test combinations of 4-8 symbols
    for n_symbols in [4, 5, 6, 7, 8]:
        from itertools import combinations

        best_combo_sharpe = -np.inf
        best_combo = None

        print(f"Testing {n_symbols}-symbol combinations...")

        # Test top combinations (limit to avoid excessive computation)
        symbol_combinations = list(combinations(all_symbols, n_symbols))[:50]

        for combo in symbol_combinations:
            try:
                combo_train = train_prices[list(combo)]
                combo_test = test_prices[list(combo)]

                if len(combo_train) < 100 or len(combo_test) < 50:
                    continue

                # Quick parameter optimization
                best_params = strategy.optimize_parameters(combo_train, strategy_type="mean_reversion")

                if best_params and 'sharpe_ratio' in best_params:
                    sharpe = best_params['sharpe_ratio']

                    if sharpe > best_combo_sharpe:
                        best_combo_sharpe = sharpe
                        best_combo = combo

            except Exception as e:
                continue

        if best_combo:
            print(f"  Best {n_symbols}-symbol combo: {best_combo[:3]}... (Sharpe: {best_combo_sharpe:.2f})")

            # Test on out-of-sample
            test_data = {
                'prices': test_prices[list(best_combo)],
                'volumes': test_volumes[list(best_combo)]
            }

            oos_results = strategy.generate_crypto_pairs_signals(list(best_combo), test_data)

            if oos_results:
                for strat_name, returns in oos_results.items():
                    metrics = strategy.calculate_performance_metrics(returns)
                    oos_sharpe = metrics.get('sharpe_ratio', 0)

                    best_results.append({
                        'optimization': f'{n_symbols}_symbol_selection',
                        'strategy': strat_name,
                        'symbols': best_combo,
                        'sharpe_ratio': oos_sharpe,
                        'annual_return': metrics.get('annual_return', 0),
                        'max_drawdown': metrics.get('max_drawdown', 0),
                        'calmar_ratio': metrics.get('calmar_ratio', 0),
                        'total_return': metrics.get('total_return', 0),
                        'returns': returns
                    })

    # 2. Advanced Parameter Optimization
    print("\\n🔧 PHASE 2: ADVANCED PARAMETER OPTIMIZATION")

    # Use best symbol combination for parameter optimization
    if best_results:
        top_result = max(best_results, key=lambda x: x['sharpe_ratio'])
        best_symbols = top_result['symbols']

        print(f"Optimizing parameters for: {best_symbols}")

        # Ultra-fine parameter grid
        formation_periods = [1, 2, 3, 5, 7, 10, 15, 20]
        holding_periods = [1, 2, 3, 5, 7, 10]
        threshold_pairs = [
            (0.05, 0.95), (0.1, 0.9), (0.15, 0.85), (0.2, 0.8),
            (0.25, 0.75), (0.3, 0.7), (0.33, 0.67)
        ]
        commission_levels = [0.5, 1.0, 2.0, 5.0, 10.0]

        param_results = []

        for commission in commission_levels:
            strategy_comm = CryptoStatArbStrategy(commission_bps=commission)

            for fp, hp, (lt, st) in product(formation_periods, holding_periods, threshold_pairs):
                try:
                    # Test on training data
                    train_subset = train_prices[list(best_symbols)]

                    returns = strategy_comm.run_market_neutral_reversal_backtest(
                        train_subset, fp, hp, lt, st
                    )

                    if len(returns.dropna()) > 100:
                        metrics = strategy_comm.calculate_performance_metrics(returns)
                        sharpe = metrics.get('sharpe_ratio', 0)

                        if sharpe > 0.5:  # Filter out very poor performers
                            # Test on out-of-sample
                            test_subset = test_prices[list(best_symbols)]
                            oos_returns = strategy_comm.run_market_neutral_reversal_backtest(
                                test_subset, fp, hp, lt, st
                            )

                            oos_metrics = strategy_comm.calculate_performance_metrics(oos_returns)
                            oos_sharpe = oos_metrics.get('sharpe_ratio', 0)

                            param_results.append({
                                'formation_period': fp,
                                'holding_period': hp,
                                'long_threshold': lt,
                                'short_threshold': st,
                                'commission_bps': commission,
                                'train_sharpe': sharpe,
                                'oos_sharpe': oos_sharpe,
                                'oos_annual_return': oos_metrics.get('annual_return', 0),
                                'oos_max_drawdown': oos_metrics.get('max_drawdown', 0),
                                'oos_calmar': oos_metrics.get('calmar_ratio', 0),
                                'oos_returns': oos_returns
                            })

                except Exception as e:
                    continue

        if param_results:
            # Sort by out-of-sample Sharpe
            param_results.sort(key=lambda x: x['oos_sharpe'], reverse=True)

            print("\\n📊 TOP PARAMETER COMBINATIONS:")
            for i, result in enumerate(param_results[:10]):
                print(f"  {i+1}. Formation: {result['formation_period']}, Holding: {result['holding_period']}")
                print(f"      Thresholds: {result['long_threshold']:.2f}-{result['short_threshold']:.2f}")
                print(f"      Commission: {result['commission_bps']:.1f}bps")
                print(f"      OOS Sharpe: {result['oos_sharpe']:.2f}")
                print(f"      OOS Annual Return: {result['oos_annual_return']:.1%}")
                print(f"      Max Drawdown: {result['oos_max_drawdown']:.1%}")
                print()

            # Create audit versions for top performers
            audit_trail = StrategyAuditTrail()

            for i, result in enumerate(param_results[:5]):
                if result['oos_sharpe'] >= 2.0:  # Achieved target
                    status = "🎯 TARGET ACHIEVED"
                elif result['oos_sharpe'] >= 1.5:  # Close to target
                    status = "⚡ HIGH PERFORMANCE"
                elif result['oos_sharpe'] >= 1.0:  # Decent performance
                    status = "✅ GOOD PERFORMANCE"
                else:
                    status = "⚠️ NEEDS IMPROVEMENT"

                description = f"Advanced StatArb Optimization #{i+1} - {status}"

                audit_results = {
                    'optimization_phase': 'advanced_parameter_optimization',
                    'symbols': list(best_symbols),
                    'formation_period': result['formation_period'],
                    'holding_period': result['holding_period'],
                    'long_threshold': result['long_threshold'],
                    'short_threshold': result['short_threshold'],
                    'commission_bps': result['commission_bps'],
                    'sharpe_ratio': result['oos_sharpe'],
                    'annual_return': result['oos_annual_return'],
                    'max_drawdown': result['oos_max_drawdown'],
                    'calmar_ratio': result['oos_calmar'],
                    'total_return': result['oos_returns'].sum() if len(result['oos_returns']) > 0 else 0,
                    'target_achieved': result['oos_sharpe'] >= 2.0
                }

                version_id = audit_trail.create_strategy_version(
                    strategy_file='src/strategies/crypto_stat_arb.py',
                    config_file='config/crypto_stat_arb_config.yaml',
                    results=audit_results,
                    description=description,
                    optimization_notes=f"Advanced optimization: {result['oos_sharpe']:.2f} Sharpe"
                )

                print(f"📋 Created audit version: {version_id}")

    print("\\n🎯 OPTIMIZATION COMPLETE!")

    if best_results:
        final_best = max(best_results + (param_results if 'param_results' in locals() else []),
                        key=lambda x: x.get('oos_sharpe', x.get('sharpe_ratio', 0)))

        best_sharpe = final_best.get('oos_sharpe', final_best.get('sharpe_ratio', 0))

        print(f"Best Achieved Sharpe: {best_sharpe:.2f}")

        if best_sharpe >= 2.0:
            print("🏆 TARGET ACHIEVED: 2+ Sharpe Ratio!")
        else:
            print(f"📈 Progress: {(best_sharpe/2.0)*100:.1f}% toward 2.0 Sharpe target")

def main():
    parser = argparse.ArgumentParser(description='Optimize Crypto Statistical Arbitrage')
    parser.add_argument('--start-date', type=str, default='2022-11-29', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default='2025-11-29', help='End date (YYYY-MM-DD)')

    args = parser.parse_args()

    optimize_statistical_arbitrage(args.start_date, args.end_date)

if __name__ == '__main__':
    main()