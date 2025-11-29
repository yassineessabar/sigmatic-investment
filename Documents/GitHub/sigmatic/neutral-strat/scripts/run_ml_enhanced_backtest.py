#!/usr/bin/env python3

"""
ML-Enhanced Relative Momentum Backtest Runner
Tests machine learning enhanced strategy versions
"""

import sys
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ccxt
import argparse
from datetime import datetime
from typing import Dict, List
import logging
from pathlib import Path

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import ConfigManager
from src.utils.logger import setup_logging
from src.utils.audit_trail import StrategyAuditTrail
from src.strategies.ml_enhanced_momentum import (
    optimize_ml_enhanced_strategy,
    backtest_ml_enhanced_momentum,
    compute_ml_metrics
)
from src.strategies.relative_momentum import compute_metrics
from src.utils.backtest_utils import (
    print_performance_comparison,
    calculate_benchmark_performance
)

# Import data fetching functions from original script
from scripts.run_relative_momentum_backtest import fetch_ohlcv_all, fetch_funding_rates, plot_portfolio_performance

plt.style.use("seaborn-v0_8-darkgrid")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_ml_enhanced_backtest(start_date=None, end_date=None,
                           config_path='config/unified_trading_config.yaml',
                           create_audit_version=True, description="",
                           optimization_notes="", ml_strategy_type="enhanced"):
    """Run ML-enhanced relative momentum backtest"""

    # Initialize audit trail
    audit_trail = StrategyAuditTrail() if create_audit_version else None

    # Load configuration
    config_manager = ConfigManager()
    config = config_manager.load_config(config_path)

    setup_logging(config.get('logging', {}))

    # Initialize exchange
    exchange = ccxt.binance({
        'enableRateLimit': True,
        'sandbox': False,
        'options': {'defaultType': 'future'}
    })

    # Extract symbols
    universe = []
    for pair in config['pairs']:
        if pair['alt'] not in universe:
            universe.append(pair['alt'])

    if 'BTCUSDT' not in universe:
        universe = ['BTCUSDT'] + universe

    print(f"Universe: {universe}")

    # Use provided dates or fall back to config
    backtest_start_date = start_date or config['backtest']['start_date']
    backtest_end_date = end_date or datetime.now().strftime('%Y-%m-%d')

    print(f"ML-Enhanced Backtest period: {backtest_start_date} to {backtest_end_date}")

    # Convert to datetime for filtering
    start_dt = pd.to_datetime(backtest_start_date)
    end_dt = pd.to_datetime(backtest_end_date)

    # Fetch data (reuse from original script)
    data = {}
    funding_data = {}
    os.makedirs("data/historical/daily", exist_ok=True)

    for symbol in universe:
        futures_symbol = symbol.replace('USDT', '/USDT:USDT') if 'USDT' in symbol and ':USDT' not in symbol else symbol

        price_fname = f"data/historical/daily/{symbol.replace('/', '').replace(':', '')}_daily_futures.csv"
        funding_fname = f"data/historical/daily/{symbol.replace('/', '').replace(':', '')}_funding_rates.csv"

        print(f"🔄 Refreshing ML data for {futures_symbol}...")

        try:
            # Check if we already have recent data
            if os.path.exists(price_fname):
                existing_df = pd.read_csv(price_fname, index_col=0, parse_dates=True)
                if len(existing_df) > 0 and existing_df.index[-1].date() >= (pd.Timestamp.now() - pd.Timedelta(days=1)).date():
                    print(f"→ Using cached data for {futures_symbol}")
                    df_filtered = existing_df.loc[start_dt:end_dt]
                    data[symbol] = df_filtered

                    if os.path.exists(funding_fname):
                        funding_df = pd.read_csv(funding_fname, index_col=0, parse_dates=True)
                        funding_data[symbol] = funding_df
                    continue

            # Fetch fresh data
            df = fetch_ohlcv_all(
                exchange, futures_symbol, timeframe='1d',
                since=exchange.parse8601(backtest_start_date + 'T00:00:00Z')
            )
            df.to_csv(price_fname)

            df_filtered = df.loc[start_dt:end_dt]
            if len(df_filtered) == 0:
                logger.warning(f"No price data for {futures_symbol}")
                continue

            data[symbol] = df_filtered

            # Fetch funding rates
            funding_df = fetch_funding_rates(exchange, symbol, backtest_start_date, backtest_end_date)
            if not funding_df.empty:
                funding_df.to_csv(funding_fname)
                funding_data[symbol] = funding_df

            print(f"→ {len(df_filtered)} candles + {len(funding_df) if not funding_df.empty else 0} funding rates")

        except Exception as e:
            logger.error(f"Failed to fetch data for {futures_symbol}: {e}")
            continue

    freq = config['backtest']['freq']
    pair_returns = {}
    summary_rows = []

    print(f"\n🤖 Running {ml_strategy_type.upper()} ML Strategy...")

    # Run ML-enhanced backtests
    for pair_config in config['pairs']:
        base_symbol = pair_config['base']
        alt_symbol = pair_config['alt']

        if base_symbol not in data or alt_symbol not in data:
            logger.warning(f"Missing data for {base_symbol}/{alt_symbol}")
            continue

        pair_name = f"{base_symbol}/{alt_symbol.replace('USDT', '')}"

        try:
            base_funding_data = funding_data.get(base_symbol, pd.DataFrame())
            alt_funding_data = funding_data.get(alt_symbol, pd.DataFrame())

            optimization_metric = config['strategy']['optimization'].get('metric', 'sharpe')

            if config['strategy']['optimization']['enabled']:
                window_start = config['strategy']['optimization']['window_range_start']
                window_end = config['strategy']['optimization']['window_range_end']

                best_result = optimize_ml_enhanced_strategy(
                    data[base_symbol], data[alt_symbol],
                    list(range(window_start, window_end)),
                    pair_config['allocation_weight'],
                    config['execution']['fees'],
                    config['execution']['slippage'],
                    freq, base_funding_data, alt_funding_data,
                    optimization_metric
                )
            else:
                # Use configured EMA window with ML enhancement
                best_result = backtest_ml_enhanced_momentum(
                    data[base_symbol], data[alt_symbol],
                    pair_config['ema_window'],
                    pair_config['allocation_weight'],
                    config['execution']['fees'],
                    config['execution']['slippage'],
                    freq, base_funding_data, alt_funding_data,
                    ml_weight=0.3
                )

            # Print results
            stats = print_ml_stats(pair_name, best_result, freq)
            summary_rows.append(stats)
            pair_returns[pair_name] = best_result['returns']

        except Exception as e:
            logger.error(f"Failed to backtest ML {pair_name}: {e}")
            continue

    if not pair_returns:
        logger.error("No successful ML backtests completed")
        return

    # Portfolio analysis
    df_all = pd.DataFrame(pair_returns).dropna()

    if len(df_all.columns) == 0:
        logger.error("No valid returns data for portfolio analysis")
        return

    # Calculate benchmark
    benchmark_returns = calculate_benchmark_performance(data, start_dt, end_dt)
    if len(benchmark_returns) > 0:
        benchmark_returns = benchmark_returns.reindex(df_all.index).fillna(0)

    # Portfolio calculations
    equal_w = np.ones(df_all.shape[1]) / df_all.shape[1]
    equal_portfolio_returns = (df_all * equal_w).sum(axis=1)

    vols = df_all.std()
    if (vols > 0).all():
        inv_vol_w = (1 / vols) / np.sum(1 / vols)
        vol_scaled_portfolio_returns = (df_all * inv_vol_w).sum(axis=1)
    else:
        vol_scaled_portfolio_returns = equal_portfolio_returns

    def calculate_portfolio_metrics(returns, name):
        ann_ret, ann_vol, sharpe, max_dd = compute_metrics(returns, freq)
        calmar = ann_ret / abs(max_dd) if max_dd != 0 else np.nan
        win_rate = (returns > 0).mean()
        final_perf = (1 + returns).cumprod().iloc[-1]

        return {
            'Pair': name,
            'EMA_Window': 'ML-Enhanced',
            'Ann.Return': ann_ret,
            'Vol': ann_vol,
            'Sharpe': sharpe,
            'Calmar': calmar,
            'Max DD': max_dd,
            'Win Rate': win_rate,
            'Final_Perf': final_perf
        }

    eq_stats = calculate_portfolio_metrics(equal_portfolio_returns, f"Equal-Weight ML-{ml_strategy_type}")
    vs_stats = calculate_portfolio_metrics(vol_scaled_portfolio_returns, f"Vol-Scaled ML-{ml_strategy_type}")

    summary_rows.extend([eq_stats, vs_stats])

    # Save results
    results_table = pd.DataFrame(summary_rows)
    os.makedirs("results", exist_ok=True)
    results_table.to_csv(f"results/ml_{ml_strategy_type}_results.csv", index=False)

    print(f"\n======= ML-{ml_strategy_type.upper()} STRATEGY PERFORMANCE =======")
    print(results_table.to_string(index=False))
    print(f"\n✅ Results saved to results/ml_{ml_strategy_type}_results.csv")

    # Create audit trail version
    if audit_trail and create_audit_version:
        try:
            backtest_results = {
                'results_table': results_table,
                'equal_portfolio_returns': equal_portfolio_returns,
                'vol_scaled_portfolio_returns': vol_scaled_portfolio_returns,
                'benchmark_returns': benchmark_returns if 'benchmark_returns' in locals() else pd.Series(),
                'config': config,
                'start_date': backtest_start_date,
                'end_date': backtest_end_date,
                'ml_strategy_type': ml_strategy_type
            }

            strategy_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                       'src', 'strategies', 'ml_enhanced_momentum.py')

            version_id = audit_trail.create_strategy_version(
                strategy_file=strategy_file,
                config_file=config_path,
                results=backtest_results,
                description=description or f"ML-{ml_strategy_type} {backtest_start_date} to {backtest_end_date}",
                optimization_notes=optimization_notes or f"Machine learning enhanced strategy with {ml_strategy_type} approach"
            )

            print(f"\n📋 Created ML audit trail version: {version_id}")

        except Exception as e:
            logger.warning(f"Failed to create ML audit trail: {e}")

    # Performance comparison
    if len(benchmark_returns) > 0:
        initial_capital = config.get('backtest', {}).get('initial_capital', 10000)

        print_performance_comparison(
            equal_portfolio_returns, benchmark_returns,
            f"ML-{ml_strategy_type} Equal-Weight", "BTC Buy & Hold", freq,
            initial_capital, None
        )

    # Plot results
    print(f"\nML Backtest Date Range: {df_all.index.min().strftime('%Y-%m-%d')} → {df_all.index.max().strftime('%Y-%m-%d')}")

    plot_portfolio_performance(
        df_all, equal_portfolio_returns, vol_scaled_portfolio_returns,
        benchmark_returns if len(benchmark_returns) > 0 else None,
        config.get('backtest', {}).get('initial_capital', 10000)
    )

    return results_table


def print_ml_stats(pair, result, freq=365):
    """Print ML strategy statistics"""

    ann_return = result.get('ann_return', 0)
    ann_vol = result.get('ann_vol', 0)
    sharpe = result.get('sharpe', 0)
    max_dd = result.get('max_dd', 0)
    ml_weight = result.get('ml_weight', 0.3)

    returns = result['returns']
    neg_returns = returns[returns < 0] if len(returns) > 0 else pd.Series()
    downside_std = max(np.std(neg_returns, ddof=1), 0.01) if len(neg_returns) > 0 else 0.01
    sortino = np.clip((ann_return / downside_std) * np.sqrt(freq), -10, 10)
    calmar = np.clip(ann_return / abs(max_dd) if max_dd != 0 else np.nan, -10, 10)
    win_rate = (returns > 0).mean() if len(returns) > 0 else 0

    print(f"\n========== {pair} (ML-Enhanced Strategy) ==========")
    print(f"Best EMA window: {result['ema_window']}d | ML Weight: {ml_weight:.1f}")
    print(f"Sharpe: {sharpe:.2f} | Sortino: {sortino:.2f} | Calmar: {calmar:.2f}")
    print(f"Ann.Return: {ann_return:.2%} | Ann.Vol: {ann_vol:.2%} | Max DD: {max_dd:.2%}")
    print(f"Win Rate: {win_rate:.2%} | Final Performance: {result['final_performance']:.2f}x")

    return {
        'Pair': pair,
        'EMA_Window': result['ema_window'],
        'Ann.Return': ann_return,
        'Vol': ann_vol,
        'Sharpe': sharpe,
        'Calmar': calmar,
        'Max DD': max_dd,
        'Win Rate': win_rate,
        'Final_Perf': result['final_performance'],
        'ML_Weight': ml_weight
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run ML-enhanced relative momentum strategy backtest')
    parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)', default=None)
    parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)', default=None)
    parser.add_argument('--config', type=str, help='Config file path', default='config/unified_trading_config.yaml')
    parser.add_argument('--description', type=str, help='Audit trail description', default='')
    parser.add_argument('--optimization-notes', type=str, help='Technical notes', default='')
    parser.add_argument('--no-audit', action='store_true', help='Skip audit trail', default=False)
    parser.add_argument('--ml-type', type=str, choices=['enhanced', 'ensemble', 'adaptive'],
                       help='ML strategy type', default='enhanced')

    args = parser.parse_args()

    # Validate dates
    if args.start_date:
        try:
            datetime.strptime(args.start_date, '%Y-%m-%d')
        except ValueError:
            print("Error: start-date must be in YYYY-MM-DD format")
            sys.exit(1)

    if args.end_date:
        try:
            datetime.strptime(args.end_date, '%Y-%m-%d')
        except ValueError:
            print("Error: end-date must be in YYYY-MM-DD format")
            sys.exit(1)

    run_ml_enhanced_backtest(
        start_date=args.start_date,
        end_date=args.end_date,
        config_path=args.config,
        create_audit_version=not args.no_audit,
        description=args.description,
        optimization_notes=args.optimization_notes,
        ml_strategy_type=args.ml_type
    )