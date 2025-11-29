#!/usr/bin/env python3

"""
Unified Relative Momentum Backtest Runner
Runs historical backtests with identical strategy parameters as live trading.
"""

import sys
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
from src.strategies.relative_momentum import (
    compute_relative_momentum_signals,
    backtest_relative_momentum_pair,
    optimize_ema_window,
    compute_metrics
)
from src.utils.backtest_utils import (
    print_performance_comparison,
    calculate_benchmark_performance,
    save_backtest_results
)

plt.style.use("seaborn-v0_8-darkgrid")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fetch_ohlcv_all(exchange, symbol, timeframe='1d', since=None, limit=1000, sleep=1):
    """Fetch all historical OHLCV data for a symbol"""
    print(f"--- Fetching historical data for {symbol} ({timeframe}) ---")
    all_ohlcv = []

    if since is None:
        since = exchange.parse8601('2023-10-01T00:00:00Z')

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
            logger.warning(f"Error fetching data for {symbol}: {e}")
            time.sleep(5)
            continue

    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('datetime', inplace=True)
    df = df[['open', 'high', 'low', 'close', 'volume']]

    print(f"→ {len(df)} candles retrieved for {symbol}.")
    return df


def fetch_funding_rates(exchange, symbol, start_date, end_date):
    """Fetch funding rate history for a futures symbol"""
    print(f"--- Fetching funding rates for {symbol} ---")

    # Convert to futures symbol format for funding rates
    if 'USDT' in symbol and ':USDT' not in symbol:
        futures_symbol = symbol.replace('USDT', '/USDT:USDT')
    else:
        futures_symbol = symbol

    try:
        # Binance funding rates are collected every 8 hours
        since_ts = exchange.parse8601(start_date + 'T00:00:00Z')
        end_ts = exchange.parse8601(end_date + 'T23:59:59Z')

        all_funding = []
        current_since = since_ts

        while current_since < end_ts:
            try:
                funding_history = exchange.fetch_funding_rate_history(
                    futures_symbol,
                    since=current_since,
                    limit=500
                )

                if not funding_history:
                    break

                all_funding.extend(funding_history)
                current_since = funding_history[-1]['timestamp'] + 1

                if len(funding_history) < 500:
                    break

                time.sleep(1)  # Rate limiting

            except Exception as e:
                logger.warning(f"Error fetching funding rates batch for {futures_symbol}: {e}")
                break

        if not all_funding:
            logger.warning(f"No funding rate data found for {futures_symbol}")
            return pd.DataFrame()

        # Convert to DataFrame
        funding_df = pd.DataFrame(all_funding)
        funding_df['datetime'] = pd.to_datetime(funding_df['timestamp'], unit='ms')
        funding_df.set_index('datetime', inplace=True)
        funding_df = funding_df[['fundingRate']].rename(columns={'fundingRate': 'funding_rate'})

        # Filter to date range
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        funding_df = funding_df.loc[start_dt:end_dt]

        print(f"→ {len(funding_df)} funding rate records for {futures_symbol}")
        return funding_df

    except Exception as e:
        logger.error(f"Failed to fetch funding rates for {futures_symbol}: {e}")
        return pd.DataFrame()


def print_stats(pair, result, freq=365):
    """Print detailed statistics for a strategy result"""
    ann_return = result['ann_return']
    ann_vol = result['ann_vol']
    sharpe = result['sharpe']
    max_dd = result['max_dd']

    returns = result['returns']
    neg_returns = returns[returns < 0]
    downside_std = max(np.std(neg_returns, ddof=1), 0.01) if len(neg_returns) > 0 else 0.01
    sortino = np.clip((ann_return / downside_std) * np.sqrt(freq), -10, 10)
    calmar = np.clip(ann_return / abs(max_dd) if max_dd != 0 else np.nan, -10, 10)
    win_rate = (returns > 0).mean()

    print(f"\n========== {pair} ==========")
    print(f"Best EMA window: {result['ema_window']}d | Sharpe: {sharpe:.2f} | Sortino: {sortino:.2f}")
    print(f"Ann.Return: {ann_return:.2%} | Ann.Vol: {ann_vol:.2%} | Max DD: {max_dd:.2%}")
    print(f"Calmar: {calmar:.2f} | Win Rate: {win_rate:.2%}")
    print(f"Final Performance: {result['final_performance']:.2f}x")

    return {
        'Pair': pair,
        'EMA_Window': result['ema_window'],
        'Ann.Return': ann_return,
        'Vol': ann_vol,
        'Sharpe': sharpe,
        'Calmar': calmar,
        'Max DD': max_dd,
        'Win Rate': win_rate,
        'Final_Perf': result['final_performance']
    }


def run_relative_momentum_backtest(start_date=None, end_date=None, config_path='config/unified_trading_config.yaml'):
    """Run the complete relative momentum backtest with unified parameters"""

    # Load configuration
    config_manager = ConfigManager()
    config = config_manager.load_config(config_path)

    setup_logging(config.get('logging', {}))

    # Initialize exchange for futures
    exchange = ccxt.binance({
        'enableRateLimit': True,
        'sandbox': False,
        'options': {
            'defaultType': 'future'
        }
    })

    # Extract symbols from config
    universe = []
    for pair in config['pairs']:
        if pair['alt'] not in universe:
            universe.append(pair['alt'])

    # Add BTC if not present
    if 'BTCUSDT' not in universe:
        universe = ['BTCUSDT'] + universe

    print(f"Universe: {universe}")

    # Use provided dates or fall back to config
    backtest_start_date = start_date or config['backtest']['start_date']
    backtest_end_date = end_date or datetime.now().strftime('%Y-%m-%d')

    print(f"Backtest period: {backtest_start_date} to {backtest_end_date}")

    # Convert to datetime objects for filtering
    start_dt = pd.to_datetime(backtest_start_date)
    end_dt = pd.to_datetime(backtest_end_date)

    # Fetch all data (price + funding rates)
    data = {}
    funding_data = {}
    os.makedirs("data/historical/daily", exist_ok=True)

    for symbol in universe:
        # Convert to futures symbol format
        if 'USDT' in symbol and ':USDT' not in symbol:
            futures_symbol = symbol.replace('USDT', '/USDT:USDT')
        else:
            futures_symbol = symbol

        # Fetch price data
        price_fname = f"data/historical/daily/{symbol.replace('/', '').replace(':', '')}_daily_futures.csv"
        funding_fname = f"data/historical/daily/{symbol.replace('/', '').replace(':', '')}_funding_rates.csv"

        print(f"🔄 Refreshing futures history for {futures_symbol}...")
        try:
            # Fetch OHLCV data
            df = fetch_ohlcv_all(
                exchange,
                futures_symbol,
                timeframe='1d',
                since=exchange.parse8601(backtest_start_date + 'T00:00:00Z')
            )
            df.to_csv(price_fname)

            # Filter data to requested date range
            df_filtered = df.loc[start_dt:end_dt]
            if len(df_filtered) == 0:
                logger.warning(f"No price data for {futures_symbol} in date range {backtest_start_date} to {backtest_end_date}")
                continue

            data[symbol] = df_filtered

            # Fetch funding rates
            funding_df = fetch_funding_rates(exchange, symbol, backtest_start_date, backtest_end_date)
            if not funding_df.empty:
                funding_df.to_csv(funding_fname)
                funding_data[symbol] = funding_df

            print(f"→ {len(df_filtered)} futures candles + {len(funding_df) if not funding_df.empty else 0} funding rates for {futures_symbol}")

        except Exception as e:
            logger.error(f"Failed to fetch futures data for {futures_symbol}: {e}")
            continue

    freq = config['backtest']['freq']
    pair_returns = {}
    summary_rows = []

    # Run backtests for each pair
    for pair_config in config['pairs']:
        base_symbol = pair_config['base']
        alt_symbol = pair_config['alt']

        if base_symbol not in data or alt_symbol not in data:
            logger.warning(f"Missing data for {base_symbol}/{alt_symbol}")
            continue

        pair_name = f"{base_symbol}/{alt_symbol.replace('USDT', '')}"

        try:
            # Get funding data for this pair
            base_funding_data = funding_data.get(base_symbol, pd.DataFrame())
            alt_funding_data = funding_data.get(alt_symbol, pd.DataFrame())

            if config['strategy']['optimization']['enabled']:
                # Optimize EMA window
                window_start = config['strategy']['optimization']['window_range_start']
                window_end = config['strategy']['optimization']['window_range_end']

                best_result = optimize_ema_window(
                    data[base_symbol],
                    data[alt_symbol],
                    range(window_start, window_end),
                    pair_config['allocation_weight'],
                    config['execution']['fees'],
                    config['execution']['slippage'],
                    freq,
                    base_funding_data,
                    alt_funding_data
                )
            else:
                # Use configured EMA window
                best_result = backtest_relative_momentum_pair(
                    data[base_symbol],
                    data[alt_symbol],
                    pair_config['ema_window'],
                    pair_config['allocation_weight'],
                    config['execution']['fees'],
                    config['execution']['slippage'],
                    freq,
                    base_funding_data,
                    alt_funding_data
                )

            # Print and store results
            stats = print_stats(pair_name, best_result, freq)
            summary_rows.append(stats)
            pair_returns[pair_name] = best_result['returns']

        except Exception as e:
            logger.error(f"Failed to backtest {pair_name}: {e}")
            continue

    if not pair_returns:
        logger.error("No successful backtests completed")
        return

    # Portfolio analysis
    df_all = pd.DataFrame(pair_returns).dropna()

    if len(df_all.columns) == 0:
        logger.error("No valid returns data for portfolio analysis")
        return

    # Calculate benchmark performance
    benchmark_returns = calculate_benchmark_performance(data, start_dt, end_dt)
    if len(benchmark_returns) > 0:
        # Align benchmark with strategy returns
        benchmark_returns = benchmark_returns.reindex(df_all.index).fillna(0)

    # Equal weight portfolio
    equal_w = np.ones(df_all.shape[1]) / df_all.shape[1]
    equal_portfolio_returns = (df_all * equal_w).sum(axis=1)

    # Volatility scaled portfolio
    vols = df_all.std()
    if (vols > 0).all():
        inv_vol_w = (1 / vols) / np.sum(1 / vols)
        vol_scaled_portfolio_returns = (df_all * inv_vol_w).sum(axis=1)
    else:
        vol_scaled_portfolio_returns = equal_portfolio_returns
        logger.warning("Using equal weights due to zero volatility pairs")

    def calculate_portfolio_metrics(returns, name):
        """Calculate portfolio metrics"""
        ann_ret, ann_vol, sharpe, max_dd = compute_metrics(returns, freq)
        calmar = ann_ret / abs(max_dd) if max_dd != 0 else np.nan
        win_rate = (returns > 0).mean()
        final_perf = (1 + returns).cumprod().iloc[-1]

        return {
            'Pair': name,
            'EMA_Window': 'N/A',
            'Ann.Return': ann_ret,
            'Vol': ann_vol,
            'Sharpe': sharpe,
            'Calmar': calmar,
            'Max DD': max_dd,
            'Win Rate': win_rate,
            'Final_Perf': final_perf
        }

    # Calculate portfolio metrics
    eq_stats = calculate_portfolio_metrics(equal_portfolio_returns, "Equal-Weight Portfolio")
    vs_stats = calculate_portfolio_metrics(vol_scaled_portfolio_returns, "Vol-Scaled Portfolio")

    summary_rows.extend([eq_stats, vs_stats])

    # Create results table
    results_table = pd.DataFrame(summary_rows)
    os.makedirs("results", exist_ok=True)
    results_table.to_csv("results/relative_momentum_results.csv", index=False)

    print("\n======= STRATEGY PERFORMANCE SUMMARY =======")
    print(results_table.to_string(index=False))
    print("\n✅ Results saved to results/relative_momentum_results.csv")

    # Performance comparison with benchmark
    if len(benchmark_returns) > 0:
        initial_capital = config.get('backtest', {}).get('initial_capital', 10000)

        print_performance_comparison(
            equal_portfolio_returns, benchmark_returns,
            "Equal-Weight Portfolio", "BTC Buy & Hold", freq,
            initial_capital, None
        )
        print_performance_comparison(
            vol_scaled_portfolio_returns, benchmark_returns,
            "Vol-Scaled Portfolio", "BTC Buy & Hold", freq,
            initial_capital, None
        )

    # Plot portfolio values
    print(f"\nBacktest Date Range: {df_all.index.min().strftime('%Y-%m-%d')} → {df_all.index.max().strftime('%Y-%m-%d')}")

    plot_portfolio_performance(
        df_all, equal_portfolio_returns, vol_scaled_portfolio_returns,
        benchmark_returns if len(benchmark_returns) > 0 else None,
        initial_capital
    )

    return results_table


def plot_portfolio_performance(individual_returns, equal_portfolio_returns, vol_scaled_portfolio_returns,
                              benchmark_returns=None, initial_capital=10000):
    """Plot comprehensive portfolio performance analysis"""

    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Portfolio Performance Analysis", fontsize=16, weight='bold')

    # 1. Portfolio Values (Top Left)
    eq_portfolio_value = initial_capital * (1 + equal_portfolio_returns).cumprod()
    vs_portfolio_value = initial_capital * (1 + vol_scaled_portfolio_returns).cumprod()

    ax1.plot(eq_portfolio_value.index, eq_portfolio_value, linewidth=3, color='black',
             linestyle='--', label='Equal-Weight Portfolio', alpha=0.9)
    ax1.plot(vs_portfolio_value.index, vs_portfolio_value, linewidth=3, color='red',
             linestyle='-.', label='Vol-Scaled Portfolio', alpha=0.9)

    if benchmark_returns is not None:
        bench_portfolio_value = initial_capital * (1 + benchmark_returns).cumprod()
        bench_portfolio_value = bench_portfolio_value.reindex(eq_portfolio_value.index).ffill()
        ax1.plot(bench_portfolio_value.index, bench_portfolio_value, linewidth=2, color='blue',
                 label='BTC Buy & Hold', alpha=0.8)

    ax1.set_title("Portfolio Value Over Time", fontweight='bold')
    ax1.set_ylabel("Portfolio Value ($)")
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 2. Cumulative Returns (Top Right)
    (1 + equal_portfolio_returns).cumprod().plot(ax=ax2, linewidth=3, color='black',
                                                  linestyle='--', label='Equal-Weight Portfolio')
    (1 + vol_scaled_portfolio_returns).cumprod().plot(ax=ax2, linewidth=3, color='red',
                                                       linestyle='-.', label='Vol-Scaled Portfolio')

    # Individual strategies with lower alpha
    for col in individual_returns.columns:
        (1 + individual_returns[col]).cumprod().plot(ax=ax2, linewidth=1.2, alpha=0.6, label=col)

    if benchmark_returns is not None:
        bench_cumret = (1 + benchmark_returns).reindex(individual_returns.index).fillna(0)
        (1 + bench_cumret).cumprod().plot(ax=ax2, linewidth=2, color='blue',
                                          label='BTC Buy & Hold', alpha=0.8)

    ax2.set_title("Cumulative Returns", fontweight='bold')
    ax2.set_ylabel("Growth (×)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # 3. Drawdown Analysis (Bottom Left)
    eq_cumret = (1 + equal_portfolio_returns).cumprod()
    vs_cumret = (1 + vol_scaled_portfolio_returns).cumprod()

    eq_dd = (eq_cumret - eq_cumret.cummax()) / eq_cumret.cummax()
    vs_dd = (vs_cumret - vs_cumret.cummax()) / vs_cumret.cummax()

    ax3.fill_between(eq_dd.index, eq_dd, 0, alpha=0.3, color='black', label='Equal-Weight DD')
    ax3.fill_between(vs_dd.index, vs_dd, 0, alpha=0.3, color='red', label='Vol-Scaled DD')

    if benchmark_returns is not None:
        bench_cumret = (1 + benchmark_returns).reindex(eq_dd.index).ffill().cumprod()
        bench_dd = (bench_cumret - bench_cumret.cummax()) / bench_cumret.cummax()
        ax3.fill_between(bench_dd.index, bench_dd, 0, alpha=0.3, color='blue', label='BTC DD')

    ax3.set_title("Drawdown Analysis", fontweight='bold')
    ax3.set_ylabel("Drawdown (%)")
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # 4. Rolling Sharpe Ratio (Bottom Right)
    window = 30  # 30-day rolling window
    eq_rolling_sharpe = equal_portfolio_returns.rolling(window).mean() / equal_portfolio_returns.rolling(window).std() * np.sqrt(365)
    vs_rolling_sharpe = vol_scaled_portfolio_returns.rolling(window).mean() / vol_scaled_portfolio_returns.rolling(window).std() * np.sqrt(365)

    ax4.plot(eq_rolling_sharpe.index, eq_rolling_sharpe, linewidth=2, color='black',
             linestyle='--', label='Equal-Weight', alpha=0.8)
    ax4.plot(vs_rolling_sharpe.index, vs_rolling_sharpe, linewidth=2, color='red',
             linestyle='-.', label='Vol-Scaled', alpha=0.8)

    if benchmark_returns is not None:
        bench_rolling_sharpe = benchmark_returns.rolling(window).mean() / benchmark_returns.rolling(window).std() * np.sqrt(365)
        bench_rolling_sharpe = bench_rolling_sharpe.reindex(eq_rolling_sharpe.index)
        ax4.plot(bench_rolling_sharpe.index, bench_rolling_sharpe, linewidth=2, color='blue',
                 label='BTC', alpha=0.8)

    ax4.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax4.set_title(f"Rolling {window}-Day Sharpe Ratio", fontweight='bold')
    ax4.set_ylabel("Sharpe Ratio")
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    plt.tight_layout()
    plt.subplots_adjust(top=0.93)

    # Save the plot
    plt.savefig("results/portfolio_performance_analysis.png", dpi=300, bbox_inches='tight')
    print("\n📊 Portfolio performance chart saved to: results/portfolio_performance_analysis.png")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run unified relative momentum strategy backtest')
    parser.add_argument('--start-date', type=str,
                       help='Start date for backtest (YYYY-MM-DD)',
                       default=None)
    parser.add_argument('--end-date', type=str,
                       help='End date for backtest (YYYY-MM-DD)',
                       default=None)
    parser.add_argument('--config', type=str,
                       help='Path to configuration file',
                       default='config/unified_trading_config.yaml')

    args = parser.parse_args()

    # Validate date formats if provided
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

    # Validate date order
    if args.start_date and args.end_date:
        start_dt = datetime.strptime(args.start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(args.end_date, '%Y-%m-%d')
        if start_dt >= end_dt:
            print("Error: start-date must be before end-date")
            sys.exit(1)

    run_relative_momentum_backtest(
        start_date=args.start_date,
        end_date=args.end_date,
        config_path=args.config
    )