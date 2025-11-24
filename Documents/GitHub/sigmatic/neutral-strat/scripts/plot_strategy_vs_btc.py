#!/usr/bin/env python3
"""
Plot Strategy Performance vs BTC Buy & Hold Comparison
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import argparse

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from data.binance_loader import BinanceDataLoader
from src.strategies.relative_momentum import backtest_relative_momentum_pair, optimize_ema_window
from src.utils.backtest_utils import calculate_all_performance_metrics

def load_strategy_results(start_date='2020-01-01', end_date='2025-01-21'):
    """Load and calculate strategy performance"""

    print("Loading market data...")
    loader = BinanceDataLoader()

    # Define pairs based on our config
    pairs = [
        ('BTCUSDT', 'AVAXUSDT'),
        ('BTCUSDT', 'ETHUSDT'),
        ('BTCUSDT', 'SOLUSDT'),
        ('BTCUSDT', 'ADAUSDT')
    ]

    # Load data for all symbols
    symbols = list(set([symbol for pair in pairs for symbol in pair]))

    data = {}
    funding_data = {}

    for symbol in symbols:
        print(f"Loading {symbol}...")
        try:
            # Load price data
            price_data = loader.load_data(symbol, start_date, end_date, interval='1d', market_type='futures')
            data[symbol] = price_data

            # Load funding data
            funding = loader.load_funding_rates(symbol, start_date, end_date)
            funding_data[symbol] = funding

        except Exception as e:
            print(f"Error loading {symbol}: {e}")
            continue

    print("Running strategy backtests...")

    # Optimal EMA windows from our previous optimization
    optimal_windows = {
        ('BTCUSDT', 'AVAXUSDT'): 6,
        ('BTCUSDT', 'ETHUSDT'): 28,
        ('BTCUSDT', 'SOLUSDT'): 18,
        ('BTCUSDT', 'ADAUSDT'): 3
    }

    pair_results = []
    pair_returns = []

    for base_symbol, alt_symbol in pairs:
        if base_symbol not in data or alt_symbol not in data:
            print(f"Skipping {base_symbol}/{alt_symbol} - missing data")
            continue

        ema_window = optimal_windows.get((base_symbol, alt_symbol), 10)

        base_funding = funding_data.get(base_symbol)
        alt_funding = funding_data.get(alt_symbol)

        try:
            result = backtest_relative_momentum_pair(
                data[base_symbol],
                data[alt_symbol],
                ema_window=ema_window,
                allocation_weight=0.75,
                fees=0.0004,
                slippage=0.0005,
                base_funding=base_funding,
                alt_funding=alt_funding
            )

            pair_results.append(result)
            pair_returns.append(result['returns'])

            print(f"{base_symbol}/{alt_symbol}: {result['sharpe']:.2f} Sharpe, {result['final_performance']:.1f}x final")

        except Exception as e:
            print(f"Error backtesting {base_symbol}/{alt_symbol}: {e}")
            continue

    if not pair_returns:
        raise ValueError("No successful pair backtests")

    # Combine into portfolio (equal weight)
    portfolio_returns = pd.concat(pair_returns, axis=1).mean(axis=1)
    portfolio_cumulative = (1 + portfolio_returns).cumprod()

    return portfolio_returns, portfolio_cumulative, data['BTCUSDT']


def create_comparison_plot(strategy_cumulative, btc_data, start_date='2020-01-01'):
    """Create comparison plot between strategy and BTC"""

    # Calculate BTC cumulative returns
    btc_returns = btc_data['close'].pct_change().dropna()
    btc_cumulative = (1 + btc_returns).cumprod()

    # Align dates
    common_dates = strategy_cumulative.index.intersection(btc_cumulative.index)
    strategy_aligned = strategy_cumulative.loc[common_dates]
    btc_aligned = btc_cumulative.loc[common_dates]

    # Convert to portfolio values from $10,000 initial
    initial_capital = 10000
    strategy_values = strategy_aligned * initial_capital
    btc_values = btc_aligned * initial_capital

    # Create the plot
    plt.figure(figsize=(15, 10))

    # Main performance plot
    plt.subplot(2, 1, 1)
    plt.plot(strategy_values.index, strategy_values,
             label=f'Strategy Portfolio (Final: ${strategy_values.iloc[-1]:,.0f})',
             linewidth=2, color='#2E8B57', alpha=0.9)
    plt.plot(btc_values.index, btc_values,
             label=f'BTC Buy & Hold (Final: ${btc_values.iloc[-1]:,.0f})',
             linewidth=2, color='#FF6B35', alpha=0.9)

    plt.title('📈 Sigmatic Strategy vs BTC Buy & Hold Performance\n5-Year Backtest (2020-2025)',
              fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('Portfolio Value ($)', fontsize=12, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # Log scale to better show both performances

    # Format y-axis with dollar signs
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

    # Add performance metrics text box
    strategy_total_return = (strategy_values.iloc[-1] / initial_capital - 1) * 100
    btc_total_return = (btc_values.iloc[-1] / initial_capital - 1) * 100

    metrics_text = f"""Strategy Performance:
• Total Return: +{strategy_total_return:.0f}%
• Final Value: ${strategy_values.iloc[-1]:,.0f}
• Advantage: +${strategy_values.iloc[-1] - btc_values.iloc[-1]:,.0f}

BTC Buy & Hold:
• Total Return: +{btc_total_return:.0f}%
• Final Value: ${btc_values.iloc[-1]:,.0f}"""

    plt.text(0.02, 0.98, metrics_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Drawdown comparison
    plt.subplot(2, 1, 2)

    # Calculate drawdowns
    strategy_running_max = strategy_values.expanding().max()
    strategy_drawdown = (strategy_values - strategy_running_max) / strategy_running_max * 100

    btc_running_max = btc_values.expanding().max()
    btc_drawdown = (btc_values - btc_running_max) / btc_running_max * 100

    plt.fill_between(strategy_drawdown.index, strategy_drawdown, 0,
                     alpha=0.7, color='#2E8B57',
                     label=f'Strategy Max DD: {strategy_drawdown.min():.1f}%')
    plt.fill_between(btc_drawdown.index, btc_drawdown, 0,
                     alpha=0.7, color='#FF6B35',
                     label=f'BTC Max DD: {btc_drawdown.min():.1f}%')

    plt.title('📉 Drawdown Comparison', fontsize=14, fontweight='bold')
    plt.ylabel('Drawdown (%)', fontsize=12, fontweight='bold')
    plt.xlabel('Date', fontsize=12, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    # Format dates on x-axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.xticks(rotation=45)

    plt.tight_layout()

    # Save the plot
    output_dir = os.path.join(project_root, 'results')
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'{output_dir}/strategy_vs_btc_comparison_{timestamp}.png'

    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Plot saved to: {filename}")

    plt.show()

    return filename


def main():
    parser = argparse.ArgumentParser(description='Plot Strategy vs BTC Performance')
    parser.add_argument('--start-date', default='2020-01-01', help='Start date for analysis')
    parser.add_argument('--end-date', default='2025-01-21', help='End date for analysis')

    args = parser.parse_args()

    try:
        print("🚀 Loading strategy performance data...")
        strategy_returns, strategy_cumulative, btc_data = load_strategy_results(args.start_date, args.end_date)

        print("📊 Creating comparison visualization...")
        filename = create_comparison_plot(strategy_cumulative, btc_data, args.start_date)

        # Print summary stats
        print("\n" + "="*80)
        print("📈 PERFORMANCE SUMMARY")
        print("="*80)

        initial_capital = 10000
        strategy_final = strategy_cumulative.iloc[-1] * initial_capital

        btc_returns = btc_data['close'].pct_change().dropna()
        btc_cumulative = (1 + btc_returns).cumprod()
        btc_final = btc_cumulative.iloc[-1] * initial_capital

        strategy_return = (strategy_final / initial_capital - 1) * 100
        btc_return = (btc_final / initial_capital - 1) * 100

        print(f"Strategy Portfolio:")
        print(f"  Initial Capital:  ${initial_capital:,.0f}")
        print(f"  Final Value:      ${strategy_final:,.0f}")
        print(f"  Total Return:     +{strategy_return:.0f}%")
        print()
        print(f"BTC Buy & Hold:")
        print(f"  Initial Capital:  ${initial_capital:,.0f}")
        print(f"  Final Value:      ${btc_final:,.0f}")
        print(f"  Total Return:     +{btc_return:.0f}%")
        print()
        print(f"Strategy Advantage: +${strategy_final - btc_final:,.0f} ({strategy_return - btc_return:+.0f}%)")
        print(f"Plot saved to: {filename}")

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())