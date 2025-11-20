"""
Shared utilities for backtesting across all strategies
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)


def calculate_sortino_ratio(returns: pd.Series, freq: int = 365) -> float:
    """Calculate Sortino ratio (risk-adjusted return using downside deviation)"""
    returns = returns.dropna()
    if len(returns) == 0:
        return np.nan

    ann_return = (np.prod(1 + returns)) ** (freq / len(returns)) - 1
    negative_returns = returns[returns < 0]

    if len(negative_returns) == 0:
        return np.inf

    downside_std = np.std(negative_returns, ddof=1)
    if downside_std == 0:
        return np.inf

    return ann_return / (downside_std * np.sqrt(freq))


def calculate_beta(strategy_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Calculate beta coefficient (systematic risk relative to benchmark)"""
    # Align the series
    aligned_data = pd.concat([strategy_returns, benchmark_returns], axis=1, join='inner').dropna()

    if len(aligned_data) < 2:
        return np.nan

    strategy_aligned = aligned_data.iloc[:, 0]
    benchmark_aligned = aligned_data.iloc[:, 1]

    # Calculate covariance and variance
    covariance = np.cov(strategy_aligned, benchmark_aligned)[0, 1]
    benchmark_variance = np.var(benchmark_aligned, ddof=1)

    if benchmark_variance == 0:
        return np.nan

    beta = covariance / benchmark_variance
    return beta


def calculate_all_performance_metrics(returns: pd.Series, freq: int = 365) -> Dict[str, float]:
    """Calculate comprehensive performance metrics for a return series"""

    if len(returns) == 0:
        return {
            'total_return': np.nan, 'ann_return': np.nan, 'volatility': np.nan,
            'sharpe': np.nan, 'sortino': np.nan, 'max_dd': np.nan,
            'win_rate': np.nan, 'final_value': np.nan, 'calmar': np.nan
        }

    returns = returns.dropna()

    # Basic returns
    total_return = (1 + returns).prod() - 1
    ann_return = (1 + returns).prod() ** (freq / len(returns)) - 1
    volatility = returns.std() * np.sqrt(freq)

    # Risk-adjusted ratios
    sharpe = ann_return / volatility if volatility != 0 else np.nan
    sortino = calculate_sortino_ratio(returns, freq)

    # Drawdown analysis
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_dd = drawdown.min()

    # Calmar ratio
    calmar = ann_return / abs(max_dd) if max_dd != 0 else np.nan

    # Trading metrics
    win_rate = (returns > 0).mean()
    final_value = (1 + returns).prod()

    return {
        'total_return': total_return,
        'ann_return': ann_return,
        'volatility': volatility,
        'sharpe': sharpe,
        'sortino': sortino,
        'max_dd': max_dd,
        'calmar': calmar,
        'win_rate': win_rate,
        'final_value': final_value
    }


def calculate_benchmark_performance(data: Dict[str, pd.DataFrame],
                                  start_date: pd.Timestamp,
                                  end_date: pd.Timestamp,
                                  benchmark_symbol: str = 'BTCUSDT') -> pd.Series:
    """Calculate benchmark performance for comparison"""
    if benchmark_symbol not in data:
        logger.warning(f"Benchmark symbol {benchmark_symbol} not found in data")
        return pd.Series(dtype=float)

    benchmark_data = data[benchmark_symbol]
    benchmark_data = benchmark_data.loc[start_date:end_date]
    benchmark_returns = benchmark_data['close'].pct_change().dropna()

    return benchmark_returns


def print_performance_comparison(strategy_returns: pd.Series,
                               benchmark_returns: pd.Series,
                               strategy_name: str,
                               benchmark_name: str = 'BTC Buy & Hold',
                               freq: int = 365) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Print standardized performance comparison between strategy and benchmark"""

    strategy_metrics = calculate_all_performance_metrics(strategy_returns, freq)
    benchmark_metrics = calculate_all_performance_metrics(benchmark_returns, freq)

    # Calculate beta
    strategy_beta = calculate_beta(strategy_returns, benchmark_returns)
    strategy_metrics['beta'] = strategy_beta

    print(f"\n{'='*70}")
    print(f"PERFORMANCE COMPARISON: {strategy_name} vs {benchmark_name}")
    print(f"{'='*70}")

    print(f"{'Metric':<20} {'Strategy':<15} {'Benchmark':<15} {'Difference':<15}")
    print(f"{'-'*70}")

    metrics = [
        ('Total Return', 'total_return', '{:.2%}'),
        ('Ann. Return', 'ann_return', '{:.2%}'),
        ('Volatility', 'volatility', '{:.2%}'),
        ('Beta', 'beta', '{:.2f}'),
        ('Sharpe Ratio', 'sharpe', '{:.2f}'),
        ('Sortino Ratio', 'sortino', '{:.2f}'),
        ('Calmar Ratio', 'calmar', '{:.2f}'),
        ('Max Drawdown', 'max_dd', '{:.2%}'),
        ('Win Rate', 'win_rate', '{:.2%}'),
        ('Final Value', 'final_value', '{:.2f}x'),
    ]

    for metric_name, metric_key, fmt in metrics:
        strat_val = strategy_metrics[metric_key]

        # Beta is special - only for strategy relative to benchmark
        if metric_key == 'beta':
            bench_str = "1.00"  # Benchmark always has beta of 1.0 relative to itself
            diff = strat_val - 1.0 if not np.isnan(strat_val) else np.nan
            diff_str = f"{diff:+.2f}" if not np.isnan(diff) else "N/A"
        else:
            bench_val = benchmark_metrics[metric_key]

            if np.isnan(strat_val) or np.isnan(bench_val):
                diff_str = "N/A"
            else:
                if metric_key in ['max_dd']:  # For drawdown, better is higher (less negative)
                    diff = strat_val - bench_val
                else:  # For other metrics, better is higher
                    diff = strat_val - bench_val

                if metric_key in ['total_return', 'ann_return', 'volatility', 'max_dd', 'win_rate']:
                    diff_str = f"{diff:+.2%}"
                else:
                    diff_str = f"{diff:+.2f}"

            bench_str = fmt.format(bench_val) if not np.isnan(bench_val) else "N/A"

        strat_str = fmt.format(strat_val) if not np.isnan(strat_val) else "N/A"

        print(f"{metric_name:<20} {strat_str:<15} {bench_str:<15} {diff_str:<15}")

    print(f"{'='*70}")

    # Risk-adjusted performance summary
    if not np.isnan(strategy_metrics['sharpe']) and not np.isnan(benchmark_metrics['sharpe']):
        sharpe_improvement = strategy_metrics['sharpe'] - benchmark_metrics['sharpe']
        if sharpe_improvement > 0:
            print(f"✅ Strategy outperforms benchmark by {sharpe_improvement:.2f} Sharpe points")
        else:
            print(f"❌ Strategy underperforms benchmark by {abs(sharpe_improvement):.2f} Sharpe points")

    # Beta interpretation
    if not np.isnan(strategy_beta):
        if strategy_beta > 1.2:
            beta_desc = "high systematic risk (more volatile than benchmark)"
        elif strategy_beta > 0.8:
            beta_desc = "moderate systematic risk (similar volatility to benchmark)"
        elif strategy_beta > 0.0:
            beta_desc = "low systematic risk (less volatile than benchmark)"
        else:
            beta_desc = "negative correlation with benchmark"

        print(f"📊 Beta {strategy_beta:.2f}: Strategy has {beta_desc}")

    print()

    return strategy_metrics, benchmark_metrics


def create_summary_table(individual_results: list, portfolio_results: list = None) -> pd.DataFrame:
    """Create standardized summary table for backtest results"""

    summary_rows = []

    # Add individual strategy results
    for result in individual_results:
        if isinstance(result, dict):
            summary_rows.append(result)

    # Add portfolio results if provided
    if portfolio_results:
        for result in portfolio_results:
            if isinstance(result, dict):
                summary_rows.append(result)

    return pd.DataFrame(summary_rows)


def save_backtest_results(results_table: pd.DataFrame,
                         strategy_name: str,
                         start_date: str,
                         end_date: str,
                         output_dir: str = "results") -> str:
    """Save backtest results with standardized naming"""

    import os
    from datetime import datetime

    os.makedirs(output_dir, exist_ok=True)

    # Create filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/{strategy_name}_backtest_{start_date}_{end_date}_{timestamp}.csv"

    results_table.to_csv(filename, index=False)

    return filename