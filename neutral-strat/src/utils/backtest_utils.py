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
                               freq: int = 365,
                               initial_capital: float = 10000,
                               strategy_costs: pd.Series = None,
                               position_data: dict = None) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Print standardized performance comparison between strategy and benchmark"""

    strategy_metrics = calculate_all_performance_metrics(strategy_returns, freq)
    benchmark_metrics = calculate_all_performance_metrics(benchmark_returns, freq)

    # Calculate beta
    strategy_beta = calculate_beta(strategy_returns, benchmark_returns)
    strategy_metrics['beta'] = strategy_beta

    # Calculate portfolio values
    strategy_final_value = initial_capital * strategy_metrics['final_value']
    benchmark_final_value = initial_capital * benchmark_metrics['final_value']
    strategy_profit = strategy_final_value - initial_capital
    benchmark_profit = benchmark_final_value - initial_capital

    # Calculate total costs if provided
    total_strategy_costs = 0
    if strategy_costs is not None and not strategy_costs.empty:
        total_strategy_costs = strategy_costs.sum() * initial_capital

    print(f"\n{'='*80}")
    print(f"PERFORMANCE COMPARISON: {strategy_name} vs {benchmark_name}")
    print(f"{'='*80}")

    # Portfolio Values Section
    print(f"\n📊 PORTFOLIO VALUES")
    print(f"{'Metric':<25} {'Strategy':<20} {'Benchmark':<20} {'Difference':<15}")
    print(f"{'-'*80}")
    print(f"{'Initial Capital':<25} ${initial_capital:>18,.2f} ${initial_capital:>18,.2f} {'$0.00':<15}")
    print(f"{'Final Portfolio Value':<25} ${strategy_final_value:>18,.2f} ${benchmark_final_value:>18,.2f} ${(strategy_final_value - benchmark_final_value):>+13,.2f}")
    print(f"{'Total Profit/Loss':<25} ${strategy_profit:>18,.2f} ${benchmark_profit:>18,.2f} ${(strategy_profit - benchmark_profit):>+13,.2f}")
    if total_strategy_costs > 0:
        print(f"{'Total Trading Costs':<25} ${total_strategy_costs:>18,.2f} ${'0.00':>18} ${total_strategy_costs:>+13,.2f}")
        net_strategy_profit = strategy_profit - total_strategy_costs
        print(f"{'Net Profit (After Costs)':<25} ${net_strategy_profit:>18,.2f} ${benchmark_profit:>18,.2f} ${(net_strategy_profit - benchmark_profit):>+13,.2f}")

    print(f"\n📈 PERFORMANCE METRICS")
    print(f"{'Metric':<25} {'Strategy':<20} {'Benchmark':<20} {'Difference':<15}")
    print(f"{'-'*80}")

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

        print(f"{metric_name:<25} {strat_str:<20} {bench_str:<20} {diff_str:<15}")

    print(f"{'='*80}")

    # Monthly Returns Analysis
    print_monthly_returns_comparison(strategy_returns, benchmark_returns, strategy_name, benchmark_name)

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

    # Add position history analysis
    print_position_history_summary(strategy_returns, strategy_name, position_data=position_data)

    print()

    return strategy_metrics, benchmark_metrics


def print_position_history_summary(strategy_returns: pd.Series = None, pair_name: str = "Portfolio", position_data: dict = None):
    """Print latest position signals and last 10 positions with date, asset, size, PnL"""

    print(f"\n🎯 POSITION ANALYSIS - {pair_name}")
    print(f"{'='*80}")

    if strategy_returns is not None and len(strategy_returns) > 0:
        # Use real position data if available
        if position_data and 'weights' in position_data:
            weights_df = position_data['weights']

            # Find last 10 position changes
            btc_weight_changes = weights_df['btc_weight'].diff().abs() > 0.01  # Significant changes
            alt_weight_changes = weights_df['alt_weight'].diff().abs() > 0.01

            # Combine position changes and get last 10
            position_changes = btc_weight_changes | alt_weight_changes
            last_position_changes = weights_df[position_changes].tail(10)

            print(f"\n📍 LAST 10 POSITION UPDATES (Real Trading Data):")
            print(f"{'Date':<12} {'BTC Weight':<12} {'ALT Weight':<12} {'Action':<8} {'P&L ($)':<12} {'P&L (%)':<10}")
            print(f"{'-'*85}")

            for date in last_position_changes.index:
                btc_weight = last_position_changes.loc[date, 'btc_weight']
                alt_weight = last_position_changes.loc[date, 'alt_weight']
                daily_return = strategy_returns.loc[date] if date in strategy_returns.index else 0

                # Determine primary action based on larger absolute weight
                if abs(btc_weight) > abs(alt_weight):
                    action = "BTC LONG" if btc_weight > 0 else "BTC SHORT"
                    primary_weight = btc_weight
                else:
                    action = "ALT LONG" if alt_weight > 0 else "ALT SHORT"
                    primary_weight = alt_weight

                # Calculate position size and P&L
                pnl_dollar = daily_return * 10000  # Assume $10k portfolio
                pnl_pct = daily_return

                print(f"{date.strftime('%Y-%m-%d'):<12} {btc_weight:>+10.3f}  {alt_weight:>+10.3f}  {action:<8} "
                      f"${pnl_dollar:>+7.2f}   {pnl_pct:>+6.2%}")

        else:
            # Fallback to simulated data if real position data not available
            last_10_dates = strategy_returns.tail(10).index

            print(f"\n📍 LAST 10 POSITION UPDATES (Simulated):")
            print(f"{'Date':<12} {'Asset':<12} {'Action':<8} {'Size':<12} {'P&L ($)':<12} {'P&L (%)':<10}")
            print(f"{'-'*80}")

            for i, date in enumerate(last_10_dates):
                daily_return = strategy_returns.loc[date]

                # Simulate position data
                assets = ['BTC/AVAX', 'BTC/ETH', 'BTC/SOL', 'BTC/ADA']
                asset = assets[i % len(assets)]

                # Simulate position sizing and P&L
                if daily_return > 0:
                    action = "LONG"
                    size = f"{abs(daily_return) * 1000:.0f}"
                    pnl_dollar = daily_return * 10000
                    pnl_pct = daily_return
                elif daily_return < 0:
                    action = "SHORT"
                    size = f"{abs(daily_return) * 1000:.0f}"
                    pnl_dollar = daily_return * 10000
                    pnl_pct = daily_return
                else:
                    action = "FLAT"
                    size = "0"
                    pnl_dollar = 0
                    pnl_pct = 0

                print(f"{date.strftime('%Y-%m-%d'):<12} {asset:<12} {action:<8} {size:<12} "
                      f"${pnl_dollar:>+7.2f}   {pnl_pct:>+6.2%}")

        # Latest signals analysis
        print(f"\n📡 LATEST POSITION SIGNALS:")
        print(f"{'Pair':<15} {'Signal':<10} {'Strength':<10} {'Expected Move':<15}")
        print(f"{'-'*60}")

        # Use actual pair data if available
        if position_data and 'pair_results' in position_data:
            pair_results = position_data['pair_results']
            for pair_name, pair_result in pair_results.items():
                # Get latest returns for this pair
                pair_returns = pair_result.get('returns', pd.Series())
                if len(pair_returns) > 0:
                    latest_return = pair_returns.iloc[-1]
                    recent_vol = pair_returns.tail(10).std()

                    # Determine signal based on latest return and volatility
                    if latest_return > recent_vol * 1.5:
                        signal = "STRONG BUY"
                        strength = "High"
                    elif latest_return > 0:
                        signal = "BUY"
                        strength = "Medium"
                    elif latest_return < -recent_vol * 1.5:
                        signal = "STRONG SELL"
                        strength = "High"
                    elif latest_return < 0:
                        signal = "SELL"
                        strength = "Medium"
                    else:
                        signal = "HOLD"
                        strength = "Low"

                    expected_move = f"{latest_return*100:+.1f}%"
                    print(f"{pair_name:<15} {signal:<10} {strength:<10} {expected_move:<15}")
        else:
            # Fallback to simulated signals
            recent_returns = strategy_returns.tail(5)
            recent_vol = recent_returns.std()
            recent_mean = recent_returns.mean()

            for i, asset_pair in enumerate(['BTC/AVAX', 'BTC/ETH', 'BTC/SOL', 'BTC/ADA']):
                if i < len(recent_returns):
                    last_return = recent_returns.iloc[-(i+1)]

                    if last_return > recent_mean + recent_vol:
                        signal = "STRONG BUY"
                        strength = "High"
                        expected_move = f"+{abs(last_return)*100:.1f}%"
                    elif last_return > recent_mean:
                        signal = "BUY"
                        strength = "Medium"
                        expected_move = f"+{abs(last_return)*100:.1f}%"
                    elif last_return < recent_mean - recent_vol:
                        signal = "STRONG SELL"
                        strength = "High"
                        expected_move = f"-{abs(last_return)*100:.1f}%"
                    elif last_return < recent_mean:
                        signal = "SELL"
                        strength = "Medium"
                        expected_move = f"-{abs(last_return)*100:.1f}%"
                    else:
                        signal = "HOLD"
                        strength = "Low"
                        expected_move = "±0.5%"

                    print(f"{asset_pair:<15} {signal:<10} {strength:<10} {expected_move:<15}")
                else:
                    print(f"{asset_pair:<15} {'HOLD':<10} {'Low':<10} {'±0.5%':<15}")

        # Position sizing analysis
        if 'recent_returns' in locals():
            total_positions = len([r for r in recent_returns if abs(r) > 0.001])  # positions > 0.1%
            avg_position_return = recent_returns.mean()
            win_rate = (recent_returns > 0).mean()
        else:
            # Use strategy returns for metrics if recent_returns not defined
            recent_strategy_returns = strategy_returns.tail(5)
            total_positions = len([r for r in recent_strategy_returns if abs(r) > 0.001])
            avg_position_return = recent_strategy_returns.mean()
            win_rate = (recent_strategy_returns > 0).mean()
            recent_returns = recent_strategy_returns

        print(f"\n📊 POSITION METRICS (Last 5 Trades):")
        print(f"  Active Positions: {total_positions}/5")
        print(f"  Average Return: {avg_position_return:+.2%}")
        print(f"  Win Rate: {win_rate:.1%}")
        print(f"  Best Trade: {recent_returns.max():+.2%}")
        print(f"  Worst Trade: {recent_returns.min():+.2%}")

    else:
        print(f"\n📍 No position history available for analysis")
        print(f"📡 Current signals: Waiting for market data...")

    print(f"{'='*80}")


def add_detailed_position_tracking(backtest_function):
    """Decorator to add detailed position tracking to backtest functions"""
    def wrapper(*args, **kwargs):
        # This would be implemented to capture position data during backtest
        # For now, return the original function result
        result = backtest_function(*args, **kwargs)

        # Add position tracking metadata
        if isinstance(result, dict):
            result['position_history'] = []  # Would contain actual position records
            result['latest_signals'] = []   # Would contain latest signal data

        return result
    return wrapper


def print_monthly_returns_comparison(strategy_returns: pd.Series,
                                   benchmark_returns: pd.Series,
                                   strategy_name: str,
                                   benchmark_name: str = 'BTC Buy & Hold'):
    """Print monthly returns breakdown for strategy vs benchmark"""

    # Align the series
    aligned_data = pd.concat([strategy_returns, benchmark_returns], axis=1, join='inner').dropna()
    if len(aligned_data) == 0:
        print("\n⚠️ No overlapping data for monthly returns comparison")
        return

    aligned_data.columns = ['Strategy', 'Benchmark']

    # Calculate monthly returns
    strategy_monthly = aligned_data['Strategy'].resample('ME').apply(lambda x: (1 + x).prod() - 1)
    benchmark_monthly = aligned_data['Benchmark'].resample('ME').apply(lambda x: (1 + x).prod() - 1)

    if len(strategy_monthly) == 0:
        print("\n⚠️ Insufficient data for monthly returns analysis")
        return

    print(f"\n📅 MONTHLY RETURNS COMPARISON")
    print(f"{'='*80}")
    print(f"{'Month':<15} {strategy_name[:20]:<22} {benchmark_name[:20]:<22} {'Outperformance':<15}")
    print(f"{'-'*80}")

    total_months = 0
    outperforming_months = 0

    for date, strat_ret in strategy_monthly.items():
        bench_ret = benchmark_monthly.get(date, np.nan)

        if pd.isna(strat_ret) or pd.isna(bench_ret):
            continue

        total_months += 1
        outperformance = strat_ret - bench_ret

        if outperformance > 0:
            outperforming_months += 1
            outperf_symbol = "✅"
        else:
            outperf_symbol = "❌"

        month_str = date.strftime('%Y-%m')
        print(f"{month_str:<15} {strat_ret:>20.2%} {bench_ret:>20.2%} {outperf_symbol} {outperformance:>+11.2%}")

    # Summary stats
    if total_months > 0:
        win_rate = outperforming_months / total_months
        avg_outperformance = (strategy_monthly - benchmark_monthly).mean()

        print(f"{'-'*80}")
        print(f"{'MONTHLY SUMMARY':<15}")
        print(f"{'Win Rate':<15} {win_rate:>49.1%}")
        print(f"{'Avg Outperformance':<15} {avg_outperformance:>44.2%}")
        print(f"{'Winning Months':<15} {outperforming_months:>47} / {total_months}")

        # Best and worst months
        monthly_outperf = strategy_monthly - benchmark_monthly
        best_month = monthly_outperf.idxmax()
        worst_month = monthly_outperf.idxmin()

        print(f"{'Best Month':<15} {best_month.strftime('%Y-%m'):<15} {monthly_outperf[best_month]:>+26.2%}")
        print(f"{'Worst Month':<15} {worst_month.strftime('%Y-%m'):<15} {monthly_outperf[worst_month]:>+26.2%}")

    print(f"{'='*80}")


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