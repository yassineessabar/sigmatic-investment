"""
Statistical Arbitrage Strategy Implementation
Based on "A 2+ Sharpe Market-Neutral Statistical Arbitrage Strategy in Cryptocurrency"
Implements market-neutral mean reversion with residual returns calculation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


def calculate_market_neutral_signal(prices_df: pd.DataFrame, formation_period: int = 1) -> pd.DataFrame:
    """
    Calculate market-neutral signal using residual returns method.

    Args:
        prices_df: DataFrame with price data for all assets
        formation_period: Lookback period for signal calculation

    Returns:
        DataFrame of market-neutral signals
    """
    # Calculate raw returns over formation period
    raw_returns = prices_df.pct_change(formation_period)

    # Calculate market-neutral returns by subtracting cross-sectional mean
    market_neutral_returns = raw_returns.subtract(raw_returns.mean(axis=1), axis=0)

    return market_neutral_returns


def run_statistical_arbitrage_backtest(
    prices_df: pd.DataFrame,
    formation_period: int = 1,
    holding_period: int = 1,
    commission_bps: float = 5,
    long_percentile: float = 0.2,
    short_percentile: float = 0.8
) -> pd.Series:
    """
    Run statistical arbitrage backtest using market-neutral mean reversion.

    Args:
        prices_df: Price data for all assets
        formation_period: Signal formation period (days)
        holding_period: Position holding period (days)
        commission_bps: Transaction costs in basis points
        long_percentile: Percentile threshold for long positions (bottom performers)
        short_percentile: Percentile threshold for short positions (top performers)

    Returns:
        Series of strategy returns
    """
    # Calculate market-neutral signal
    signal = calculate_market_neutral_signal(prices_df, formation_period)

    # Rank signals across assets (ascending = True means worst performers get rank 1)
    ranks = signal.rank(axis=1, ascending=True, pct=True)

    # Create long/short positions based on percentile thresholds
    longs = (ranks <= long_percentile).astype(int)  # Bottom 20% - underperformers
    shorts = (ranks >= short_percentile).astype(int) * -1  # Top 20% - outperformers

    # Combine long and short positions
    raw_weights = longs + shorts

    # Normalize weights so they sum to 0 (market neutral) and positions are equally weighted
    positions = raw_weights.div(raw_weights.abs().sum(axis=1), axis=0).fillna(0)

    # Apply holding period - rebalance every holding_period days
    strategy_positions = positions[::holding_period].reindex(
        positions.index, method='ffill'
    ).fillna(0)

    # Calculate strategy returns
    asset_returns = prices_df.pct_change()
    strategy_returns = (strategy_positions.shift(1) * asset_returns).sum(axis=1)

    # Calculate transaction costs
    turnover = (strategy_positions - strategy_positions.shift(1)).abs().sum(axis=1)
    transaction_costs = turnover * (commission_bps / 10000)

    # Return net strategy returns after costs
    net_returns = strategy_returns - transaction_costs

    return net_returns.fillna(0)


def calculate_statistical_arbitrage_metrics(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series = None,
    risk_free_rate: float = 0.02
) -> Dict:
    """
    Calculate comprehensive performance metrics for statistical arbitrage strategy.

    Args:
        strategy_returns: Strategy return series
        benchmark_returns: Benchmark return series (optional)
        risk_free_rate: Annual risk-free rate

    Returns:
        Dictionary of performance metrics
    """
    # Annualization factor (assuming daily data)
    periods_per_year = 252

    # Basic performance metrics
    total_return = (1 + strategy_returns).prod() - 1
    annual_return = (1 + strategy_returns.mean()) ** periods_per_year - 1
    annual_volatility = strategy_returns.std() * np.sqrt(periods_per_year)

    # Risk-adjusted metrics
    sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility

    # Drawdown analysis
    cumulative = (1 + strategy_returns).cumprod()
    peak = cumulative.expanding().max()
    drawdown = (cumulative - peak) / peak
    max_drawdown = drawdown.min()

    # Sortino ratio (using downside deviation)
    downside_returns = strategy_returns[strategy_returns < 0]
    downside_deviation = downside_returns.std() * np.sqrt(periods_per_year)
    sortino_ratio = (annual_return - risk_free_rate) / downside_deviation if len(downside_returns) > 0 else np.inf

    # Calmar ratio
    calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else np.inf

    # Win rate
    win_rate = (strategy_returns > 0).mean()

    # Information ratio vs benchmark
    if benchmark_returns is not None:
        excess_returns = strategy_returns - benchmark_returns
        tracking_error = excess_returns.std() * np.sqrt(periods_per_year)
        information_ratio = excess_returns.mean() * np.sqrt(periods_per_year) / tracking_error
    else:
        information_ratio = None

    # Market neutrality test (strategy should have low beta to market)
    if benchmark_returns is not None:
        correlation = strategy_returns.corr(benchmark_returns)
        beta = strategy_returns.cov(benchmark_returns) / benchmark_returns.var()
    else:
        correlation = None
        beta = None

    metrics = {
        'total_return': total_return,
        'annual_return': annual_return,
        'annual_volatility': annual_volatility,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'calmar_ratio': calmar_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'information_ratio': information_ratio,
        'correlation_to_market': correlation,
        'beta_to_market': beta,
        'final_value': 1 + total_return
    }

    return metrics


def optimize_statistical_arbitrage_parameters(
    prices_df: pd.DataFrame,
    formation_periods: List[int] = [1, 2, 3, 5],
    holding_periods: List[int] = [1, 2, 3, 5],
    commission_bps: float = 5,
    optimize_metric: str = 'sharpe_ratio'
) -> Tuple[Dict, pd.DataFrame]:
    """
    Optimize statistical arbitrage strategy parameters.

    Args:
        prices_df: Price data for optimization
        formation_periods: Formation periods to test
        holding_periods: Holding periods to test
        commission_bps: Transaction costs
        optimize_metric: Metric to optimize ('sharpe_ratio', 'calmar_ratio', etc.)

    Returns:
        Tuple of (best_params, results_df)
    """
    results = []

    for fp in formation_periods:
        for hp in holding_periods:
            try:
                # Run backtest with current parameters
                returns = run_statistical_arbitrage_backtest(
                    prices_df, fp, hp, commission_bps
                )

                # Calculate metrics
                metrics = calculate_statistical_arbitrage_metrics(returns)

                # Store results
                result = {
                    'formation_period': fp,
                    'holding_period': hp,
                    **metrics
                }
                results.append(result)

            except Exception as e:
                print(f"Error with FP={fp}, HP={hp}: {e}")
                continue

    # Convert to DataFrame and find best parameters
    results_df = pd.DataFrame(results)

    if len(results_df) == 0:
        return None, None

    best_idx = results_df[optimize_metric].idxmax()
    best_params = results_df.loc[best_idx].to_dict()

    return best_params, results_df


def enhanced_statistical_arbitrage_strategy(
    pair_data: Dict[str, pd.DataFrame],
    config: Dict,
    enable_optimizations: bool = True
) -> pd.DataFrame:
    """
    Enhanced statistical arbitrage strategy implementation for crypto pairs.

    Args:
        pair_data: Dictionary of pair DataFrames with OHLCV data
        config: Strategy configuration
        enable_optimizations: Whether to enable enhanced features

    Returns:
        DataFrame with strategy positions and metadata
    """
    # Extract configuration
    formation_period = config.get('formation_period', 1)
    holding_period = config.get('holding_period', 1)
    commission_bps = config.get('commission_bps', 5)
    long_percentile = config.get('long_percentile', 0.2)
    short_percentile = config.get('short_percentile', 0.8)

    # Prepare price matrix for all pairs
    price_data = {}
    for pair_name, data in pair_data.items():
        if 'close' in data.columns:
            price_data[pair_name] = data['close']
        elif len(data.columns) == 1:
            price_data[pair_name] = data.iloc[:, 0]

    prices_df = pd.DataFrame(price_data).dropna()

    if len(prices_df.columns) < 2:
        # Fallback to simple momentum if insufficient pairs
        return pd.DataFrame()

    # Run statistical arbitrage strategy
    strategy_returns = run_statistical_arbitrage_backtest(
        prices_df,
        formation_period=formation_period,
        holding_period=holding_period,
        commission_bps=commission_bps,
        long_percentile=long_percentile,
        short_percentile=short_percentile
    )

    # Calculate positions for each pair
    signal = calculate_market_neutral_signal(prices_df, formation_period)
    ranks = signal.rank(axis=1, ascending=True, pct=True)

    # Create long/short positions
    longs = (ranks <= long_percentile).astype(int)
    shorts = (ranks >= short_percentile).astype(int) * -1
    raw_weights = longs + shorts

    # Normalize positions
    positions = raw_weights.div(raw_weights.abs().sum(axis=1), axis=0).fillna(0)

    # Apply holding period
    strategy_positions = positions[::holding_period].reindex(
        positions.index, method='ffill'
    ).fillna(0)

    # Prepare output DataFrame
    result_df = strategy_positions.copy()
    result_df['strategy_returns'] = strategy_returns
    result_df['signal_strength'] = signal.abs().mean(axis=1)

    return result_df


if __name__ == "__main__":
    # Example usage and testing
    print("Statistical Arbitrage Strategy Module")
    print("Based on market-neutral mean reversion research")
    print("Target: 2+ Sharpe ratio with low drawdown")