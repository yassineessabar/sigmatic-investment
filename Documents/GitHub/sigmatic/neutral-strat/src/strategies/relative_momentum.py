from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from dataclasses import dataclass
import logging

from .neutral_pairs import Signal, PairSignal

logger = logging.getLogger(__name__)


def compute_relative_momentum_signals(data: Dict[str, pd.DataFrame], config: Dict) -> List[PairSignal]:
    """
    Relative momentum strategy that trades BTC/ALT pairs based on exponential moving average
    of relative price ratio. When BTC/ALT ratio > EMA, allocate more to BTC, otherwise to ALT.
    """
    signals = []

    for pair_config in config['pairs']:
        base_symbol = pair_config['base']
        alt_symbol = pair_config['alt']
        ema_window = pair_config['ema_window']
        allocation_weight = pair_config.get('allocation_weight', 0.75)
        max_notional = pair_config['max_notional']

        if base_symbol not in data or alt_symbol not in data:
            logger.warning(f"Missing data for pair {base_symbol}/{alt_symbol}")
            continue

        base_data = data[base_symbol]
        alt_data = data[alt_symbol]

        if len(base_data) < ema_window or len(alt_data) < ema_window:
            logger.warning(f"Insufficient data for pair {base_symbol}/{alt_symbol}")
            continue

        base_prices = base_data['close']
        alt_prices = alt_data['close']

        # Calculate relative price ratio and EMA
        relative_ratio = base_prices / alt_prices
        ema_ratio = relative_ratio.ewm(span=ema_window, adjust=False).mean()

        if len(relative_ratio) < 2 or len(ema_ratio) < 2:
            continue

        current_ratio = relative_ratio.iloc[-1]
        current_ema = ema_ratio.iloc[-1]

        # Determine previous weights to calculate position changes
        prev_ratio = relative_ratio.iloc[-2] if len(relative_ratio) > 1 else current_ratio
        prev_ema = ema_ratio.iloc[-2] if len(ema_ratio) > 1 else current_ema

        current_time = base_prices.index[-1]
        base_price = base_prices.iloc[-1]
        alt_price = alt_prices.iloc[-1]

        pair_name = f"{base_symbol}_{alt_symbol}"

        # Calculate current weights
        btc_weight = allocation_weight if current_ratio > current_ema else -allocation_weight
        alt_weight = allocation_weight if current_ratio < current_ema else -allocation_weight

        # Calculate previous weights
        prev_btc_weight = allocation_weight if prev_ratio > prev_ema else -allocation_weight
        prev_alt_weight = allocation_weight if prev_ratio < prev_ema else -allocation_weight

        # Only generate signals if weights changed
        if btc_weight != prev_btc_weight or alt_weight != prev_alt_weight:
            # Calculate position sizes
            base_position_size = abs(btc_weight) * max_notional / base_price
            alt_position_size = abs(alt_weight) * max_notional / alt_price

            # Determine sides
            base_side = 'long' if btc_weight > 0 else 'short'
            alt_side = 'long' if alt_weight > 0 else 'short'

            # Calculate confidence based on distance from EMA
            ratio_distance = abs(current_ratio - current_ema) / current_ema
            confidence = min(ratio_distance * 10, 2.0)  # Scale to reasonable range

            base_signal = Signal(
                symbol=base_symbol,
                side=base_side,
                size=base_position_size,
                price=base_price,
                timestamp=current_time,
                confidence=confidence,
                metadata={
                    'relative_ratio': current_ratio,
                    'ema_ratio': current_ema,
                    'weight': btc_weight
                }
            )

            alt_signal = Signal(
                symbol=alt_symbol,
                side=alt_side,
                size=alt_position_size,
                price=alt_price,
                timestamp=current_time,
                confidence=confidence,
                metadata={
                    'relative_ratio': current_ratio,
                    'ema_ratio': current_ema,
                    'weight': alt_weight
                }
            )

            entry_reason = f"Relative momentum: {base_symbol}/{alt_symbol} ratio {current_ratio:.6f} {'>' if current_ratio > current_ema else '<'} EMA {current_ema:.6f}"

            pair_signal = PairSignal(
                base_signal=base_signal,
                hedge_signal=alt_signal,
                pair_name=pair_name,
                spread_zscore=ratio_distance,  # Use ratio distance as proxy for signal strength
                entry_reason=entry_reason
            )
            signals.append(pair_signal)

    return signals


def compute_metrics(returns: pd.Series, freq: int = 365) -> Tuple[float, float, float, float]:
    """Calculate annualized return, volatility, Sharpe ratio, and max drawdown"""
    if isinstance(returns, pd.DataFrame):
        returns = returns.squeeze()

    returns = returns.dropna()
    if len(returns) == 0:
        return np.nan, np.nan, np.nan, np.nan

    # Annualized metrics
    ann_return = (np.prod(1 + returns)) ** (freq / len(returns)) - 1
    ann_vol = np.std(returns, ddof=1) * np.sqrt(freq)
    sharpe_ratio = ann_return / ann_vol if ann_vol != 0 else np.nan

    # Maximum drawdown
    cumulative = (1 + returns).cumprod()
    max_dd = (cumulative / cumulative.cummax()).min() - 1

    return ann_return, ann_vol, sharpe_ratio, max_dd


def backtest_relative_momentum_pair(
    base_data: pd.DataFrame,
    alt_data: pd.DataFrame,
    ema_window: int,
    allocation_weight: float = 0.75,
    fees: float = 0.0006,
    slippage: float = 0.001,
    freq: int = 365,
    base_funding: pd.DataFrame = None,
    alt_funding: pd.DataFrame = None
) -> Dict:
    """
    Backtest the relative momentum strategy for a single BTC/ALT pair
    """
    # Align data
    aligned_data = base_data['close'].to_frame(name='base').join(
        alt_data['close'].to_frame(name='alt'), how='inner'
    )

    # Calculate relative ratio and returns
    aligned_data['relative_ratio'] = aligned_data['base'] / aligned_data['alt']
    aligned_data['base_ret'] = aligned_data['base'].pct_change()
    aligned_data['alt_ret'] = aligned_data['alt'].pct_change()
    aligned_data.dropna(inplace=True)

    # Calculate EMA
    aligned_data['ema_ratio'] = aligned_data['relative_ratio'].ewm(
        span=ema_window, adjust=False
    ).mean()

    # Calculate weights
    aligned_data['btc_weight'] = np.where(
        aligned_data['relative_ratio'] > aligned_data['ema_ratio'],
        allocation_weight,
        -allocation_weight
    )
    aligned_data['alt_weight'] = np.where(
        aligned_data['relative_ratio'] < aligned_data['ema_ratio'],
        allocation_weight,
        -allocation_weight
    )

    # Shift weights to avoid look-ahead bias
    aligned_data[['btc_weight', 'alt_weight']] = aligned_data[['btc_weight', 'alt_weight']].shift(1)
    aligned_data.dropna(inplace=True)

    # Calculate strategy returns
    aligned_data['strategy_ret'] = (
        aligned_data['btc_weight'] * aligned_data['base_ret'] +
        aligned_data['alt_weight'] * aligned_data['alt_ret']
    )

    # Calculate trading costs
    weight_changes = (aligned_data['btc_weight'].shift(1) != aligned_data['btc_weight']).astype(int)
    aligned_data['trading_costs'] = weight_changes * 2 * (fees + slippage)

    # Calculate funding costs for futures positions
    aligned_data['funding_costs'] = 0.0

    if base_funding is not None and alt_funding is not None:
        # Resample funding rates to daily frequency (funding paid every 8 hours)
        base_funding_daily = base_funding.resample('D').mean().reindex(aligned_data.index).fillna(0)
        alt_funding_daily = alt_funding.resample('D').mean().reindex(aligned_data.index).fillna(0)

        # Funding cost = position_weight * funding_rate * 3 (3 funding payments per day)
        # Positive weight = long position pays funding, negative weight = short position receives funding
        base_funding_cost = aligned_data['btc_weight'] * base_funding_daily['funding_rate'] * 3
        alt_funding_cost = aligned_data['alt_weight'] * alt_funding_daily['funding_rate'] * 3

        aligned_data['funding_costs'] = base_funding_cost + alt_funding_cost

    aligned_data['net_ret'] = aligned_data['strategy_ret'] - aligned_data['trading_costs'] - aligned_data['funding_costs']

    # Calculate cumulative performance
    cumulative_returns = (1 + aligned_data['net_ret']).cumprod()

    # Calculate metrics
    ann_return, ann_vol, sharpe, max_dd = compute_metrics(aligned_data['net_ret'], freq)

    return {
        'ema_window': ema_window,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'ann_return': ann_return,
        'ann_vol': ann_vol,
        'final_performance': cumulative_returns.iloc[-1],
        'returns': aligned_data['net_ret'],
        'cumulative': cumulative_returns,
        'weights': aligned_data[['btc_weight', 'alt_weight']].copy()
    }


def optimize_ema_window(
    base_data: pd.DataFrame,
    alt_data: pd.DataFrame,
    window_range: range = range(1, 30),
    allocation_weight: float = 0.75,
    fees: float = 0.0006,
    slippage: float = 0.001,
    freq: int = 365,
    base_funding: pd.DataFrame = None,
    alt_funding: pd.DataFrame = None
) -> Dict:
    """
    Optimize EMA window for relative momentum strategy
    """
    results = []

    for window in window_range:
        try:
            result = backtest_relative_momentum_pair(
                base_data, alt_data, window, allocation_weight, fees, slippage, freq,
                base_funding, alt_funding
            )
            result['window'] = window
            results.append(result)
        except Exception as e:
            logger.warning(f"Failed to backtest window {window}: {e}")
            continue

    if not results:
        raise ValueError("No valid backtest results")

    # Return best result based on Sharpe ratio
    best_result = max(results, key=lambda x: x['sharpe'] if not np.isnan(x['sharpe']) else -np.inf)
    best_result['all_results'] = results

    return best_result