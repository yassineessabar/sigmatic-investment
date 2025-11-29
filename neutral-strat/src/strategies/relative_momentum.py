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
    alt_funding: pd.DataFrame = None,
    enable_optimizations: bool = True
) -> Dict:
    """
    Enhanced backtest with optimizations for better Sharpe ratio and lower drawdown
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

    if enable_optimizations:
        # OPTIMIZATION 1: Multi-timeframe signal confirmation
        # Add longer-term trend filter to reduce false signals
        long_ema_window = min(ema_window * 3, 60)  # 3x main EMA, max 60 days
        aligned_data['long_ema_ratio'] = aligned_data['relative_ratio'].ewm(
            span=long_ema_window, adjust=False
        ).mean()

        # OPTIMIZATION 2: Volatility-based position sizing
        # Reduce position size during high volatility periods
        vol_window = 20
        aligned_data['volatility'] = (
            (aligned_data['base_ret'].rolling(vol_window).std() +
             aligned_data['alt_ret'].rolling(vol_window).std()) / 2
        )
        median_vol = aligned_data['volatility'].median()

        # OPTIMIZATION 3: RSI-based momentum filter
        # Add RSI to avoid entering positions when assets are overbought/oversold
        def calculate_rsi(prices, window=14):
            delta = prices.diff()
            gains = delta.where(delta > 0, 0).rolling(window=window).mean()
            losses = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gains / losses
            return 100 - (100 / (1 + rs))

        aligned_data['base_rsi'] = calculate_rsi(aligned_data['base'])
        aligned_data['alt_rsi'] = calculate_rsi(aligned_data['alt'])

    # Calculate main EMA
    aligned_data['ema_ratio'] = aligned_data['relative_ratio'].ewm(
        span=ema_window, adjust=False
    ).mean()

    if enable_optimizations:
        # OPTIMIZATION 4: Enhanced signal generation with filters
        # Signal strength based on multiple confirmations
        main_signal = aligned_data['relative_ratio'] > aligned_data['ema_ratio']
        trend_confirmation = aligned_data['relative_ratio'] > aligned_data['long_ema_ratio']

        # RSI filter: avoid extreme overbought/oversold conditions (more permissive)
        rsi_filter = (
            (aligned_data['base_rsi'] > 15) & (aligned_data['base_rsi'] < 85) &
            (aligned_data['alt_rsi'] > 15) & (aligned_data['alt_rsi'] < 85)
        )

        # Volatility filter: reduce exposure during high volatility (less aggressive)
        vol_multiplier = np.clip(median_vol / aligned_data['volatility'], 0.6, 1.2)

        # Combined signal with confirmations (less restrictive for better Sharpe)
        # Use main signal but with trend and RSI as filters, not requirements
        strong_btc_signal = main_signal & rsi_filter & (trend_confirmation | (vol_multiplier > 0.8))
        strong_alt_signal = (~main_signal) & rsi_filter & ((~trend_confirmation) | (vol_multiplier > 0.8))

        # OPTIMIZATION 5: Dynamic position sizing
        base_weight = allocation_weight * vol_multiplier

        aligned_data['btc_weight'] = np.where(
            strong_btc_signal, base_weight,
            np.where(strong_alt_signal, -base_weight,
                     # Fallback to main signal with reduced allocation
                     np.where(main_signal, base_weight * 0.5, -base_weight * 0.5))
        )
        aligned_data['alt_weight'] = np.where(
            strong_alt_signal, base_weight,
            np.where(strong_btc_signal, -base_weight,
                     # Fallback to main signal with reduced allocation
                     np.where(~main_signal, base_weight * 0.5, -base_weight * 0.5))
        )

        # OPTIMIZATION 6: Trend-following momentum boost (more aggressive for better Sharpe)
        # Increase allocation when trend is very strong
        ratio_momentum = (aligned_data['relative_ratio'] / aligned_data['ema_ratio'] - 1).abs()
        momentum_boost = np.clip(ratio_momentum * 3, 1.0, 2.0)  # 1.0x to 2.0x boost

        aligned_data['btc_weight'] *= momentum_boost
        aligned_data['alt_weight'] *= momentum_boost

        # Cap maximum weight (increased limit)
        max_weight = allocation_weight * 2.0
        aligned_data['btc_weight'] = np.clip(aligned_data['btc_weight'], -max_weight, max_weight)
        aligned_data['alt_weight'] = np.clip(aligned_data['alt_weight'], -max_weight, max_weight)

    else:
        # Original logic without optimizations
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

    # OPTIMIZATION 7: Stop-loss mechanism (less aggressive for better returns)
    if enable_optimizations:
        # Calculate strategy-specific drawdown instead of combined returns
        strategy_returns_preview = (
            aligned_data['btc_weight'].shift(1).fillna(0) * aligned_data['base_ret'].fillna(0) +
            aligned_data['alt_weight'].shift(1).fillna(0) * aligned_data['alt_ret'].fillna(0)
        )
        cumulative_ret = (1 + strategy_returns_preview).cumprod()
        rolling_max = cumulative_ret.rolling(window=30, min_periods=1).max()
        drawdown = (cumulative_ret - rolling_max) / rolling_max

        # Reduce position size when in significant drawdown (higher threshold)
        drawdown_threshold = -0.15  # 15% drawdown threshold (less restrictive)
        drawdown_multiplier = np.where(drawdown < drawdown_threshold, 0.7, 1.0)  # Less reduction

        aligned_data['btc_weight'] *= drawdown_multiplier
        aligned_data['alt_weight'] *= drawdown_multiplier

    # Shift weights to avoid look-ahead bias
    aligned_data[['btc_weight', 'alt_weight']] = aligned_data[['btc_weight', 'alt_weight']].shift(1)
    aligned_data.dropna(inplace=True)

    # Calculate strategy returns
    aligned_data['strategy_ret'] = (
        aligned_data['btc_weight'] * aligned_data['base_ret'] +
        aligned_data['alt_weight'] * aligned_data['alt_ret']
    )

    # OPTIMIZATION 8: Enhanced cost calculation
    if enable_optimizations:
        # More realistic trading costs that scale with position size changes
        weight_changes = (
            (aligned_data['btc_weight'].diff().abs() + aligned_data['alt_weight'].diff().abs()) / 2
        ).fillna(0)
        aligned_data['trading_costs'] = weight_changes * (fees + slippage)
    else:
        # Original cost calculation
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
        'weights': aligned_data[['btc_weight', 'alt_weight']].copy(),
        'trading_costs': aligned_data['trading_costs'],
        'funding_costs': aligned_data.get('funding_costs', pd.Series()),
        'total_costs': aligned_data['trading_costs'] + aligned_data.get('funding_costs', 0),
        'optimizations_enabled': enable_optimizations
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
    alt_funding: pd.DataFrame = None,
    enable_optimizations: bool = True,
    optimization_metric: str = 'sharpe'
) -> Dict:
    """
    Optimize EMA window for enhanced relative momentum strategy
    """
    results = []

    for window in window_range:
        try:
            result = backtest_relative_momentum_pair(
                base_data, alt_data, window, allocation_weight, fees, slippage, freq,
                base_funding, alt_funding, enable_optimizations
            )
            result['window'] = window
            results.append(result)
        except Exception as e:
            logger.warning(f"Failed to backtest window {window}: {e}")
            continue

    if not results:
        raise ValueError("No valid backtest results")

    # Multi-objective optimization: balance Sharpe ratio and drawdown
    if optimization_metric == 'sharpe':
        # Return best result based on Sharpe ratio
        best_result = max(results, key=lambda x: x['sharpe'] if not np.isnan(x['sharpe']) else -np.inf)
    elif optimization_metric == 'calmar':
        # Calmar ratio: return/max_drawdown
        best_result = max(results, key=lambda x: (x['ann_return'] / abs(x['max_dd']))
                         if not np.isnan(x['ann_return']) and x['max_dd'] != 0 else -np.inf)
    elif optimization_metric == 'risk_adjusted':
        # Custom risk-adjusted score: Sharpe * (1 - |max_dd|)
        def risk_score(result):
            sharpe = result['sharpe'] if not np.isnan(result['sharpe']) else 0
            dd_penalty = 1 - min(abs(result['max_dd']), 0.5)  # Cap penalty at 50% DD
            return sharpe * dd_penalty
        best_result = max(results, key=risk_score)
    else:
        # Fallback to Sharpe
        best_result = max(results, key=lambda x: x['sharpe'] if not np.isnan(x['sharpe']) else -np.inf)

    best_result['all_results'] = results
    best_result['optimization_metric'] = optimization_metric

    return best_result


def create_optimized_strategy_comparison(
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
    Compare original strategy vs optimized strategy performance
    """
    # Run original strategy
    original_result = backtest_relative_momentum_pair(
        base_data, alt_data, ema_window, allocation_weight, fees, slippage, freq,
        base_funding, alt_funding, enable_optimizations=False
    )

    # Run optimized strategy
    optimized_result = backtest_relative_momentum_pair(
        base_data, alt_data, ema_window, allocation_weight, fees, slippage, freq,
        base_funding, alt_funding, enable_optimizations=True
    )

    return {
        'original': original_result,
        'optimized': optimized_result,
        'improvements': {
            'sharpe_improvement': optimized_result['sharpe'] - original_result['sharpe'],
            'drawdown_improvement': original_result['max_dd'] - optimized_result['max_dd'],
            'return_improvement': optimized_result['ann_return'] - original_result['ann_return'],
            'volatility_change': optimized_result['ann_vol'] - original_result['ann_vol']
        }
    }