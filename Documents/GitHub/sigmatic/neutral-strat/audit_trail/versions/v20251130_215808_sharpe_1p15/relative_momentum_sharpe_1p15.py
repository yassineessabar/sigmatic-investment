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

    # Calculate main EMA (needed for all paths)
    aligned_data['ema_ratio'] = aligned_data['relative_ratio'].ewm(
        span=ema_window, adjust=False
    ).mean()

    if enable_optimizations:
        # ENHANCED SIGNAL GENERATION: Multi-timeframe momentum analysis
        fast_ema_window = max(int(ema_window * 0.7), 3)    # Fast momentum
        slow_ema_window = int(ema_window * 2.5)             # Intermediate trend
        very_slow_ema_window = int(ema_window * 5.0)        # Long-term trend

        # Multiple timeframe EMAs for signal confirmation
        aligned_data['fast_ema_ratio'] = aligned_data['relative_ratio'].ewm(
            span=fast_ema_window, adjust=False
        ).mean()
        aligned_data['slow_ema_ratio'] = aligned_data['relative_ratio'].ewm(
            span=slow_ema_window, adjust=False
        ).mean()
        aligned_data['very_slow_ema_ratio'] = aligned_data['relative_ratio'].ewm(
            span=very_slow_ema_window, adjust=False
        ).mean()

        # Signal strength based on timeframe alignment
        fast_signal = aligned_data['relative_ratio'] > aligned_data['fast_ema_ratio']
        main_signal = aligned_data['relative_ratio'] > aligned_data['ema_ratio']
        slow_signal = aligned_data['relative_ratio'] > aligned_data['slow_ema_ratio']
        very_slow_signal = aligned_data['relative_ratio'] > aligned_data['very_slow_ema_ratio']

        # Signal confirmation score (0-1 based on timeframe agreement)
        aligned_data['signal_confirmation'] = (
            fast_signal.astype(int) + main_signal.astype(int) +
            slow_signal.astype(int) + very_slow_signal.astype(int)
        ) / 4.0

        # ENHANCED VOLATILITY REGIME ANALYSIS
        vol_lookback = 25
        corr_lookback = 15
        trend_lookback = 30

        # Multi-scale volatility analysis
        aligned_data['short_vol'] = (
            (aligned_data['base_ret'].rolling(vol_lookback).std() +
             aligned_data['alt_ret'].rolling(vol_lookback).std()) / 2
        )
        aligned_data['long_vol'] = (
            (aligned_data['base_ret'].rolling(vol_lookback * 3).std() +
             aligned_data['alt_ret'].rolling(vol_lookback * 3).std()) / 2
        )

        # Volatility regime classification
        vol_median = aligned_data['short_vol'].rolling(vol_lookback * 4).median()
        vol_upper = aligned_data['short_vol'].rolling(vol_lookback * 4).quantile(0.8)
        vol_lower = aligned_data['short_vol'].rolling(vol_lookback * 4).quantile(0.2)

        aligned_data['vol_regime_score'] = np.where(
            aligned_data['short_vol'] < vol_lower, 1.0,  # Low vol = favorable
            np.where(aligned_data['short_vol'] > vol_upper, 0.3, 0.7)  # High vol = unfavorable
        )

        # Cross-asset correlation analysis
        aligned_data['correlation'] = aligned_data['base_ret'].rolling(
            corr_lookback
        ).corr(aligned_data['alt_ret'])

        # Correlation regime (lower correlation = better for pairs trading)
        aligned_data['corr_regime_score'] = np.clip(
            (0.5 - aligned_data['correlation'].abs()) * 2, 0.2, 1.0
        )

        # ENHANCED MOMENTUM AND TREND ANALYSIS
        def calculate_rsi(prices, window=14):
            delta = prices.diff()
            gains = delta.where(delta > 0, 0).rolling(window=window).mean()
            losses = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gains / losses
            return 100 - (100 / (1 + rs))

        def calculate_macd(prices, fast=12, slow=26, signal=9):
            exp1 = prices.ewm(span=fast).mean()
            exp2 = prices.ewm(span=slow).mean()
            macd_line = exp1 - exp2
            signal_line = macd_line.ewm(span=signal).mean()
            return macd_line, signal_line, macd_line - signal_line

        # Enhanced RSI analysis with optimal bounds
        aligned_data['base_rsi'] = calculate_rsi(aligned_data['base'])
        aligned_data['alt_rsi'] = calculate_rsi(aligned_data['alt'])

        # RSI momentum quality score (better when not in extreme zones)
        aligned_data['rsi_quality'] = (
            ((aligned_data['base_rsi'] > 25) & (aligned_data['base_rsi'] < 75)).astype(float) +
            ((aligned_data['alt_rsi'] > 25) & (aligned_data['alt_rsi'] < 75)).astype(float)
        ) / 2.0

        # Enhanced MACD analysis
        base_macd, base_signal_line, base_histogram = calculate_macd(aligned_data['base'])
        alt_macd, alt_signal_line, alt_histogram = calculate_macd(aligned_data['alt'])

        # MACD momentum strength
        aligned_data['base_macd_strength'] = base_histogram / aligned_data['base']
        aligned_data['alt_macd_strength'] = alt_histogram / aligned_data['alt']

        # Momentum acceleration (rate of change in momentum)
        aligned_data['momentum_acceleration'] = (
            aligned_data['base_macd_strength'].diff() - aligned_data['alt_macd_strength'].diff()
        ).abs().rolling(5).mean()

        # ENHANCED MARKET STRESS AND TREND STRENGTH ANALYSIS
        # Market stress detection with multiple indicators
        combined_returns = (aligned_data['base_ret'] + aligned_data['alt_ret']) / 2
        stress_window = 25

        # Volatility-based stress indicator
        short_stress = combined_returns.rolling(stress_window).std()
        long_stress = combined_returns.rolling(stress_window * 3).std()
        aligned_data['stress_indicator'] = short_stress / long_stress

        # Market stress score (inverted - higher is better)
        aligned_data['market_stress_score'] = np.clip(
            2.0 - aligned_data['stress_indicator'], 0.2, 1.0
        )

        # Trend strength measurement
        ratio_sma = aligned_data['relative_ratio'].rolling(30).mean()
        ratio_trend_strength = (
            (aligned_data['relative_ratio'] - ratio_sma).abs() / ratio_sma
        ).rolling(30).mean()

        aligned_data['trend_strength_score'] = np.clip(
            ratio_trend_strength * 20, 0.3, 1.0
        )

        # ADVANCED SIGNAL QUALITY SCORING SYSTEM
        # Combine all signal components into quality score

        # Base directional signals
        btc_favor = aligned_data['signal_confirmation'] > 0.5
        alt_favor = aligned_data['signal_confirmation'] < 0.5

        # Comprehensive signal quality score (0-1)
        momentum_score = (
            aligned_data['signal_confirmation'] * 0.4 +
            aligned_data['rsi_quality'] * 0.2 +
            np.clip(aligned_data['momentum_acceleration'].fillna(0) * 10, 0, 1) * 0.15 +
            (aligned_data['base_macd_strength'].abs() + aligned_data['alt_macd_strength'].abs()).fillna(0) * 0.25
        )

        trend_score = (
            aligned_data['trend_strength_score'] * 0.6 +
            aligned_data['signal_confirmation'] * 0.4
        )

        volatility_score = (
            aligned_data['vol_regime_score'] * 0.6 +
            aligned_data['corr_regime_score'] * 0.25 +
            aligned_data['market_stress_score'] * 0.15
        )

        # Overall signal quality (weighted combination)
        aligned_data['signal_quality'] = (
            momentum_score * 0.4 +
            trend_score * 0.3 +
            volatility_score * 0.3
        )

        # Apply minimum signal quality threshold
        min_quality_threshold = 0.4
        aligned_data['quality_filter'] = aligned_data['signal_quality'] >= min_quality_threshold

        # Enhanced signal classification with quality weighting
        strong_btc_signal = btc_favor & aligned_data['quality_filter'] & (aligned_data['signal_quality'] > 0.7)
        medium_btc_signal = btc_favor & aligned_data['quality_filter'] & (aligned_data['signal_quality'] > 0.5)
        weak_btc_signal = btc_favor & aligned_data['quality_filter']

        strong_alt_signal = alt_favor & aligned_data['quality_filter'] & (aligned_data['signal_quality'] > 0.7)
        medium_alt_signal = alt_favor & aligned_data['quality_filter'] & (aligned_data['signal_quality'] > 0.5)
        weak_alt_signal = alt_favor & aligned_data['quality_filter']

        # SIGNAL QUALITY-BASED POSITION SIZING
        # Base volatility adjustment
        vol_multiplier = np.clip(
            aligned_data['vol_regime_score'] * aligned_data['market_stress_score'], 0.5, 1.3
        )

        # Signal strength-based sizing multipliers
        strong_multiplier = 1.5 * aligned_data['signal_quality']
        medium_multiplier = 1.0 * aligned_data['signal_quality']
        weak_multiplier = 0.6 * aligned_data['signal_quality']

        # Calculate base weight with volatility adjustment
        base_weight = allocation_weight * vol_multiplier

        # Advanced position sizing based on signal quality and strength
        aligned_data['btc_weight'] = np.where(
            strong_btc_signal, base_weight * strong_multiplier,
            np.where(medium_btc_signal, base_weight * medium_multiplier,
                np.where(weak_btc_signal, base_weight * weak_multiplier,
                    np.where(strong_alt_signal, -base_weight * strong_multiplier,
                        np.where(medium_alt_signal, -base_weight * medium_multiplier,
                            np.where(weak_alt_signal, -base_weight * weak_multiplier, 0)))))
        )

        aligned_data['alt_weight'] = np.where(
            strong_alt_signal, base_weight * strong_multiplier,
            np.where(medium_alt_signal, base_weight * medium_multiplier,
                np.where(weak_alt_signal, base_weight * weak_multiplier,
                    np.where(strong_btc_signal, -base_weight * strong_multiplier,
                        np.where(medium_btc_signal, -base_weight * medium_multiplier,
                            np.where(weak_btc_signal, -base_weight * weak_multiplier, 0)))))
        )

        # ENHANCED MOMENTUM BOOST WITH SIGNAL QUALITY
        # Multi-component momentum measurement
        ratio_momentum = (aligned_data['relative_ratio'] / aligned_data['ema_ratio'] - 1).abs()
        trend_momentum = aligned_data['trend_strength_score']
        timeframe_momentum = aligned_data['signal_confirmation']

        # Combined momentum boost based on signal quality
        momentum_boost = np.clip(
            1.0 + (ratio_momentum * 2 + trend_momentum * 0.5 + timeframe_momentum * 0.5) * aligned_data['signal_quality'],
            0.7,  # Minimum 70% for weak signals
            1.8   # Maximum 180% for strongest signals
        )

        # Apply momentum boost
        aligned_data['btc_weight'] *= momentum_boost
        aligned_data['alt_weight'] *= momentum_boost

        # Dynamic weight caps based on signal quality
        max_weight_base = allocation_weight * 2.0
        max_weight_dynamic = max_weight_base * (0.8 + 0.4 * aligned_data['signal_quality'])

        aligned_data['btc_weight'] = np.clip(aligned_data['btc_weight'], -max_weight_dynamic, max_weight_dynamic)
        aligned_data['alt_weight'] = np.clip(aligned_data['alt_weight'], -max_weight_dynamic, max_weight_dynamic)

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

    # OPTIMIZATION 8: Back to simple stop-loss that worked
    if enable_optimizations:
        # Simple strategy drawdown (what worked in 1-year)
        strategy_returns_preview = (
            aligned_data['btc_weight'].shift(1).fillna(0) * aligned_data['base_ret'].fillna(0) +
            aligned_data['alt_weight'].shift(1).fillna(0) * aligned_data['alt_ret'].fillna(0)
        )
        cumulative_ret = (1 + strategy_returns_preview).cumprod()
        rolling_max = cumulative_ret.rolling(window=30, min_periods=1).max()
        drawdown = (cumulative_ret - rolling_max) / rolling_max

        # Simple threshold that worked well
        drawdown_threshold = -0.12  # 12% drawdown threshold
        drawdown_multiplier = np.where(drawdown < drawdown_threshold, 0.7, 1.0)

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