#!/usr/bin/env python3

"""
Enhanced Relative Momentum Strategy
Advanced technical analysis with adaptive position sizing and regime detection
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging

# Import base functions from relative_momentum
from .relative_momentum import compute_metrics

logger = logging.getLogger(__name__)


def calculate_advanced_features(base_data: pd.DataFrame, alt_data: pd.DataFrame) -> Dict[str, pd.Series]:
    """Calculate advanced technical features for enhanced strategy"""

    base_close = base_data['close']
    alt_close = alt_data['close']
    base_volume = base_data.get('volume', pd.Series(index=base_close.index, data=1))
    alt_volume = alt_data.get('volume', pd.Series(index=alt_close.index, data=1))

    # Relative ratio
    ratio = base_close / alt_close

    features = {}

    # 1. Multi-timeframe EMAs
    features['ema_5'] = ratio.ewm(span=5).mean()
    features['ema_10'] = ratio.ewm(span=10).mean()
    features['ema_20'] = ratio.ewm(span=20).mean()
    features['ema_50'] = ratio.ewm(span=50).mean()

    # 2. Momentum indicators
    features['roc_5'] = ratio.pct_change(5)
    features['roc_10'] = ratio.pct_change(10)
    features['roc_20'] = ratio.pct_change(20)

    # 3. Volatility measures
    features['volatility_10'] = ratio.rolling(10).std()
    features['volatility_20'] = ratio.rolling(20).std()
    features['volatility_ratio'] = features['volatility_10'] / features['volatility_20']

    # 4. RSI for both assets
    features['base_rsi'] = calculate_rsi(base_close, 14)
    features['alt_rsi'] = calculate_rsi(alt_close, 14)
    features['rsi_divergence'] = features['base_rsi'] - features['alt_rsi']

    # 5. Volume-weighted features
    features['base_vwap'] = calculate_vwap(base_close, base_volume, 20)
    features['alt_vwap'] = calculate_vwap(alt_close, alt_volume, 20)
    features['vwap_ratio'] = features['base_vwap'] / features['alt_vwap']

    # 6. Bollinger Bands for ratio
    bb_period = 20
    bb_std = 2
    features['bb_middle'] = ratio.rolling(bb_period).mean()
    features['bb_std'] = ratio.rolling(bb_period).std()
    features['bb_upper'] = features['bb_middle'] + (bb_std * features['bb_std'])
    features['bb_lower'] = features['bb_middle'] - (bb_std * features['bb_std'])
    features['bb_position'] = (ratio - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])

    # 7. Regime detection
    features['trend_strength'] = abs(features['ema_5'] - features['ema_20']) / features['ema_20']
    features['volatility_regime'] = (features['volatility_20'] > features['volatility_20'].rolling(50).quantile(0.7)).astype(int)

    # 8. Mean reversion indicators
    features['z_score'] = (ratio - ratio.rolling(20).mean()) / ratio.rolling(20).std()
    features['mean_reversion_signal'] = np.where(
        abs(features['z_score']) > 2,
        -np.sign(features['z_score']),
        0
    )

    return features


def calculate_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """Calculate RSI indicator"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_vwap(prices: pd.Series, volume: pd.Series, window: int) -> pd.Series:
    """Calculate Volume Weighted Average Price"""
    vwap = (prices * volume).rolling(window).sum() / volume.rolling(window).sum()
    return vwap


def generate_enhanced_signals(features: Dict[str, pd.Series], ratio: pd.Series, ema_window: int) -> Dict[str, pd.Series]:
    """Generate enhanced trading signals"""

    signals = {}

    # 1. Primary momentum signal
    ema_signal = ratio > features[f'ema_{ema_window}'] if f'ema_{ema_window}' in features else ratio > features['ema_10']
    signals['momentum'] = ema_signal.astype(int) * 2 - 1  # Convert to -1, 1

    # 2. Trend strength filter
    trend_threshold = 0.02  # 2% trend strength required
    strong_trend = features['trend_strength'] > trend_threshold
    signals['trend_filter'] = strong_trend

    # 3. Multi-timeframe confirmation
    short_above_medium = features['ema_5'] > features['ema_10']
    medium_above_long = features['ema_10'] > features['ema_20']
    signals['multi_tf_bull'] = short_above_medium & medium_above_long
    signals['multi_tf_bear'] = ~short_above_medium & ~medium_above_long

    # 4. RSI filter (avoid overbought/oversold extremes)
    rsi_neutral_base = (features['base_rsi'] > 25) & (features['base_rsi'] < 75)
    rsi_neutral_alt = (features['alt_rsi'] > 25) & (features['alt_rsi'] < 75)
    signals['rsi_filter'] = rsi_neutral_base & rsi_neutral_alt

    # 5. Volatility-based position sizing
    vol_percentile = features['volatility_20'].rolling(100).rank(pct=True)
    signals['vol_sizing'] = np.where(
        vol_percentile > 0.8, 0.5,  # Reduce size in high vol
        np.where(vol_percentile < 0.2, 1.2, 1.0)  # Increase size in low vol
    )

    # 6. Bollinger Bands mean reversion
    bb_oversold = features['bb_position'] < 0.1
    bb_overbought = features['bb_position'] > 0.9
    signals['bb_reversion'] = np.where(bb_oversold, 1, np.where(bb_overbought, -1, 0))

    # 7. Volume confirmation
    volume_trend = features['vwap_ratio'] > features['vwap_ratio'].shift(1)
    signals['volume_confirm'] = volume_trend

    return signals


def backtest_enhanced_momentum(base_data: pd.DataFrame, alt_data: pd.DataFrame,
                              ema_window: int = 10, allocation_weight: float = 0.75,
                              fees: float = 0.0004, slippage: float = 0.0005,
                              freq: int = 365, base_funding_data: pd.DataFrame = None,
                              alt_funding_data: pd.DataFrame = None,
                              enhancement_level: str = "moderate") -> Dict[str, Any]:
    """
    Backtest enhanced relative momentum strategy

    Args:
        enhancement_level: "conservative", "moderate", "aggressive"
    """

    # Align data
    common_idx = base_data.index.intersection(alt_data.index)
    base_aligned = base_data.loc[common_idx]
    alt_aligned = alt_data.loc[common_idx]

    if len(common_idx) < ema_window + 50:
        logger.warning("Insufficient data for enhanced momentum strategy")
        return {'returns': pd.Series(dtype=float), 'final_performance': 1.0, 'ema_window': ema_window}

    # Calculate features and signals
    base_close = base_aligned['close']
    alt_close = alt_aligned['close']
    ratio = base_close / alt_close

    features = calculate_advanced_features(base_aligned, alt_aligned)
    signals = generate_enhanced_signals(features, ratio, ema_window)

    # Strategy parameters based on enhancement level
    if enhancement_level == "conservative":
        signal_threshold = 0.7
        vol_adjustment = 0.8
        use_filters = ["trend_filter", "rsi_filter"]
    elif enhancement_level == "aggressive":
        signal_threshold = 0.3
        vol_adjustment = 1.2
        use_filters = ["trend_filter"]
    else:  # moderate
        signal_threshold = 0.5
        vol_adjustment = 1.0
        use_filters = ["trend_filter", "rsi_filter"]

    positions = []
    trades = []

    for i in range(50, len(common_idx)):  # Start after warmup period
        current_idx = common_idx[i]

        # Primary momentum signal
        momentum_signal = signals['momentum'].iloc[i]

        # Apply filters
        signal_strength = 1.0
        for filter_name in use_filters:
            if filter_name in signals and not pd.isna(signals[filter_name].iloc[i]):
                if not signals[filter_name].iloc[i]:
                    signal_strength *= 0.5  # Reduce signal strength

        # Multi-timeframe confirmation
        if 'multi_tf_bull' in signals and 'multi_tf_bear' in signals:
            if momentum_signal > 0 and signals['multi_tf_bull'].iloc[i]:
                signal_strength *= 1.2
            elif momentum_signal < 0 and signals['multi_tf_bear'].iloc[i]:
                signal_strength *= 1.2
            else:
                signal_strength *= 0.8

        # Volume confirmation
        if 'volume_confirm' in signals and not pd.isna(signals['volume_confirm'].iloc[i]):
            if (momentum_signal > 0 and signals['volume_confirm'].iloc[i]) or \
               (momentum_signal < 0 and not signals['volume_confirm'].iloc[i]):
                signal_strength *= 1.1

        # Volatility-based position sizing
        vol_sizing = signals['vol_sizing'].iloc[i] if not pd.isna(signals['vol_sizing'].iloc[i]) else 1.0
        vol_sizing *= vol_adjustment

        # Mean reversion override for extreme conditions
        if 'bb_reversion' in signals and signals['bb_reversion'].iloc[i] != 0:
            bb_signal = signals['bb_reversion'].iloc[i]
            # Override momentum signal if BB signal is strong and opposite
            if abs(bb_signal) > 0 and np.sign(bb_signal) != np.sign(momentum_signal):
                if abs(features['bb_position'].iloc[i] - 0.5) > 0.4:  # Extreme BB position
                    momentum_signal = bb_signal
                    signal_strength = 0.8  # Moderate confidence in mean reversion

        # Final position calculation
        if signal_strength >= signal_threshold:
            target_position = momentum_signal * allocation_weight * signal_strength * vol_sizing
            target_position = np.clip(target_position, -allocation_weight * 1.5, allocation_weight * 1.5)
        else:
            target_position = 0

        positions.append({
            'date': current_idx,
            'position': target_position,
            'momentum_signal': momentum_signal,
            'signal_strength': signal_strength,
            'vol_sizing': vol_sizing
        })

    if not positions:
        return {'returns': pd.Series(dtype=float), 'final_performance': 1.0, 'ema_window': ema_window}

    # Convert to DataFrame
    positions_df = pd.DataFrame(positions).set_index('date')

    # Calculate returns
    returns = calculate_enhanced_returns(
        positions_df, ratio, base_aligned, alt_aligned,
        fees, slippage, base_funding_data, alt_funding_data
    )

    # Calculate performance metrics
    ann_return, ann_vol, sharpe, max_dd = compute_metrics(returns, freq)
    final_performance = (1 + returns).prod()

    return {
        'returns': returns,
        'final_performance': final_performance,
        'ann_return': ann_return,
        'ann_vol': ann_vol,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'ema_window': ema_window,
        'enhancement_level': enhancement_level,
        'positions': positions_df,
        'features': features,
        'optimizations_enabled': True
    }


def calculate_enhanced_returns(positions_df: pd.DataFrame, ratio: pd.Series,
                              base_data: pd.DataFrame, alt_data: pd.DataFrame,
                              fees: float, slippage: float,
                              base_funding_data: pd.DataFrame = None,
                              alt_funding_data: pd.DataFrame = None) -> pd.Series:
    """Calculate strategy returns with enhanced cost calculation"""

    returns = []
    prev_position = 0

    for i in range(len(positions_df)):
        current_position = positions_df['position'].iloc[i]
        current_date = positions_df.index[i]

        if i > 0:
            # Calculate return from ratio change
            ratio_return = ratio.pct_change().loc[current_date]
            position_return = prev_position * ratio_return

            # Enhanced transaction costs (higher for larger position changes)
            position_change = abs(current_position - prev_position)
            size_penalty = 1 + (position_change * 0.1)  # Penalty for large position changes
            transaction_cost = position_change * (fees + slippage) * size_penalty

            # Funding costs
            funding_cost = 0
            if base_funding_data is not None and alt_funding_data is not None:
                try:
                    base_funding_mask = base_funding_data.index <= current_date
                    alt_funding_mask = alt_funding_data.index <= current_date

                    if base_funding_mask.any() and alt_funding_mask.any():
                        base_funding = base_funding_data.loc[base_funding_mask, 'funding_rate'].iloc[-1]
                        alt_funding = alt_funding_data.loc[alt_funding_mask, 'funding_rate'].iloc[-1]
                        funding_cost = abs(prev_position) * (base_funding - alt_funding) / (365 * 3)
                except:
                    funding_cost = 0

            total_return = position_return - transaction_cost - funding_cost
            returns.append(total_return)
        else:
            returns.append(0)

        prev_position = current_position

    return pd.Series(returns, index=positions_df.index)


def optimize_enhanced_strategy(base_data: pd.DataFrame, alt_data: pd.DataFrame,
                              ema_windows: List[int], allocation_weight: float = 0.75,
                              fees: float = 0.0004, slippage: float = 0.0005,
                              freq: int = 365, base_funding_data: pd.DataFrame = None,
                              alt_funding_data: pd.DataFrame = None,
                              optimization_metric: str = 'sharpe') -> Dict[str, Any]:
    """Optimize enhanced strategy parameters"""

    best_result = None
    best_score = -np.inf

    enhancement_levels = ["conservative", "moderate", "aggressive"]

    for ema_window in ema_windows:
        for enhancement_level in enhancement_levels:
            try:
                result = backtest_enhanced_momentum(
                    base_data, alt_data, ema_window, allocation_weight,
                    fees, slippage, freq, base_funding_data, alt_funding_data,
                    enhancement_level
                )

                if optimization_metric == 'sharpe':
                    score = result['sharpe']
                elif optimization_metric == 'calmar':
                    score = result['ann_return'] / abs(result['max_dd']) if result['max_dd'] != 0 else 0
                else:
                    score = result['sharpe']

                if score > best_score:
                    best_score = score
                    best_result = result

            except Exception as e:
                logger.warning(f"Error optimizing EMA {ema_window}, level {enhancement_level}: {e}")
                continue

    return best_result if best_result else {
        'returns': pd.Series(dtype=float),
        'final_performance': 1.0,
        'ema_window': ema_windows[0] if ema_windows else 10,
        'enhancement_level': 'moderate'
    }