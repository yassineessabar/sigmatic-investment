"""
Market Neutral Pairs Trading Strategy

This module implements a truly market-neutral pairs trading approach
designed to minimize correlation with Bitcoin and broader crypto markets.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass

from .neutral_pairs import Signal, PairSignal
from .relative_momentum import compute_metrics

logger = logging.getLogger(__name__)


def compute_market_neutral_signals(data: Dict[str, pd.DataFrame], config: Dict) -> List[PairSignal]:
    """
    Market neutral pairs strategy with correlation control and sector diversification.

    Key Features:
    1. Forces market neutral positioning (long/short balance)
    2. Minimizes correlation with BTC through dynamic hedging
    3. Uses cross-asset pairs for diversification
    4. Implements correlation-based position sizing
    """
    signals = []

    # Calculate BTC returns for correlation control
    btc_returns = data.get('BTCUSDT', pd.DataFrame()).get('close', pd.Series()).pct_change()

    for pair_config in config['pairs']:
        base_symbol = pair_config['base']
        alt_symbol = pair_config['alt']
        ema_window = pair_config['ema_window']
        allocation_weight = pair_config.get('allocation_weight', 0.5)
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

        # Calculate relative ratio and EMA
        relative_ratio = base_prices / alt_prices
        ema_ratio = relative_ratio.ewm(span=ema_window, adjust=False).mean()

        # Calculate returns for correlation analysis
        base_returns = base_prices.pct_change()
        alt_returns = alt_prices.pct_change()
        pair_returns = base_returns - alt_returns  # Pairs return

        # Market Neutral Enhancements
        if config.get('strategy', {}).get('enhancements', {}).get('enabled', True):

            # 1. Correlation-based position sizing
            correlation_config = config.get('strategy', {}).get('enhancements', {}).get('correlation_sizing', {})
            if correlation_config.get('enabled', False):
                lookback = correlation_config.get('lookback_period', 60)
                target_corr = correlation_config.get('target_correlation', 0.0)
                penalty = correlation_config.get('correlation_penalty', 2.0)

                # Calculate correlation with BTC
                if len(btc_returns) > lookback and len(pair_returns) > lookback:
                    rolling_corr = pair_returns.rolling(lookback).corr(btc_returns)
                    # Reduce allocation based on correlation
                    corr_adjustment = 1.0 - penalty * abs(rolling_corr - target_corr)
                    corr_adjustment = np.clip(corr_adjustment, 0.1, 1.0)
                    allocation_weight *= corr_adjustment.iloc[-1] if not pd.isna(corr_adjustment.iloc[-1]) else 1.0

            # 2. Market neutrality constraints
            neutral_config = config.get('strategy', {}).get('enhancements', {}).get('market_neutral_constraints', {})
            if neutral_config.get('enabled', False):
                # Force alternating long/short to maintain neutrality
                pair_id = hash(f"{base_symbol}_{alt_symbol}") % 2
                neutrality_multiplier = 1.0 if pair_id == 0 else -1.0

                # Apply beta targeting
                beta_target = neutral_config.get('beta_target', 0.0)
                if len(btc_returns) > 30 and len(pair_returns) > 30:
                    pair_beta = np.cov(pair_returns.dropna()[-30:], btc_returns.dropna()[-30:])[0, 1] / np.var(btc_returns.dropna()[-30:])
                    beta_adjustment = 1.0 - abs(pair_beta - beta_target)
                    allocation_weight *= max(0.1, beta_adjustment)

        current_time = base_prices.index[-1]
        base_price = base_prices.iloc[-1]
        alt_price = alt_prices.iloc[-1]

        pair_name = f"{base_symbol}_{alt_symbol}"

        # Market Neutral Signal Generation
        if len(relative_ratio) < 2 or len(ema_ratio) < 2:
            continue

        current_ratio = relative_ratio.iloc[-1]
        current_ema = ema_ratio.iloc[-1]
        prev_ratio = relative_ratio.iloc[-2]
        prev_ema = ema_ratio.iloc[-2]

        # Current and previous signals
        current_signal = 1 if current_ratio > current_ema else -1
        prev_signal = 1 if prev_ratio > prev_ema else -1

        # Only trade on signal changes to reduce correlation
        if current_signal != prev_signal:

            # Market neutral positioning - always long one, short the other
            if current_signal > 0:
                # Long base, short alt (market neutral)
                base_side = 'long'
                alt_side = 'short'
                base_weight = allocation_weight
                alt_weight = -allocation_weight  # Short position
            else:
                # Short base, long alt (market neutral)
                base_side = 'short'
                alt_side = 'long'
                base_weight = -allocation_weight  # Short position
                alt_weight = allocation_weight

            # Position sizing with market neutrality
            base_position_size = abs(base_weight) * max_notional / base_price
            alt_position_size = abs(alt_weight) * max_notional / alt_price

            # Calculate confidence based on signal strength and market neutrality
            ratio_distance = abs(current_ratio - current_ema) / current_ema

            # Reduce confidence if correlation with BTC is high
            if len(btc_returns) > 30 and len(pair_returns) > 30:
                recent_corr = pair_returns.tail(30).corr(btc_returns.tail(30))
                if not pd.isna(recent_corr):
                    correlation_penalty = abs(recent_corr) * 0.5  # Penalize correlation
                    ratio_distance *= (1.0 - correlation_penalty)

            confidence = min(ratio_distance * 10, 2.0)

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
                    'weight': base_weight,
                    'market_neutral': True,
                    'pair_type': 'market_neutral'
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
                    'weight': alt_weight,
                    'market_neutral': True,
                    'pair_type': 'market_neutral'
                }
            )

            entry_reason = f"Market Neutral: {base_symbol} {base_side}/{alt_symbol} {alt_side} (ratio {current_ratio:.6f} vs EMA {current_ema:.6f})"

            pair_signal = PairSignal(
                base_signal=base_signal,
                hedge_signal=alt_signal,
                pair_name=pair_name,
                spread_zscore=ratio_distance,
                entry_reason=entry_reason
            )

            signals.append(pair_signal)

    return signals


def calculate_market_neutral_metrics(returns: pd.Series, benchmark_returns: pd.Series, freq: int = 365) -> Dict:
    """
    Calculate market neutral specific metrics including correlation and beta analysis.
    """
    if isinstance(returns, pd.DataFrame):
        returns = returns.squeeze()
    if isinstance(benchmark_returns, pd.DataFrame):
        benchmark_returns = benchmark_returns.squeeze()

    returns = returns.dropna()
    benchmark_returns = benchmark_returns.dropna()

    if len(returns) == 0 or len(benchmark_returns) == 0:
        return {}

    # Align returns
    aligned_returns = returns.align(benchmark_returns, join='inner')
    strategy_ret = aligned_returns[0].dropna()
    benchmark_ret = aligned_returns[1].dropna()

    if len(strategy_ret) < 30:
        return {}

    # Standard metrics
    ann_return, ann_vol, sharpe, max_dd = compute_metrics(strategy_ret, freq)

    # Market neutral specific metrics
    correlation = strategy_ret.corr(benchmark_ret) if len(benchmark_ret) > 0 else np.nan

    # Calculate beta
    if len(benchmark_ret) > 0:
        covariance = np.cov(strategy_ret, benchmark_ret)[0, 1]
        benchmark_var = np.var(benchmark_ret)
        beta = covariance / benchmark_var if benchmark_var > 0 else np.nan
    else:
        beta = np.nan

    # Tracking error (volatility of excess returns)
    if len(benchmark_ret) > 0:
        excess_returns = strategy_ret - benchmark_ret
        tracking_error = np.std(excess_returns) * np.sqrt(freq)
    else:
        tracking_error = np.nan

    # Information ratio (excess return / tracking error)
    if len(benchmark_ret) > 0:
        excess_return = (np.mean(strategy_ret) - np.mean(benchmark_ret)) * freq
        information_ratio = excess_return / tracking_error if tracking_error > 0 else np.nan
    else:
        information_ratio = np.nan

    # Rolling correlation analysis
    rolling_corr = strategy_ret.rolling(30).corr(benchmark_ret) if len(benchmark_ret) > 0 else pd.Series()
    avg_correlation = rolling_corr.mean() if len(rolling_corr) > 0 else np.nan
    correlation_volatility = rolling_corr.std() if len(rolling_corr) > 0 else np.nan

    return {
        'ann_return': ann_return,
        'ann_vol': ann_vol,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'correlation': correlation,
        'beta': beta,
        'tracking_error': tracking_error,
        'information_ratio': information_ratio,
        'avg_correlation': avg_correlation,
        'correlation_volatility': correlation_volatility,
        'market_neutral_score': 1.0 - abs(correlation) if not pd.isna(correlation) else np.nan
    }