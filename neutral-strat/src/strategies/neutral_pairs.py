from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class Signal:
    symbol: str
    side: str  # 'long', 'short', 'close'
    size: float
    price: float
    timestamp: pd.Timestamp
    confidence: float = 1.0
    metadata: Optional[Dict] = None


@dataclass
class PairSignal:
    base_signal: Signal
    hedge_signal: Signal
    pair_name: str
    spread_zscore: float
    entry_reason: str


def compute_spread_zscore(
    base_prices: pd.Series,
    hedge_prices: pd.Series,
    lookback: int
) -> Tuple[float, float, float]:
    if len(base_prices) < lookback or len(hedge_prices) < lookback:
        return np.nan, np.nan, np.nan

    base_returns = base_prices.pct_change()
    hedge_returns = hedge_prices.pct_change()

    spread = base_returns - hedge_returns
    spread_window = spread.tail(lookback)

    current_spread = spread.iloc[-1]
    mean_spread = spread_window.mean()
    std_spread = spread_window.std()

    if std_spread == 0:
        return np.nan, current_spread, 0

    zscore = (current_spread - mean_spread) / std_spread
    return zscore, current_spread, std_spread


def compute_signals(data: Dict[str, pd.DataFrame], config: Dict) -> List[PairSignal]:
    signals = []

    for pair_config in config['pairs']:
        base_symbol = pair_config['base']
        hedge_symbol = pair_config['hedge']
        lookback = pair_config['lookback']
        entry_z = pair_config['entry_z']
        exit_z = pair_config['exit_z']
        max_notional = pair_config['max_notional']

        if base_symbol not in data or hedge_symbol not in data:
            logger.warning(f"Missing data for pair {base_symbol}/{hedge_symbol}")
            continue

        base_data = data[base_symbol]
        hedge_data = data[hedge_symbol]

        if len(base_data) < lookback or len(hedge_data) < lookback:
            logger.warning(f"Insufficient data for pair {base_symbol}/{hedge_symbol}")
            continue

        base_prices = base_data['close']
        hedge_prices = hedge_data['close']

        zscore, spread, spread_std = compute_spread_zscore(
            base_prices, hedge_prices, lookback
        )

        if np.isnan(zscore):
            continue

        current_time = base_prices.index[-1]
        base_price = base_prices.iloc[-1]
        hedge_price = hedge_prices.iloc[-1]

        pair_name = f"{base_symbol}_{hedge_symbol}"

        if abs(zscore) >= entry_z:
            position_size = min(max_notional / base_price, max_notional / hedge_price) * 0.5

            if zscore > entry_z:
                base_signal = Signal(
                    symbol=base_symbol,
                    side='short',
                    size=position_size,
                    price=base_price,
                    timestamp=current_time,
                    confidence=min(abs(zscore) / entry_z, 2.0),
                    metadata={'zscore': zscore, 'spread': spread}
                )
                hedge_signal = Signal(
                    symbol=hedge_symbol,
                    side='long',
                    size=position_size,
                    price=hedge_price,
                    timestamp=current_time,
                    confidence=min(abs(zscore) / entry_z, 2.0),
                    metadata={'zscore': zscore, 'spread': spread}
                )
                entry_reason = f"Mean reversion: spread z-score {zscore:.2f} > {entry_z}"

            else:  # zscore < -entry_z
                base_signal = Signal(
                    symbol=base_symbol,
                    side='long',
                    size=position_size,
                    price=base_price,
                    timestamp=current_time,
                    confidence=min(abs(zscore) / entry_z, 2.0),
                    metadata={'zscore': zscore, 'spread': spread}
                )
                hedge_signal = Signal(
                    symbol=hedge_symbol,
                    side='short',
                    size=position_size,
                    price=hedge_price,
                    timestamp=current_time,
                    confidence=min(abs(zscore) / entry_z, 2.0),
                    metadata={'zscore': zscore, 'spread': spread}
                )
                entry_reason = f"Mean reversion: spread z-score {zscore:.2f} < -{entry_z}"

            pair_signal = PairSignal(
                base_signal=base_signal,
                hedge_signal=hedge_signal,
                pair_name=pair_name,
                spread_zscore=zscore,
                entry_reason=entry_reason
            )
            signals.append(pair_signal)

        elif abs(zscore) <= exit_z:
            close_signal_base = Signal(
                symbol=base_symbol,
                side='close',
                size=0,
                price=base_price,
                timestamp=current_time,
                confidence=1.0,
                metadata={'zscore': zscore, 'spread': spread}
            )
            close_signal_hedge = Signal(
                symbol=hedge_symbol,
                side='close',
                size=0,
                price=hedge_price,
                timestamp=current_time,
                confidence=1.0,
                metadata={'zscore': zscore, 'spread': spread}
            )

            pair_signal = PairSignal(
                base_signal=close_signal_base,
                hedge_signal=close_signal_hedge,
                pair_name=pair_name,
                spread_zscore=zscore,
                entry_reason=f"Exit signal: spread z-score {zscore:.2f} within exit threshold {exit_z}"
            )
            signals.append(pair_signal)

    return signals


def validate_signals(signals: List[PairSignal], config: Dict) -> List[PairSignal]:
    validated_signals = []

    for signal in signals:
        if not _validate_single_signal(signal, config):
            logger.warning(f"Signal validation failed for {signal.pair_name}")
            continue

        validated_signals.append(signal)

    return validated_signals


def _validate_single_signal(signal: PairSignal, config: Dict) -> bool:
    if signal.base_signal.size < 0 or signal.hedge_signal.size < 0:
        return False

    max_position_size = config.get('risk', {}).get('max_position_size', 0.1)
    if signal.base_signal.size > max_position_size or signal.hedge_signal.size > max_position_size:
        return False

    if np.isnan(signal.spread_zscore) or np.isinf(signal.spread_zscore):
        return False

    return True