import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.strategies.neutral_pairs import compute_signals, compute_spread_zscore, validate_signals
from src.utils.config import create_default_config


@pytest.fixture
def sample_config():
    return create_default_config()


@pytest.fixture
def sample_data():
    dates = pd.date_range(start='2023-01-01', periods=200, freq='1H')

    np.random.seed(42)
    btc_prices = 20000 + np.cumsum(np.random.randn(200) * 100)
    eth_prices = 1500 + np.cumsum(np.random.randn(200) * 50)

    btc_data = pd.DataFrame({
        'symbol': 'BTCUSDT',
        'open': btc_prices * 0.999,
        'high': btc_prices * 1.002,
        'low': btc_prices * 0.998,
        'close': btc_prices,
        'volume': np.random.rand(200) * 1000
    }, index=dates)

    eth_data = pd.DataFrame({
        'symbol': 'ETHUSDT',
        'open': eth_prices * 0.999,
        'high': eth_prices * 1.002,
        'low': eth_prices * 0.998,
        'close': eth_prices,
        'volume': np.random.rand(200) * 1000
    }, index=dates)

    return {
        'BTCUSDT': btc_data,
        'ETHUSDT': eth_data
    }


def test_compute_spread_zscore(sample_data):
    btc_prices = sample_data['BTCUSDT']['close']
    eth_prices = sample_data['ETHUSDT']['close']

    zscore, spread, std = compute_spread_zscore(btc_prices, eth_prices, lookback=50)

    assert not np.isnan(zscore)
    assert not np.isnan(spread)
    assert std >= 0
    assert isinstance(zscore, float)


def test_compute_signals_basic(sample_data, sample_config):
    signals = compute_signals(sample_data, sample_config)

    assert isinstance(signals, list)

    for signal in signals:
        assert hasattr(signal, 'base_signal')
        assert hasattr(signal, 'hedge_signal')
        assert hasattr(signal, 'pair_name')
        assert hasattr(signal, 'spread_zscore')


def test_compute_signals_no_data():
    empty_data = {}
    config = create_default_config()

    signals = compute_signals(empty_data, config)
    assert len(signals) == 0


def test_compute_signals_insufficient_data(sample_config):
    dates = pd.date_range(start='2023-01-01', periods=10, freq='1H')

    btc_data = pd.DataFrame({
        'symbol': 'BTCUSDT',
        'open': [20000] * 10,
        'high': [20100] * 10,
        'low': [19900] * 10,
        'close': [20000] * 10,
        'volume': [100] * 10
    }, index=dates)

    insufficient_data = {'BTCUSDT': btc_data}

    signals = compute_signals(insufficient_data, sample_config)
    assert len(signals) == 0


def test_validate_signals(sample_data, sample_config):
    signals = compute_signals(sample_data, sample_config)
    validated_signals = validate_signals(signals, sample_config)

    assert isinstance(validated_signals, list)
    assert len(validated_signals) <= len(signals)


def test_signal_entry_conditions(sample_data, sample_config):
    config = sample_config.copy()
    config['pairs'][0]['entry_z'] = 0.5  # Very low threshold for testing

    signals = compute_signals(sample_data, config)

    if signals:
        signal = signals[0]
        assert abs(signal.spread_zscore) >= 0.5
        assert signal.base_signal.side in ['long', 'short', 'close']
        assert signal.hedge_signal.side in ['long', 'short', 'close']


def test_signal_size_constraints(sample_data, sample_config):
    signals = compute_signals(sample_data, sample_config)

    for signal in signals:
        max_notional = sample_config['pairs'][0]['max_notional']
        base_value = signal.base_signal.size * signal.base_signal.price
        hedge_value = signal.hedge_signal.size * signal.hedge_signal.price

        if signal.base_signal.side != 'close':
            assert base_value <= max_notional
        if signal.hedge_signal.side != 'close':
            assert hedge_value <= max_notional


def test_opposite_positions():
    dates = pd.date_range(start='2023-01-01', periods=150, freq='1H')

    np.random.seed(123)
    base_returns = np.random.randn(150) * 0.01
    hedge_returns = -base_returns + np.random.randn(150) * 0.005  # Negatively correlated

    base_prices = 1000 * (1 + base_returns).cumprod()
    hedge_prices = 1000 * (1 + hedge_returns).cumprod()

    data = {
        'BTCUSDT': pd.DataFrame({
            'symbol': 'BTCUSDT',
            'open': base_prices,
            'high': base_prices * 1.01,
            'low': base_prices * 0.99,
            'close': base_prices,
            'volume': 100
        }, index=dates),
        'ETHUSDT': pd.DataFrame({
            'symbol': 'ETHUSDT',
            'open': hedge_prices,
            'high': hedge_prices * 1.01,
            'low': hedge_prices * 0.99,
            'close': hedge_prices,
            'volume': 100
        }, index=dates)
    }

    config = create_default_config()
    config['pairs'][0]['entry_z'] = 1.0

    signals = compute_signals(data, config)

    if signals:
        signal = signals[0]
        base_side = signal.base_signal.side
        hedge_side = signal.hedge_signal.side

        if base_side == 'long' and hedge_side == 'short':
            assert True  # Correct opposing positions
        elif base_side == 'short' and hedge_side == 'long':
            assert True  # Correct opposing positions
        elif base_side == 'close' and hedge_side == 'close':
            assert True  # Both closing positions
        else:
            assert False, f"Invalid position combination: {base_side}, {hedge_side}"