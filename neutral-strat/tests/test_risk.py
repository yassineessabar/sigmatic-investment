import pytest
import numpy as np
from datetime import datetime, timedelta
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.risk.risk_checks import (
    enforce_risk_limits,
    calculate_position_size,
    validate_order_risk,
    RiskMetrics,
    RiskMonitor
)
from src.utils.config import create_default_config


@pytest.fixture
def risk_config():
    return {
        'risk': {
            'max_daily_dd': 0.05,
            'max_total_dd': 0.10,
            'leverage_limit': 2.0,
            'max_position_size': 0.1,
            'max_single_order': 0.05,
            'volatility_target': 0.15
        }
    }


@pytest.fixture
def mock_portfolio_state():
    class MockPortfolioState:
        def __init__(self):
            self.daily_dd = 0.02
            self.total_equity = 100000
            self.gross_exposure = 150000
            self.unrealized_pnl = 1000

    return MockPortfolioState()


def test_enforce_risk_limits_normal(mock_portfolio_state, risk_config):
    result = enforce_risk_limits(mock_portfolio_state, risk_config)
    assert result == "OK"


def test_enforce_risk_limits_daily_drawdown(mock_portfolio_state, risk_config):
    mock_portfolio_state.daily_dd = 0.08  # Exceeds 5% limit
    result = enforce_risk_limits(mock_portfolio_state, risk_config)
    assert result == "STOP_TRADING"


def test_enforce_risk_limits_leverage(mock_portfolio_state, risk_config):
    mock_portfolio_state.gross_exposure = 250000  # 2.5x leverage, exceeds 2.0x limit
    result = enforce_risk_limits(mock_portfolio_state, risk_config)
    assert result == "REDUCE_POSITION"


def test_calculate_position_size_basic(risk_config):
    position_size = calculate_position_size(
        signal_confidence=1.0,
        portfolio_equity=100000,
        volatility=0.15,
        config=risk_config
    )

    max_position = risk_config['risk']['max_position_size']
    assert 0 < position_size <= max_position


def test_calculate_position_size_low_confidence(risk_config):
    low_conf_size = calculate_position_size(
        signal_confidence=0.5,
        portfolio_equity=100000,
        volatility=0.15,
        config=risk_config
    )

    high_conf_size = calculate_position_size(
        signal_confidence=1.0,
        portfolio_equity=100000,
        volatility=0.15,
        config=risk_config
    )

    assert low_conf_size < high_conf_size


def test_calculate_position_size_high_volatility(risk_config):
    low_vol_size = calculate_position_size(
        signal_confidence=1.0,
        portfolio_equity=100000,
        volatility=0.05,  # Low volatility
        config=risk_config
    )

    high_vol_size = calculate_position_size(
        signal_confidence=1.0,
        portfolio_equity=100000,
        volatility=0.30,  # High volatility
        config=risk_config
    )

    assert low_vol_size > high_vol_size


def test_validate_order_risk_normal(risk_config):
    current_positions = {}

    is_valid, message = validate_order_risk(
        order_symbol='BTCUSDT',
        order_size=0.03,  # 3%, within limits
        order_side='buy',
        current_positions=current_positions,
        config=risk_config
    )

    assert is_valid
    assert message == "OK"


def test_validate_order_risk_oversized(risk_config):
    current_positions = {}

    is_valid, message = validate_order_risk(
        order_symbol='BTCUSDT',
        order_size=0.08,  # 8%, exceeds 5% single order limit
        order_side='buy',
        current_positions=current_positions,
        config=risk_config
    )

    assert not is_valid
    assert "single order limit" in message


def test_validate_order_risk_position_limit(risk_config):
    current_positions = {
        'BTCUSDT': {
            'size': 0.08,
            'side': 'long'
        }
    }

    is_valid, message = validate_order_risk(
        order_symbol='BTCUSDT',
        order_size=0.05,  # Would result in 13% position, exceeds 10% limit
        order_side='buy',
        current_positions=current_positions,
        config=risk_config
    )

    assert not is_valid
    assert "position size" in message


def test_risk_metrics_initialization():
    metrics = RiskMetrics()

    assert len(metrics.daily_returns) == 0
    assert len(metrics.equity_history) == 0
    assert metrics.max_drawdown_start is None


def test_risk_metrics_update():
    metrics = RiskMetrics()

    timestamp1 = datetime.now()
    timestamp2 = timestamp1 + timedelta(days=1)

    metrics.update(timestamp1, 100000)
    metrics.update(timestamp2, 105000)

    assert len(metrics.equity_history) == 2
    assert len(metrics.daily_returns) == 1
    assert metrics.daily_returns[0] == 0.05  # 5% return


def test_risk_metrics_sharpe_calculation():
    metrics = RiskMetrics()

    # Add some sample returns
    for i in range(30):
        timestamp = datetime.now() + timedelta(days=i)
        equity = 100000 * (1.01 ** i)  # 1% daily returns
        metrics.update(timestamp, equity)

    sharpe = metrics.calculate_sharpe_ratio()
    assert isinstance(sharpe, float)
    assert sharpe > 0  # Positive returns should give positive Sharpe


def test_risk_metrics_max_drawdown():
    metrics = RiskMetrics()

    equities = [100000, 105000, 98000, 95000, 102000]  # Peak at 105k, trough at 95k
    for i, equity in enumerate(equities):
        timestamp = datetime.now() + timedelta(days=i)
        metrics.update(timestamp, equity)

    max_dd, start_date, end_date = metrics.calculate_max_drawdown()

    expected_dd = (105000 - 95000) / 105000  # ~9.5%
    assert abs(max_dd - expected_dd) < 0.001
    assert start_date is not None
    assert end_date is not None


def test_risk_monitor_initialization(risk_config):
    monitor = RiskMonitor(risk_config)

    assert monitor.config == risk_config
    assert isinstance(monitor.risk_metrics, RiskMetrics)
    assert not monitor.trading_halted


def test_risk_monitor_halt_functionality(risk_config, mock_portfolio_state):
    monitor = RiskMonitor(risk_config)

    mock_portfolio_state.daily_dd = 0.08  # Exceeds limit
    risk_status = monitor.check_risk_limits(mock_portfolio_state)

    assert risk_status == "STOP_TRADING"
    assert monitor.trading_halted

    monitor.reset_halt()
    assert not monitor.trading_halted


def test_risk_monitor_report(risk_config):
    monitor = RiskMonitor(risk_config)

    # Add some sample data
    for i in range(10):
        timestamp = datetime.now() + timedelta(days=i)
        equity = 100000 + i * 1000
        monitor.risk_metrics.update(timestamp, equity)

    report = monitor.get_risk_report()

    required_keys = [
        'max_drawdown', 'sharpe_ratio', 'var_95',
        'trading_halted', 'last_check'
    ]

    for key in required_keys:
        assert key in report