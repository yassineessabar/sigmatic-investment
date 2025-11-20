import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.backtest.portfolio import PortfolioState, Position, Trade
from src.backtest.engine import BacktestEngine
from src.strategies.neutral_pairs import compute_signals
from src.utils.config import create_default_config


@pytest.fixture
def portfolio():
    return PortfolioState(initial_capital=100000)


@pytest.fixture
def sample_trade():
    return Trade(
        symbol='BTCUSDT',
        side='buy',
        size=0.5,
        price=20000,
        timestamp=datetime.now(),
        commission=10.0
    )


@pytest.fixture
def market_data():
    dates = pd.date_range(start='2023-01-01', periods=100, freq='1H')

    return {
        'BTCUSDT': pd.DataFrame({
            'symbol': 'BTCUSDT',
            'open': 20000,
            'high': 20100,
            'low': 19900,
            'close': 20000,
            'volume': 100
        }, index=dates),
        'ETHUSDT': pd.DataFrame({
            'symbol': 'ETHUSDT',
            'open': 1500,
            'high': 1510,
            'low': 1490,
            'close': 1500,
            'volume': 100
        }, index=dates)
    }


def test_portfolio_initialization(portfolio):
    assert portfolio.initial_capital == 100000
    assert portfolio.cash == 100000
    assert len(portfolio.positions) == 0
    assert len(portfolio.trades) == 0
    assert portfolio.total_equity == 100000


def test_portfolio_buy_trade(portfolio, sample_trade):
    portfolio._execute_buy(sample_trade)

    assert portfolio.cash == 100000 - (0.5 * 20000 + 10.0)
    assert 'BTCUSDT' in portfolio.positions
    assert portfolio.positions['BTCUSDT'].size == 0.5
    assert portfolio.positions['BTCUSDT'].side == 'long'


def test_portfolio_sell_trade(portfolio):
    sell_trade = Trade(
        symbol='BTCUSDT',
        side='sell',
        size=0.5,
        price=20000,
        timestamp=datetime.now(),
        commission=10.0
    )

    portfolio._execute_sell(sell_trade)

    expected_cash = 100000 + (0.5 * 20000 - 10.0)
    assert portfolio.cash == expected_cash
    assert 'BTCUSDT' in portfolio.positions
    assert portfolio.positions['BTCUSDT'].side == 'short'


def test_position_update():
    position = Position(
        symbol='BTCUSDT',
        size=1.0,
        entry_price=20000,
        entry_time=datetime.now(),
        side='long'
    )

    position.update_price(21000)
    assert position.current_price == 21000
    assert position.unrealized_pnl == 1000  # (21000 - 20000) * 1.0

    position = Position(
        symbol='BTCUSDT',
        size=1.0,
        entry_price=20000,
        entry_time=datetime.now(),
        side='short'
    )

    position.update_price(19000)
    assert position.unrealized_pnl == 1000  # (20000 - 19000) * 1.0


def test_portfolio_update_positions(portfolio, market_data):
    portfolio.positions['BTCUSDT'] = Position(
        symbol='BTCUSDT',
        size=1.0,
        entry_price=19000,
        entry_time=datetime.now(),
        side='long'
    )

    timestamp = datetime.now()
    portfolio.update_positions(market_data, timestamp)

    assert portfolio.positions['BTCUSDT'].current_price == 20000
    assert portfolio.positions['BTCUSDT'].unrealized_pnl == 1000
    assert len(portfolio.equity_curve) == 1


def test_portfolio_metrics_calculation(portfolio):
    for i in range(10):
        timestamp = datetime.now() + timedelta(days=i)
        equity = 100000 + i * 1000
        portfolio.equity_curve.append((timestamp, equity))

        if i > 0:
            prev_equity = 100000 + (i-1) * 1000
            daily_return = (equity - prev_equity) / prev_equity
            portfolio.daily_returns.append(daily_return)

    metrics = portfolio.get_results()

    assert metrics.total_return > 0
    assert isinstance(metrics.sharpe_ratio, float)
    assert isinstance(metrics.max_drawdown, float)


def test_backtest_engine_initialization():
    config = create_default_config()
    engine = BacktestEngine(config)

    assert engine.config == config
    assert isinstance(engine.portfolio, PortfolioState)
    assert engine.portfolio.initial_capital == config['backtest']['initial_capital']


def test_long_short_position_interaction(portfolio):
    buy_trade = Trade(
        symbol='BTCUSDT',
        side='buy',
        size=1.0,
        price=20000,
        timestamp=datetime.now(),
        commission=10.0
    )

    sell_trade = Trade(
        symbol='BTCUSDT',
        side='sell',
        size=0.5,
        price=21000,
        timestamp=datetime.now(),
        commission=10.0
    )

    portfolio._execute_buy(buy_trade)
    initial_cash = portfolio.cash

    portfolio._execute_sell(sell_trade)

    assert portfolio.positions['BTCUSDT'].size == 0.5
    assert portfolio.positions['BTCUSDT'].side == 'long'

    cash_change = 0.5 * 21000 - 10.0  # Proceeds from sale minus commission
    assert portfolio.cash == initial_cash + cash_change


def test_portfolio_gross_net_exposure(portfolio):
    portfolio.positions['BTCUSDT'] = Position(
        symbol='BTCUSDT',
        size=1.0,
        entry_price=20000,
        entry_time=datetime.now(),
        current_price=20000,
        side='long'
    )

    portfolio.positions['ETHUSDT'] = Position(
        symbol='ETHUSDT',
        size=2.0,
        entry_price=1500,
        entry_time=datetime.now(),
        current_price=1500,
        side='short'
    )

    gross_exposure = portfolio.gross_exposure
    net_exposure = portfolio.net_exposure

    expected_gross = 1.0 * 20000 + 2.0 * 1500
    expected_net = 1.0 * 20000 - 2.0 * 1500

    assert gross_exposure == expected_gross
    assert net_exposure == expected_net