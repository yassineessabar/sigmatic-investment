from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class Position:
    symbol: str
    size: float
    entry_price: float
    entry_time: datetime
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    side: str = 'long'  # 'long' or 'short'

    def update_price(self, new_price: float):
        self.current_price = new_price
        if self.side == 'long':
            self.unrealized_pnl = (new_price - self.entry_price) * self.size
        else:  # short
            self.unrealized_pnl = (self.entry_price - new_price) * self.size


@dataclass
class Trade:
    symbol: str
    side: str
    size: float
    price: float
    timestamp: datetime
    trade_id: str = ""
    commission: float = 0.0
    metadata: Optional[Dict] = None


@dataclass
class PortfolioMetrics:
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0
    avg_trade_duration: float = 0.0


class PortfolioState:
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_curve: List[Tuple[datetime, float]] = []
        self.daily_returns: List[float] = []

    @property
    def total_equity(self) -> float:
        position_value = sum(pos.size * pos.current_price for pos in self.positions.values())
        return self.cash + position_value

    @property
    def unrealized_pnl(self) -> float:
        return sum(pos.unrealized_pnl for pos in self.positions.values())

    @property
    def gross_exposure(self) -> float:
        return sum(abs(pos.size * pos.current_price) for pos in self.positions.values())

    @property
    def net_exposure(self) -> float:
        long_exposure = sum(
            pos.size * pos.current_price
            for pos in self.positions.values()
            if pos.side == 'long'
        )
        short_exposure = sum(
            abs(pos.size * pos.current_price)
            for pos in self.positions.values()
            if pos.side == 'short'
        )
        return long_exposure - short_exposure

    def update_positions(self, market_data: Dict[str, pd.DataFrame], timestamp: datetime):
        for symbol, position in self.positions.items():
            if symbol in market_data:
                current_price = market_data[symbol]['close'].iloc[-1]
                position.update_price(current_price)

        equity = self.total_equity
        self.equity_curve.append((timestamp, equity))

        if len(self.equity_curve) > 1:
            prev_equity = self.equity_curve[-2][1]
            daily_return = (equity - prev_equity) / prev_equity
            self.daily_returns.append(daily_return)

    def compute_trades(self, signals, config: Dict) -> List[Trade]:
        trades = []
        current_time = datetime.now()

        for signal in signals:
            base_signal = signal.base_signal
            hedge_signal = signal.hedge_signal

            base_trade = self._create_trade_from_signal(base_signal, current_time, config)
            hedge_trade = self._create_trade_from_signal(hedge_signal, current_time, config)

            if base_trade:
                trades.append(base_trade)
            if hedge_trade:
                trades.append(hedge_trade)

        return trades

    def _create_trade_from_signal(self, signal, timestamp: datetime, config: Dict) -> Optional[Trade]:
        commission_rate = config.get('execution', {}).get('fees', 0.001)

        if signal.side == 'close':
            if signal.symbol in self.positions:
                position = self.positions[signal.symbol]
                size = position.size
                side = 'sell' if position.side == 'long' else 'buy'

                return Trade(
                    symbol=signal.symbol,
                    side=side,
                    size=size,
                    price=signal.price,
                    timestamp=timestamp,
                    commission=size * signal.price * commission_rate,
                    metadata=signal.metadata
                )
        else:
            side = 'buy' if signal.side == 'long' else 'sell'
            commission = signal.size * signal.price * commission_rate

            return Trade(
                symbol=signal.symbol,
                side=side,
                size=signal.size,
                price=signal.price,
                timestamp=timestamp,
                commission=commission,
                metadata=signal.metadata
            )

        return None

    def apply_trades(self, trades: List[Trade], market_data: Dict[str, pd.DataFrame], config: Dict):
        slippage_rate = config.get('execution', {}).get('slippage', 0.001)

        for trade in trades:
            adjusted_price = trade.price
            if trade.side == 'buy':
                adjusted_price *= (1 + slippage_rate)
            else:
                adjusted_price *= (1 - slippage_rate)

            trade_value = trade.size * adjusted_price
            total_cost = trade_value + trade.commission

            if trade.side == 'buy':
                if self.cash >= total_cost:
                    self._execute_buy(trade, adjusted_price)
                else:
                    logger.warning(f"Insufficient cash for trade: {trade.symbol}")
            else:
                self._execute_sell(trade, adjusted_price)

            self.trades.append(trade)

    def _execute_buy(self, trade: Trade, price: float):
        total_cost = trade.size * price + trade.commission
        self.cash -= total_cost

        if trade.symbol in self.positions:
            existing_pos = self.positions[trade.symbol]
            if existing_pos.side == 'short':
                if trade.size >= existing_pos.size:
                    remaining_size = trade.size - existing_pos.size
                    del self.positions[trade.symbol]
                    if remaining_size > 0:
                        self.positions[trade.symbol] = Position(
                            symbol=trade.symbol,
                            size=remaining_size,
                            entry_price=price,
                            entry_time=trade.timestamp,
                            current_price=price,
                            side='long'
                        )
                else:
                    existing_pos.size -= trade.size
            else:
                existing_pos.size += trade.size
                avg_price = (existing_pos.entry_price * (existing_pos.size - trade.size) +
                           price * trade.size) / existing_pos.size
                existing_pos.entry_price = avg_price
        else:
            self.positions[trade.symbol] = Position(
                symbol=trade.symbol,
                size=trade.size,
                entry_price=price,
                entry_time=trade.timestamp,
                current_price=price,
                side='long'
            )

    def _execute_sell(self, trade: Trade, price: float):
        trade_value = trade.size * price
        self.cash += trade_value - trade.commission

        if trade.symbol in self.positions:
            existing_pos = self.positions[trade.symbol]
            if existing_pos.side == 'long':
                if trade.size >= existing_pos.size:
                    remaining_size = trade.size - existing_pos.size
                    del self.positions[trade.symbol]
                    if remaining_size > 0:
                        self.positions[trade.symbol] = Position(
                            symbol=trade.symbol,
                            size=remaining_size,
                            entry_price=price,
                            entry_time=trade.timestamp,
                            current_price=price,
                            side='short'
                        )
                else:
                    existing_pos.size -= trade.size
            else:
                existing_pos.size += trade.size
                avg_price = (existing_pos.entry_price * (existing_pos.size - trade.size) +
                           price * trade.size) / existing_pos.size
                existing_pos.entry_price = avg_price
        else:
            self.positions[trade.symbol] = Position(
                symbol=trade.symbol,
                size=trade.size,
                entry_price=price,
                entry_time=trade.timestamp,
                current_price=price,
                side='short'
            )

    def get_results(self) -> PortfolioMetrics:
        if len(self.equity_curve) < 2:
            return PortfolioMetrics()

        final_equity = self.equity_curve[-1][1]
        total_return = (final_equity - self.initial_capital) / self.initial_capital

        if len(self.daily_returns) > 0:
            returns_array = np.array(self.daily_returns)
            sharpe_ratio = np.mean(returns_array) / np.std(returns_array) * np.sqrt(252) if np.std(returns_array) > 0 else 0

            cumulative_returns = np.cumprod(1 + returns_array)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = (cumulative_returns - running_max) / running_max
            max_drawdown = np.min(drawdowns)
        else:
            sharpe_ratio = 0
            max_drawdown = 0

        winning_trades = [t for t in self.trades if self._calculate_trade_pnl(t) > 0]
        total_trades = len(self.trades)
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0

        gross_profit = sum(self._calculate_trade_pnl(t) for t in self.trades if self._calculate_trade_pnl(t) > 0)
        gross_loss = abs(sum(self._calculate_trade_pnl(t) for t in self.trades if self._calculate_trade_pnl(t) < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        return PortfolioMetrics(
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=total_trades
        )

    def _calculate_trade_pnl(self, trade: Trade) -> float:
        return 0