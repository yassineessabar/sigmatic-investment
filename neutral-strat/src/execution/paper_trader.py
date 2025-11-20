from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from dataclasses import dataclass

from ..backtest.portfolio import Position, Trade

logger = logging.getLogger(__name__)


@dataclass
class PaperOrder:
    order_id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    size: float
    price: float
    order_type: str  # 'market' or 'limit'
    timestamp: datetime
    status: str = 'pending'  # 'pending', 'filled', 'cancelled'
    filled_size: float = 0.0
    filled_price: float = 0.0


class PaperTradingEngine:
    def __init__(self, initial_balance: float = 100000, config: Dict = None):
        self.config = config or {}
        self.cash_balance = initial_balance
        self.positions: Dict[str, Position] = {}
        self.orders: List[PaperOrder] = []
        self.trades: List[Trade] = []
        self.order_counter = 0

        self.slippage_rate = config.get('execution', {}).get('slippage', 0.001)
        self.commission_rate = config.get('execution', {}).get('fees', 0.001)

    def get_account_balance(self) -> Dict[str, float]:
        position_value = sum(
            pos.size * pos.current_price
            for pos in self.positions.values()
        )

        return {
            'cash': self.cash_balance,
            'positions_value': position_value,
            'total_equity': self.cash_balance + position_value,
            'unrealized_pnl': sum(pos.unrealized_pnl for pos in self.positions.values())
        }

    def get_positions(self) -> Dict[str, Dict]:
        position_data = {}
        for symbol, position in self.positions.items():
            position_data[symbol] = {
                'symbol': symbol,
                'size': position.size,
                'side': position.side,
                'entry_price': position.entry_price,
                'current_price': position.current_price,
                'unrealized_pnl': position.unrealized_pnl,
                'entry_time': position.entry_time
            }
        return position_data

    def place_order(
        self,
        symbol: str,
        side: str,
        size: float,
        order_type: str = 'market',
        price: Optional[float] = None
    ) -> PaperOrder:
        self.order_counter += 1
        order_id = f"paper_{self.order_counter:06d}"

        order = PaperOrder(
            order_id=order_id,
            symbol=symbol,
            side=side,
            size=size,
            price=price or 0.0,
            order_type=order_type,
            timestamp=datetime.now()
        )

        self.orders.append(order)
        logger.info(f"Paper order placed: {order_id} {side} {size} {symbol} @ {price}")

        return order

    def update_market_data(self, market_data: Dict[str, pd.DataFrame]):
        for symbol, data in market_data.items():
            current_price = data['close'].iloc[-1]

            if symbol in self.positions:
                self.positions[symbol].update_price(current_price)

            self._process_pending_orders(symbol, current_price, data)

    def _process_pending_orders(self, symbol: str, current_price: float, market_data: pd.DataFrame):
        pending_orders = [o for o in self.orders if o.symbol == symbol and o.status == 'pending']

        for order in pending_orders:
            if order.order_type == 'market':
                self._fill_market_order(order, current_price)
            elif order.order_type == 'limit':
                self._try_fill_limit_order(order, current_price, market_data)

    def _fill_market_order(self, order: PaperOrder, market_price: float):
        fill_price = self._apply_slippage(market_price, order.side)

        if self._validate_order_execution(order, fill_price):
            order.status = 'filled'
            order.filled_size = order.size
            order.filled_price = fill_price

            trade = Trade(
                symbol=order.symbol,
                side=order.side,
                size=order.size,
                price=fill_price,
                timestamp=order.timestamp,
                trade_id=order.order_id,
                commission=order.size * fill_price * self.commission_rate
            )

            self._execute_trade(trade)
            logger.info(f"Paper order filled: {order.order_id} @ {fill_price}")

    def _try_fill_limit_order(self, order: PaperOrder, current_price: float, market_data: pd.DataFrame):
        can_fill = False

        if order.side == 'buy' and current_price <= order.price:
            can_fill = True
        elif order.side == 'sell' and current_price >= order.price:
            can_fill = True

        if can_fill and self._validate_order_execution(order, order.price):
            order.status = 'filled'
            order.filled_size = order.size
            order.filled_price = order.price

            trade = Trade(
                symbol=order.symbol,
                side=order.side,
                size=order.size,
                price=order.price,
                timestamp=datetime.now(),
                trade_id=order.order_id,
                commission=order.size * order.price * self.commission_rate
            )

            self._execute_trade(trade)
            logger.info(f"Paper limit order filled: {order.order_id} @ {order.price}")

    def _apply_slippage(self, price: float, side: str) -> float:
        if side == 'buy':
            return price * (1 + self.slippage_rate)
        else:  # sell
            return price * (1 - self.slippage_rate)

    def _validate_order_execution(self, order: PaperOrder, execution_price: float) -> bool:
        if order.side == 'buy':
            required_cash = order.size * execution_price * (1 + self.commission_rate)
            return self.cash_balance >= required_cash
        else:  # sell
            if order.symbol in self.positions:
                position = self.positions[order.symbol]
                if position.side == 'long':
                    return position.size >= order.size
                else:  # short position
                    return True  # Can always add to short
            return True  # Can create new short position

    def _execute_trade(self, trade: Trade):
        self.trades.append(trade)

        if trade.side == 'buy':
            self._execute_buy(trade)
        else:
            self._execute_sell(trade)

    def _execute_buy(self, trade: Trade):
        total_cost = trade.size * trade.price + trade.commission
        self.cash_balance -= total_cost

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
                            entry_price=trade.price,
                            entry_time=trade.timestamp,
                            current_price=trade.price,
                            side='long'
                        )
                else:
                    existing_pos.size -= trade.size
            else:  # existing long position
                total_size = existing_pos.size + trade.size
                avg_price = (existing_pos.entry_price * existing_pos.size +
                           trade.price * trade.size) / total_size
                existing_pos.size = total_size
                existing_pos.entry_price = avg_price
        else:
            self.positions[trade.symbol] = Position(
                symbol=trade.symbol,
                size=trade.size,
                entry_price=trade.price,
                entry_time=trade.timestamp,
                current_price=trade.price,
                side='long'
            )

    def _execute_sell(self, trade: Trade):
        proceeds = trade.size * trade.price - trade.commission
        self.cash_balance += proceeds

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
                            entry_price=trade.price,
                            entry_time=trade.timestamp,
                            current_price=trade.price,
                            side='short'
                        )
                else:
                    existing_pos.size -= trade.size
            else:  # existing short position
                total_size = existing_pos.size + trade.size
                avg_price = (existing_pos.entry_price * existing_pos.size +
                           trade.price * trade.size) / total_size
                existing_pos.size = total_size
                existing_pos.entry_price = avg_price
        else:
            self.positions[trade.symbol] = Position(
                symbol=trade.symbol,
                size=trade.size,
                entry_price=trade.price,
                entry_time=trade.timestamp,
                current_price=trade.price,
                side='short'
            )

    def cancel_order(self, order_id: str) -> bool:
        for order in self.orders:
            if order.order_id == order_id and order.status == 'pending':
                order.status = 'cancelled'
                logger.info(f"Paper order cancelled: {order_id}")
                return True
        return False

    def get_open_orders(self) -> List[PaperOrder]:
        return [o for o in self.orders if o.status == 'pending']