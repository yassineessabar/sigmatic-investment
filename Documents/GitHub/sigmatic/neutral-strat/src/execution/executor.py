from typing import Dict, List, Optional, Union
import logging
from datetime import datetime

from ...data.binance_client import BinanceDataClient
from .paper_trader import PaperTradingEngine, PaperOrder
from ..strategies.neutral_pairs import PairSignal
from ..risk.risk_checks import enforce_risk_limits

logger = logging.getLogger(__name__)


class ExecutionEngine:
    def __init__(self, binance_client: Optional[BinanceDataClient], config: Dict):
        self.binance_client = binance_client
        self.config = config
        self.mode = config.get('execution', {}).get('mode', 'paper')

        if self.mode == 'paper':
            initial_capital = config.get('backtest', {}).get('initial_capital', 100000)
            self.paper_trader = PaperTradingEngine(initial_capital, config)
        else:
            self.paper_trader = None

        self.risk_checks_enabled = config.get('risk', {}).get('enabled', True)

    def process_signals(self, signals: List[PairSignal], market_data: Dict = None) -> List[str]:
        if not signals:
            return []

        if self.mode == 'paper' and market_data:
            self.paper_trader.update_market_data(market_data)

        if self.risk_checks_enabled:
            risk_status = self._check_risk_limits()
            if risk_status != 'OK':
                logger.warning(f"Risk check failed: {risk_status}")
                return []

        execution_results = []

        for signal in signals:
            try:
                result = self._execute_pair_signal(signal)
                execution_results.extend(result)
            except Exception as e:
                logger.error(f"Error executing signal for {signal.pair_name}: {e}")

        return execution_results

    def _execute_pair_signal(self, signal: PairSignal) -> List[str]:
        results = []

        base_result = self._execute_single_signal(signal.base_signal)
        if base_result:
            results.append(base_result)

        hedge_result = self._execute_single_signal(signal.hedge_signal)
        if hedge_result:
            results.append(hedge_result)

        logger.info(
            f"Executed pair signal: {signal.pair_name} | "
            f"Base: {signal.base_signal.side} {signal.base_signal.symbol} | "
            f"Hedge: {signal.hedge_signal.side} {signal.hedge_signal.symbol}"
        )

        return results

    def _execute_single_signal(self, signal) -> Optional[str]:
        if signal.side == 'close':
            return self._close_position(signal.symbol)
        else:
            size = signal.size
            if size <= 0:
                return None

            return self._place_order(
                symbol=signal.symbol,
                side='buy' if signal.side == 'long' else 'sell',
                size=size,
                price=signal.price
            )

    def _place_order(
        self,
        symbol: str,
        side: str,
        size: float,
        price: Optional[float] = None
    ) -> Optional[str]:
        try:
            if self.mode == 'live' and self.binance_client:
                order_type = 'MARKET' if price is None else 'LIMIT'
                order = self.binance_client.place_order(
                    symbol=symbol,
                    side=side.upper(),
                    order_type=order_type,
                    quantity=size,
                    price=price
                )
                return order.get('orderId', 'unknown')

            elif self.mode == 'paper':
                order_type = 'market' if price is None else 'limit'
                order = self.paper_trader.place_order(
                    symbol=symbol,
                    side=side,
                    size=size,
                    order_type=order_type,
                    price=price
                )
                return order.order_id

        except Exception as e:
            logger.error(f"Error placing order for {symbol}: {e}")
            return None

    def _close_position(self, symbol: str) -> Optional[str]:
        try:
            current_position = self.get_position(symbol)
            if not current_position or current_position['size'] == 0:
                return None

            close_side = 'sell' if current_position['side'] == 'long' else 'buy'
            close_size = abs(current_position['size'])

            return self._place_order(
                symbol=symbol,
                side=close_side,
                size=close_size
            )

        except Exception as e:
            logger.error(f"Error closing position for {symbol}: {e}")
            return None

    def get_position(self, symbol: str) -> Optional[Dict]:
        try:
            if self.mode == 'live' and self.binance_client:
                account = self.binance_client.get_account_balance()
                return account.get(symbol, None)

            elif self.mode == 'paper':
                positions = self.paper_trader.get_positions()
                return positions.get(symbol, None)

        except Exception as e:
            logger.error(f"Error getting position for {symbol}: {e}")
            return None

    def get_all_positions(self) -> Dict[str, Dict]:
        try:
            if self.mode == 'live' and self.binance_client:
                return self.binance_client.get_account_balance()
            elif self.mode == 'paper':
                return self.paper_trader.get_positions()
        except Exception as e:
            logger.error(f"Error getting all positions: {e}")
            return {}

    def get_account_summary(self) -> Dict:
        try:
            if self.mode == 'live' and self.binance_client:
                balance = self.binance_client.get_account_balance()
                total_value = sum(
                    asset_info.get('total', 0) * self._get_usd_price(asset)
                    for asset, asset_info in balance.items()
                )
                return {'total_value_usd': total_value, 'balances': balance}

            elif self.mode == 'paper':
                return self.paper_trader.get_account_balance()

        except Exception as e:
            logger.error(f"Error getting account summary: {e}")
            return {}

    def _get_usd_price(self, asset: str) -> float:
        if asset == 'USDT':
            return 1.0

        try:
            if self.mode == 'live' and self.binance_client:
                usdt_pair = f"{asset}USDT"
                price = self.binance_client.get_current_price(usdt_pair)
                return float(price)
        except:
            pass

        return 1.0

    def _check_risk_limits(self) -> str:
        try:
            if self.mode == 'paper':
                account_summary = self.paper_trader.get_account_balance()
                portfolio_state_mock = type('obj', (object,), {
                    'total_equity': account_summary.get('total_equity', 0),
                    'unrealized_pnl': account_summary.get('unrealized_pnl', 0),
                    'gross_exposure': sum(
                        abs(pos.get('size', 0) * pos.get('current_price', 0))
                        for pos in self.paper_trader.get_positions().values()
                    )
                })()

                portfolio_state_mock.daily_dd = 0  # TODO: Calculate actual daily drawdown

                return enforce_risk_limits(portfolio_state_mock, self.config)

            return 'OK'

        except Exception as e:
            logger.error(f"Error in risk checks: {e}")
            return 'ERROR'

    def cancel_order(self, order_id: str) -> bool:
        try:
            if self.mode == 'live' and self.binance_client:
                logger.warning("Live order cancellation not implemented")
                return False
            elif self.mode == 'paper':
                return self.paper_trader.cancel_order(order_id)
        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
            return False


def compute_order_diff(current_positions: Dict, target_positions: Dict) -> List[Dict]:
    orders = []

    all_symbols = set(current_positions.keys()) | set(target_positions.keys())

    for symbol in all_symbols:
        current_size = current_positions.get(symbol, {}).get('size', 0)
        target_size = target_positions.get(symbol, {}).get('size', 0)

        size_diff = target_size - current_size

        if abs(size_diff) > 1e-8:  # Avoid tiny positions
            side = 'buy' if size_diff > 0 else 'sell'
            orders.append({
                'symbol': symbol,
                'side': side,
                'size': abs(size_diff),
                'type': 'market'
            })

    return orders


def simulate_trade(order: Dict) -> Dict:
    logger.info(f"SIMULATED TRADE: {order}")
    return {
        'order_id': f"sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        'status': 'filled',
        'filled_size': order['size'],
        'timestamp': datetime.now()
    }