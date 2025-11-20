from typing import Dict, List, Optional, Tuple
import pandas as pd
from binance.client import Client
from binance.exceptions import BinanceAPIException
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class BinanceDataClient:
    def __init__(self, api_key: str, api_secret: str, testnet: bool = True):
        self.client = Client(
            api_key=api_key,
            api_secret=api_secret,
            testnet=testnet
        )

    def get_historical_klines(
        self,
        symbol: str,
        interval: str,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        try:
            klines = self.client.get_historical_klines(
                symbol=symbol,
                interval=interval,
                start_str=start_time,
                end_str=end_time,
                limit=limit
            )

            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])

            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')

            numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'quote_volume']
            df[numeric_columns] = df[numeric_columns].astype(float)

            df.set_index('timestamp', inplace=True)
            df['symbol'] = symbol

            return df[['symbol', 'open', 'high', 'low', 'close', 'volume']]

        except BinanceAPIException as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            raise

    def get_current_price(self, symbol: str) -> float:
        try:
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            return float(ticker['price'])
        except BinanceAPIException as e:
            logger.error(f"Error fetching price for {symbol}: {e}")
            raise

    def get_account_balance(self) -> Dict[str, float]:
        try:
            account = self.client.get_account()
            balances = {}
            for balance in account['balances']:
                if float(balance['free']) > 0 or float(balance['locked']) > 0:
                    balances[balance['asset']] = {
                        'free': float(balance['free']),
                        'locked': float(balance['locked']),
                        'total': float(balance['free']) + float(balance['locked'])
                    }
            return balances
        except BinanceAPIException as e:
            logger.error(f"Error fetching account balance: {e}")
            raise

    def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float] = None,
        time_in_force: str = 'GTC'
    ) -> Dict:
        try:
            order_params = {
                'symbol': symbol,
                'side': side,
                'type': order_type,
                'quantity': quantity,
                'timeInForce': time_in_force
            }

            if price and order_type != 'MARKET':
                order_params['price'] = price

            order = self.client.create_order(**order_params)
            logger.info(f"Order placed: {order}")
            return order

        except BinanceAPIException as e:
            logger.error(f"Error placing order: {e}")
            raise

    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        try:
            orders = self.client.get_open_orders(symbol=symbol)
            return orders
        except BinanceAPIException as e:
            logger.error(f"Error fetching open orders: {e}")
            raise