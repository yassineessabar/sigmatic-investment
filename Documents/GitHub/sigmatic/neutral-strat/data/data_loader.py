from typing import Dict, List, Tuple, Optional, Generator
import pandas as pd
from datetime import datetime, timedelta
import logging
from .binance_client import BinanceDataClient

logger = logging.getLogger(__name__)


class DataLoader:
    def __init__(self, binance_client: BinanceDataClient):
        self.client = binance_client
        self._cache = {}

    def load_pair_data(
        self,
        symbols: List[str],
        interval: str,
        lookback_hours: int = 24 * 7,
        end_time: Optional[datetime] = None
    ) -> Dict[str, pd.DataFrame]:
        if end_time is None:
            end_time = datetime.utcnow()

        start_time = end_time - timedelta(hours=lookback_hours)

        data = {}
        for symbol in symbols:
            cache_key = f"{symbol}_{interval}_{start_time}_{end_time}"

            if cache_key in self._cache:
                logger.info(f"Using cached data for {symbol}")
                data[symbol] = self._cache[cache_key]
            else:
                logger.info(f"Fetching data for {symbol}")
                df = self.client.get_historical_klines(
                    symbol=symbol,
                    interval=interval,
                    start_time=start_time.isoformat(),
                    end_time=end_time.isoformat()
                )
                data[symbol] = df
                self._cache[cache_key] = df

        return data

    def create_backtest_iterator(
        self,
        symbols: List[str],
        interval: str,
        start_date: datetime,
        end_date: datetime,
        lookback_periods: int = 100
    ) -> Generator[Tuple[datetime, Dict[str, pd.DataFrame]], None, None]:
        full_data = self.load_pair_data(
            symbols=symbols,
            interval=interval,
            lookback_hours=int((end_date - start_date).total_seconds() / 3600) + lookback_periods,
            end_time=end_date
        )

        if not full_data:
            raise ValueError("No data loaded")

        timestamps = sorted(set().union(*[df.index for df in full_data.values()]))
        timestamps = [ts for ts in timestamps if start_date <= ts <= end_date]

        for i, current_time in enumerate(timestamps):
            if i < lookback_periods:
                continue

            lookback_start_idx = i - lookback_periods
            current_data = {}

            for symbol, df in full_data.items():
                symbol_timestamps = df.index
                valid_timestamps = [
                    ts for ts in symbol_timestamps[lookback_start_idx:i+1]
                    if ts in symbol_timestamps
                ]

                if valid_timestamps:
                    current_data[symbol] = df.loc[valid_timestamps]

            if len(current_data) == len(symbols):
                yield current_time, current_data

    def get_live_data_slice(
        self,
        symbols: List[str],
        interval: str,
        lookback_periods: int = 100
    ) -> Dict[str, pd.DataFrame]:
        return self.load_pair_data(
            symbols=symbols,
            interval=interval,
            lookback_hours=lookback_periods * self._interval_to_hours(interval)
        )

    @staticmethod
    def _interval_to_hours(interval: str) -> float:
        interval_map = {
            '1m': 1/60,
            '5m': 5/60,
            '15m': 15/60,
            '30m': 0.5,
            '1h': 1,
            '2h': 2,
            '4h': 4,
            '6h': 6,
            '8h': 8,
            '12h': 12,
            '1d': 24,
        }
        return interval_map.get(interval, 1)

    def align_data(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        if len(data) < 2:
            return data

        common_timestamps = set(data[list(data.keys())[0]].index)
        for df in data.values():
            common_timestamps &= set(df.index)

        if not common_timestamps:
            logger.warning("No common timestamps found in data")
            return data

        aligned_data = {}
        for symbol, df in data.items():
            aligned_data[symbol] = df.loc[sorted(common_timestamps)]

        return aligned_data