import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import schedule
from datetime import datetime
import logging
from src.utils.config import ConfigManager
from src.utils.logger import setup_logging
from .data_loader import DataLoader
from .binance_client import BinanceDataClient

logger = logging.getLogger(__name__)


class DataCollector:
    def __init__(self, config):
        self.config = config
        self.binance_client = BinanceDataClient(
            api_key=os.getenv('BINANCE_API_KEY'),
            api_secret=os.getenv('BINANCE_API_SECRET'),
            testnet=config['binance']['testnet']
        )
        self.data_loader = DataLoader(self.binance_client)

    def collect_current_data(self):
        logger.info("Collecting current market data...")

        symbols = []
        for pair in self.config['pairs']:
            symbols.extend([pair['base'], pair['hedge']])
        symbols = list(set(symbols))

        try:
            current_data = self.data_loader.get_live_data_slice(
                symbols=symbols,
                interval=self.config['execution']['interval'],
                lookback_periods=10
            )

            for symbol, data in current_data.items():
                if not data.empty:
                    latest_price = data['close'].iloc[-1]
                    latest_time = data.index[-1]
                    logger.info(f"{symbol}: ${latest_price:.4f} at {latest_time}")

        except Exception as e:
            logger.error(f"Error collecting data: {e}")

    def start_collection(self):
        interval = self.config['execution']['interval']

        if interval == '1m':
            schedule.every().minute.do(self.collect_current_data)
        elif interval == '5m':
            schedule.every(5).minutes.do(self.collect_current_data)
        elif interval == '15m':
            schedule.every(15).minutes.do(self.collect_current_data)
        elif interval == '30m':
            schedule.every(30).minutes.do(self.collect_current_data)
        elif interval == '1h':
            schedule.every().hour.do(self.collect_current_data)
        else:
            schedule.every().hour.do(self.collect_current_data)

        logger.info(f"Data collection scheduled every {interval}")

        while True:
            try:
                schedule.run_pending()
                time.sleep(30)
            except KeyboardInterrupt:
                logger.info("Data collection stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in data collection loop: {e}")
                time.sleep(60)


def main():
    config_manager = ConfigManager()
    config = config_manager.load_config('config/neutral_pairs.yaml')

    setup_logging(config)

    collector = DataCollector(config)
    collector.start_collection()


if __name__ == "__main__":
    main()