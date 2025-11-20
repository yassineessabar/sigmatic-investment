import logging
import logging.handlers
import os
from typing import Dict
from pathlib import Path


def setup_logging(config: Dict):
    log_config = config.get('logging', {})
    log_level = log_config.get('level', 'INFO').upper()
    log_file = log_config.get('file', 'logs/trading.log')

    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level, logging.INFO))

    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(getattr(logging, log_level, logging.INFO))
    root_logger.addHandler(console_handler)

    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(getattr(logging, log_level, logging.INFO))
    root_logger.addHandler(file_handler)

    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('binance').setLevel(logging.WARNING)

    logging.info("Logging configured successfully")


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


class TradingLogger:
    def __init__(self, config: Dict):
        self.config = config
        self.setup_specialized_loggers()

    def setup_specialized_loggers(self):
        signals_logger = logging.getLogger('signals')
        signals_handler = logging.FileHandler('logs/signals.log')
        signals_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(message)s')
        )
        signals_logger.addHandler(signals_handler)
        signals_logger.setLevel(logging.INFO)

        trades_logger = logging.getLogger('trades')
        trades_handler = logging.FileHandler('logs/trades.log')
        trades_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(message)s')
        )
        trades_logger.addHandler(trades_handler)
        trades_logger.setLevel(logging.INFO)

        risk_logger = logging.getLogger('risk')
        risk_handler = logging.FileHandler('logs/risk.log')
        risk_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )
        risk_logger.addHandler(risk_handler)
        risk_logger.setLevel(logging.WARNING)

    def log_signal(self, signal_data: Dict):
        signals_logger = logging.getLogger('signals')
        signals_logger.info(f"Signal: {signal_data}")

    def log_trade(self, trade_data: Dict):
        trades_logger = logging.getLogger('trades')
        trades_logger.info(f"Trade: {trade_data}")

    def log_risk_event(self, risk_event: str, level: str = 'WARNING'):
        risk_logger = logging.getLogger('risk')
        if level == 'CRITICAL':
            risk_logger.critical(risk_event)
        elif level == 'ERROR':
            risk_logger.error(risk_event)
        else:
            risk_logger.warning(risk_event)