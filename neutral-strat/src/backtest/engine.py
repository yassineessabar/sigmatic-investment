from typing import Dict, Callable, Optional, List
import pandas as pd
from datetime import datetime, timedelta
import logging

from .portfolio import PortfolioState, PortfolioMetrics
from ..strategies.neutral_pairs import compute_signals, validate_signals
from ...data.data_loader import DataLoader

logger = logging.getLogger(__name__)


class BacktestEngine:
    def __init__(self, config: Dict):
        self.config = config
        self.portfolio = PortfolioState(
            initial_capital=config.get('backtest', {}).get('initial_capital', 100000)
        )

    def run_backtest(
        self,
        data_loader: DataLoader,
        strategy_fn: Callable,
        start_date: datetime,
        end_date: datetime,
        symbols: List[str]
    ) -> PortfolioMetrics:
        logger.info(f"Starting backtest from {start_date} to {end_date}")

        lookback_periods = max([pair['lookback'] for pair in self.config['pairs']])

        try:
            data_iterator = data_loader.create_backtest_iterator(
                symbols=symbols,
                interval=self.config['execution']['interval'],
                start_date=start_date,
                end_date=end_date,
                lookback_periods=lookback_periods
            )

            for current_time, data_slice in data_iterator:
                self._process_timestep(current_time, data_slice, strategy_fn)

                if len(self.portfolio.equity_curve) % 100 == 0:
                    logger.info(f"Processed {len(self.portfolio.equity_curve)} timesteps")

            logger.info("Backtest completed successfully")
            return self.portfolio.get_results()

        except Exception as e:
            logger.error(f"Error during backtest: {e}")
            raise

    def _process_timestep(
        self,
        timestamp: datetime,
        data_slice: Dict[str, pd.DataFrame],
        strategy_fn: Callable
    ):
        self.portfolio.update_positions(data_slice, timestamp)

        try:
            signals = strategy_fn(data_slice, self.config)
            validated_signals = validate_signals(signals, self.config)

            trades = self.portfolio.compute_trades(validated_signals, self.config)
            self.portfolio.apply_trades(trades, data_slice, self.config)

        except Exception as e:
            logger.warning(f"Error processing signals at {timestamp}: {e}")

    def run_walk_forward_analysis(
        self,
        data_loader: DataLoader,
        strategy_fn: Callable,
        start_date: datetime,
        end_date: datetime,
        symbols: List[str],
        window_days: int = 30,
        step_days: int = 7
    ) -> List[PortfolioMetrics]:
        results = []
        current_start = start_date

        while current_start + timedelta(days=window_days) <= end_date:
            current_end = min(current_start + timedelta(days=window_days), end_date)

            logger.info(f"Running walk-forward window: {current_start} to {current_end}")

            window_engine = BacktestEngine(self.config)
            window_result = window_engine.run_backtest(
                data_loader=data_loader,
                strategy_fn=strategy_fn,
                start_date=current_start,
                end_date=current_end,
                symbols=symbols
            )

            results.append(window_result)
            current_start += timedelta(days=step_days)

        return results


def run_backtest(data_loader: DataLoader, strategy_fn: Callable, config: Dict) -> PortfolioMetrics:
    engine = BacktestEngine(config)

    start_date = datetime.strptime(
        config.get('backtest', {}).get('start_date', '2023-01-01'),
        '%Y-%m-%d'
    )
    end_date = datetime.strptime(
        config.get('backtest', {}).get('end_date', '2023-12-31'),
        '%Y-%m-%d'
    )

    symbols = []
    for pair in config['pairs']:
        symbols.extend([pair['base'], pair['hedge']])
    symbols = list(set(symbols))

    portfolio = PortfolioState(initial_capital=10000)

    try:
        data_iterator = data_loader.create_backtest_iterator(
            symbols=symbols,
            interval=config['execution']['interval'],
            start_date=start_date,
            end_date=end_date,
            lookback_periods=max([pair['lookback'] for pair in config['pairs']])
        )

        for t, data_slice in data_iterator:
            portfolio.update_positions(data_slice, t)

            signals = strategy_fn(data_slice, config)
            validated_signals = validate_signals(signals, config)

            trades = portfolio.compute_trades(validated_signals, config)
            portfolio.apply_trades(trades, data_slice, config)

        return portfolio.get_results()

    except Exception as e:
        logger.error(f"Error in backtest: {e}")
        raise