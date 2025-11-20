from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
import numpy as np

logger = logging.getLogger(__name__)


class RiskMetrics:
    def __init__(self):
        self.daily_returns: List[float] = []
        self.equity_history: List[Tuple[datetime, float]] = []
        self.max_drawdown_start: Optional[datetime] = None
        self.max_drawdown_end: Optional[datetime] = None

    def update(self, timestamp: datetime, equity: float):
        self.equity_history.append((timestamp, equity))

        if len(self.equity_history) > 1:
            prev_equity = self.equity_history[-2][1]
            if prev_equity > 0:
                daily_return = (equity - prev_equity) / prev_equity
                self.daily_returns.append(daily_return)

    def calculate_sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        if len(self.daily_returns) < 2:
            return 0.0

        returns_array = np.array(self.daily_returns)
        excess_returns = returns_array - risk_free_rate / 252

        if np.std(excess_returns) == 0:
            return 0.0

        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)

    def calculate_max_drawdown(self) -> Tuple[float, Optional[datetime], Optional[datetime]]:
        if len(self.equity_history) < 2:
            return 0.0, None, None

        equities = [eq[1] for eq in self.equity_history]
        timestamps = [eq[0] for eq in self.equity_history]

        peak_equity = equities[0]
        peak_idx = 0
        max_dd = 0.0
        max_dd_start = None
        max_dd_end = None

        for i, equity in enumerate(equities):
            if equity > peak_equity:
                peak_equity = equity
                peak_idx = i
            else:
                drawdown = (peak_equity - equity) / peak_equity
                if drawdown > max_dd:
                    max_dd = drawdown
                    max_dd_start = timestamps[peak_idx]
                    max_dd_end = timestamps[i]

        return max_dd, max_dd_start, max_dd_end

    def calculate_var(self, confidence_level: float = 0.05) -> float:
        if len(self.daily_returns) < 30:
            return 0.0

        returns_array = np.array(self.daily_returns)
        return np.percentile(returns_array, confidence_level * 100)


def enforce_risk_limits(portfolio_state, config: Dict) -> str:
    risk_config = config.get('risk', {})

    max_daily_dd = risk_config.get('max_daily_dd', 0.05)
    max_total_dd = risk_config.get('max_total_dd', 0.10)
    leverage_limit = risk_config.get('leverage_limit', 2.0)
    max_position_size = risk_config.get('max_position_size', 0.1)

    try:
        if hasattr(portfolio_state, 'daily_dd') and portfolio_state.daily_dd > max_daily_dd:
            logger.critical(f"Daily drawdown {portfolio_state.daily_dd:.2%} exceeds limit {max_daily_dd:.2%}")
            return "STOP_TRADING"

        total_equity = getattr(portfolio_state, 'total_equity', 0)
        gross_exposure = getattr(portfolio_state, 'gross_exposure', 0)

        if total_equity > 0:
            leverage_ratio = gross_exposure / total_equity
            if leverage_ratio > leverage_limit:
                logger.warning(f"Leverage ratio {leverage_ratio:.2f} exceeds limit {leverage_limit:.2f}")
                return "REDUCE_POSITION"

        return "OK"

    except Exception as e:
        logger.error(f"Error in risk limit enforcement: {e}")
        return "ERROR"


def calculate_position_size(
    signal_confidence: float,
    portfolio_equity: float,
    volatility: float,
    config: Dict
) -> float:
    risk_config = config.get('risk', {})
    max_position_size = risk_config.get('max_position_size', 0.1)
    volatility_target = risk_config.get('volatility_target', 0.15)

    base_size = max_position_size * signal_confidence

    if volatility > 0:
        vol_adjustment = volatility_target / volatility
        vol_adjustment = np.clip(vol_adjustment, 0.1, 3.0)
        base_size *= vol_adjustment

    return min(base_size, max_position_size)


def check_correlation_limits(positions: Dict, config: Dict) -> bool:
    correlation_limit = config.get('risk', {}).get('max_correlation', 0.7)

    if len(positions) < 2:
        return True

    return True


def validate_order_risk(
    order_symbol: str,
    order_size: float,
    order_side: str,
    current_positions: Dict,
    config: Dict
) -> Tuple[bool, str]:
    risk_config = config.get('risk', {})
    max_position_size = risk_config.get('max_position_size', 0.1)
    max_single_order = risk_config.get('max_single_order', 0.05)

    if order_size > max_single_order:
        return False, f"Order size {order_size:.2%} exceeds single order limit {max_single_order:.2%}"

    current_position = current_positions.get(order_symbol, {})
    current_size = current_position.get('size', 0)
    current_side = current_position.get('side', 'none')

    if order_side == 'buy':
        if current_side == 'long':
            new_size = current_size + order_size
        elif current_side == 'short':
            new_size = max(0, order_size - current_size)
        else:
            new_size = order_size
    else:  # sell
        if current_side == 'short':
            new_size = current_size + order_size
        elif current_side == 'long':
            new_size = max(0, order_size - current_size)
        else:
            new_size = order_size

    if new_size > max_position_size:
        return False, f"Resulting position size {new_size:.2%} exceeds limit {max_position_size:.2%}"

    return True, "OK"


class RiskMonitor:
    def __init__(self, config: Dict):
        self.config = config
        self.risk_metrics = RiskMetrics()
        self.last_risk_check = datetime.now()
        self.trading_halted = False

    def update_portfolio_state(self, portfolio_state, timestamp: datetime = None):
        if timestamp is None:
            timestamp = datetime.now()

        equity = getattr(portfolio_state, 'total_equity', 0)
        self.risk_metrics.update(timestamp, equity)

        self.last_risk_check = timestamp

    def check_risk_limits(self, portfolio_state) -> str:
        risk_status = enforce_risk_limits(portfolio_state, self.config)

        if risk_status == "STOP_TRADING":
            self.trading_halted = True
            logger.critical("Trading halted due to risk limits")

        return risk_status

    def get_risk_report(self) -> Dict:
        max_dd, dd_start, dd_end = self.risk_metrics.calculate_max_drawdown()
        sharpe = self.risk_metrics.calculate_sharpe_ratio()
        var_95 = self.risk_metrics.calculate_var(0.05)

        return {
            'max_drawdown': max_dd,
            'drawdown_start': dd_start,
            'drawdown_end': dd_end,
            'sharpe_ratio': sharpe,
            'var_95': var_95,
            'daily_returns_count': len(self.risk_metrics.daily_returns),
            'trading_halted': self.trading_halted,
            'last_check': self.last_risk_check
        }

    def reset_halt(self):
        self.trading_halted = False
        logger.info("Trading halt reset")