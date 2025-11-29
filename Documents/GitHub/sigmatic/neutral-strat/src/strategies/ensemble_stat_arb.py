#!/usr/bin/env python3

"""
Ensemble Statistical Arbitrage Strategy
Combines multiple strategies for enhanced 2+ Sharpe performance
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

from .crypto_stat_arb import CryptoStatArbStrategy
from ..risk.dynamic_risk_manager import DynamicRiskManager, RiskLevel

logger = logging.getLogger(__name__)


@dataclass
class StrategyComponent:
    """Individual strategy component in the ensemble"""
    name: str
    strategy: Any
    weight: float
    performance_history: List[float]
    sharpe_history: List[float]
    enabled: bool = True


class EnsembleStatArbStrategy:
    """
    Ensemble Statistical Arbitrage Strategy

    Features:
    - Multiple strategy combination
    - Dynamic weight adjustment
    - Performance-based allocation
    - Risk-aware ensemble management
    - Regime-specific strategy selection
    """

    def __init__(self,
                 commission_bps: float = 1.0,
                 min_strategy_weight: float = 0.1,
                 max_strategy_weight: float = 0.6,
                 performance_window: int = 60,
                 rebalance_frequency: int = 5):

        self.commission_bps = commission_bps
        self.min_strategy_weight = min_strategy_weight
        self.max_strategy_weight = max_strategy_weight
        self.performance_window = performance_window
        self.rebalance_frequency = rebalance_frequency
        self.periods_per_year = 365

        # Initialize strategy components
        self.strategies = {}
        self.risk_manager = DynamicRiskManager()

        # Performance tracking
        self.ensemble_history = []
        self.weight_history = []

        self._initialize_strategies()

    def _initialize_strategies(self):
        """Initialize individual strategy components"""

        # Basic mean reversion strategy
        self.strategies['mean_reversion'] = StrategyComponent(
            name='mean_reversion',
            strategy=CryptoStatArbStrategy(commission_bps=self.commission_bps),
            weight=0.25,
            performance_history=[],
            sharpe_history=[]
        )

        # Volume momentum strategy
        self.strategies['volume_momentum'] = StrategyComponent(
            name='volume_momentum',
            strategy=CryptoStatArbStrategy(commission_bps=self.commission_bps),
            weight=0.25,
            performance_history=[],
            sharpe_history=[]
        )

        # Enhanced momentum strategy (alternative to ML)
        self.strategies['enhanced_momentum'] = StrategyComponent(
            name='enhanced_momentum',
            strategy=CryptoStatArbStrategy(commission_bps=self.commission_bps),
            weight=0.25,
            performance_history=[],
            sharpe_history=[]
        )

        # Adaptive strategy (parameter adjustment)
        self.strategies['adaptive'] = StrategyComponent(
            name='adaptive',
            strategy=CryptoStatArbStrategy(commission_bps=self.commission_bps),
            weight=0.25,
            performance_history=[],
            sharpe_history=[]
        )

    def calculate_strategy_signals(self, data: Dict[str, pd.DataFrame],
                                 current_date: pd.Timestamp) -> Dict[str, pd.Series]:
        """Calculate signals from all strategy components"""

        signals = {}
        prices_df = data['prices']
        volumes_df = data.get('volumes', None)

        # Get data up to current date
        historical_data = {
            'prices': prices_df.loc[:current_date],
            'volumes': volumes_df.loc[:current_date] if volumes_df is not None else None
        }

        if len(historical_data['prices']) < 60:  # Need sufficient history
            return signals

        try:
            # Mean reversion signals
            mr_strategy = self.strategies['mean_reversion'].strategy
            mr_results = mr_strategy.generate_crypto_pairs_signals(
                list(prices_df.columns), historical_data
            )

            if 'mean_reversion' in mr_results:
                signals['mean_reversion'] = mr_results['mean_reversion']

            # Volume momentum signals
            vm_strategy = self.strategies['volume_momentum'].strategy
            vm_results = vm_strategy.generate_crypto_pairs_signals(
                list(prices_df.columns), historical_data
            )

            if 'volume_momentum' in vm_results:
                signals['volume_momentum'] = vm_results['volume_momentum']

            # Enhanced momentum signals (using shorter periods)
            em_strategy = self.strategies['enhanced_momentum'].strategy
            em_returns = em_strategy.run_volume_momentum_backtest(
                historical_data['prices'],
                historical_data['volumes'] if historical_data['volumes'] is not None else historical_data['prices'].copy(),
                formation_period=5,
                holding_period=1,
                long_threshold=0.25,
                short_threshold=0.75,
                signal_lag=1
            )
            signals['enhanced_momentum'] = em_returns

            # Adaptive signals (with different parameters)
            adaptive_strategy = self.strategies['adaptive'].strategy
            # Use shorter formation period for adaptive
            adaptive_returns = adaptive_strategy.run_market_neutral_reversal_backtest(
                historical_data['prices'],
                formation_period=2,
                holding_period=1,
                long_threshold=0.15,
                short_threshold=0.85
            )
            signals['adaptive'] = adaptive_returns

        except Exception as e:
            logger.warning(f"Error calculating strategy signals: {e}")

        return signals

    def update_strategy_weights(self, strategy_returns: Dict[str, pd.Series]):
        """Update strategy weights based on recent performance"""

        if not strategy_returns:
            return

        # Calculate recent performance metrics
        strategy_metrics = {}

        for name, returns in strategy_returns.items():
            if name in self.strategies and len(returns) >= self.performance_window:
                recent_returns = returns.tail(self.performance_window)

                # Calculate metrics
                sharpe = self.calculate_sharpe_ratio(recent_returns)
                total_return = (1 + recent_returns).prod() - 1
                volatility = recent_returns.std() * np.sqrt(self.periods_per_year)
                max_dd = self.calculate_max_drawdown(recent_returns)

                # Composite score (sharpe-focused with risk adjustment)
                risk_penalty = max(0, (abs(max_dd) - 0.1) * 10)  # Penalize >10% drawdown
                score = sharpe - risk_penalty

                strategy_metrics[name] = {
                    'sharpe': sharpe,
                    'return': total_return,
                    'volatility': volatility,
                    'max_dd': max_dd,
                    'score': score
                }

                # Update history
                self.strategies[name].performance_history.append(total_return)
                self.strategies[name].sharpe_history.append(sharpe)

                # Keep only recent history
                if len(self.strategies[name].performance_history) > 100:
                    self.strategies[name].performance_history = \
                        self.strategies[name].performance_history[-100:]
                    self.strategies[name].sharpe_history = \
                        self.strategies[name].sharpe_history[-100:]

        if not strategy_metrics:
            return

        # Calculate new weights based on performance
        scores = np.array([metrics['score'] for metrics in strategy_metrics.values()])

        # Handle negative scores
        if scores.min() < 0:
            scores = scores - scores.min() + 0.1

        # Softmax allocation with temperature parameter
        temperature = 2.0  # Controls allocation sharpness
        exp_scores = np.exp(scores / temperature)
        raw_weights = exp_scores / exp_scores.sum()

        # Apply weight constraints
        constrained_weights = np.clip(raw_weights,
                                    self.min_strategy_weight,
                                    self.max_strategy_weight)

        # Normalize to sum to 1
        constrained_weights = constrained_weights / constrained_weights.sum()

        # Update strategy weights
        for i, name in enumerate(strategy_metrics.keys()):
            old_weight = self.strategies[name].weight
            new_weight = constrained_weights[i]

            # Smooth weight transitions
            alpha = 0.3  # Learning rate
            self.strategies[name].weight = (1 - alpha) * old_weight + alpha * new_weight

        # Record weight changes
        current_weights = {name: comp.weight for name, comp in self.strategies.items()}
        self.weight_history.append(current_weights)

        if len(self.weight_history) > 200:
            self.weight_history = self.weight_history[-200:]

    def generate_ensemble_signals(self, strategy_signals: Dict[str, pd.Series],
                                current_date: pd.Timestamp) -> pd.Series:
        """Combine individual strategy signals using current weights"""

        if not strategy_signals:
            return pd.Series(dtype=float)

        # Get the latest signals from each strategy
        ensemble_signal = None
        total_weight = 0

        for name, signals in strategy_signals.items():
            if name in self.strategies and self.strategies[name].enabled:

                weight = self.strategies[name].weight

                if len(signals) > 0:
                    # Get latest signal
                    latest_signal = signals.iloc[-1] if not np.isnan(signals.iloc[-1]) else 0

                    if ensemble_signal is None:
                        ensemble_signal = latest_signal * weight
                    else:
                        ensemble_signal += latest_signal * weight

                    total_weight += weight

        # Normalize by total weight
        if total_weight > 0 and ensemble_signal is not None:
            ensemble_signal = ensemble_signal / total_weight
        else:
            ensemble_signal = 0

        return pd.Series([ensemble_signal], index=[current_date])

    def run_ensemble_backtest(self, data: Dict[str, pd.DataFrame],
                            lookback_period: int = 90) -> pd.Series:
        """Run ensemble strategy backtest"""

        prices_df = data['prices']

        if len(prices_df) < lookback_period + 30:
            logger.warning("Insufficient data for ensemble backtest")
            return pd.Series(dtype=float)

        ensemble_returns = []
        strategy_return_history = {name: [] for name in self.strategies.keys()}

        # Walk forward through the data
        for i in range(lookback_period, len(prices_df), self.rebalance_frequency):

            current_date = prices_df.index[i]

            # Get strategy signals
            strategy_signals = self.calculate_strategy_signals(data, current_date)

            # Update individual strategy returns
            for name, signals in strategy_signals.items():
                if len(signals) >= self.rebalance_frequency:
                    recent_returns = signals.tail(self.rebalance_frequency)
                    strategy_return_history[name].extend(recent_returns.tolist())

            # Update weights based on recent performance
            if i % (self.rebalance_frequency * 4) == 0:  # Update weights less frequently
                strategy_series = {}
                for name, returns in strategy_return_history.items():
                    if len(returns) >= self.performance_window:
                        strategy_series[name] = pd.Series(returns[-self.performance_window:])

                self.update_strategy_weights(strategy_series)

            # Generate ensemble signal
            ensemble_signal = self.generate_ensemble_signals(strategy_signals, current_date)

            if len(ensemble_signal) > 0:
                ensemble_returns.extend(ensemble_signal.tolist())

        # Convert to pandas Series
        if ensemble_returns:
            return pd.Series(ensemble_returns,
                           index=prices_df.index[lookback_period:lookback_period+len(ensemble_returns)])
        else:
            return pd.Series(dtype=float)

    def optimize_ensemble(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Optimize ensemble parameters"""

        best_params = {}
        best_sharpe = -np.inf
        best_returns = None

        # Parameter grid for optimization
        lookback_periods = [60, 90, 120]
        rebalance_frequencies = [3, 5, 7]
        performance_windows = [30, 45, 60]

        # Split data
        split_idx = int(len(data['prices']) * 0.7)
        train_data = {
            'prices': data['prices'].iloc[:split_idx],
            'volumes': data['volumes'].iloc[:split_idx] if 'volumes' in data else None
        }

        test_data = {
            'prices': data['prices'].iloc[split_idx:],
            'volumes': data['volumes'].iloc[split_idx:] if 'volumes' in data else None
        }

        for lookback in lookback_periods:
            for rebal_freq in rebalance_frequencies:
                for perf_window in performance_windows:

                    try:
                        # Set parameters
                        self.rebalance_frequency = rebal_freq
                        self.performance_window = perf_window

                        # Train on training data
                        train_returns = self.run_ensemble_backtest(train_data, lookback)

                        if len(train_returns) > 50:
                            train_sharpe = self.calculate_sharpe_ratio(train_returns)

                            if train_sharpe > 0.5:  # Filter promising candidates
                                # Test on validation data
                                test_returns = self.run_ensemble_backtest(test_data, lookback)
                                test_sharpe = self.calculate_sharpe_ratio(test_returns)

                                if test_sharpe > best_sharpe:
                                    best_sharpe = test_sharpe
                                    best_returns = test_returns
                                    best_params = {
                                        'lookback_period': lookback,
                                        'rebalance_frequency': rebal_freq,
                                        'performance_window': perf_window,
                                        'strategy_type': 'ensemble_stat_arb'
                                    }

                    except Exception as e:
                        logger.warning(f"Error in ensemble optimization: {e}")
                        continue

        # Calculate final metrics
        if best_returns is not None:
            metrics = self.calculate_performance_metrics(best_returns)
            best_params.update(metrics)

        return best_params

    def calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate annualized Sharpe ratio"""
        if len(returns) == 0 or returns.std() == 0:
            return 0

        mean_return = returns.mean() * self.periods_per_year
        volatility = returns.std() * np.sqrt(self.periods_per_year)

        return mean_return / volatility if volatility > 0 else 0

    def calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        if len(returns) == 0:
            return 0

        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max

        return drawdown.min()

    def calculate_performance_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate comprehensive performance metrics"""
        if len(returns) == 0:
            return {}

        returns_clean = returns.dropna()

        # Basic metrics
        total_return = (1 + returns_clean).prod() - 1
        annual_return = returns_clean.mean() * self.periods_per_year
        annual_vol = returns_clean.std() * np.sqrt(self.periods_per_year)
        sharpe = annual_return / annual_vol if annual_vol > 0 else 0

        # Drawdown metrics
        max_drawdown = self.calculate_max_drawdown(returns_clean)
        calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # Additional metrics
        negative_returns = returns_clean[returns_clean < 0]
        downside_std = negative_returns.std() * np.sqrt(self.periods_per_year) if len(negative_returns) > 0 else 0
        sortino = annual_return / downside_std if downside_std > 0 else 0

        win_rate = (returns_clean > 0).mean()

        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar,
            'sortino_ratio': sortino,
            'win_rate': win_rate,
            'num_trades': len(returns_clean),
            'final_performance': 1 + total_return
        }

    def get_strategy_attribution(self) -> Dict[str, Any]:
        """Get performance attribution by strategy"""

        attribution = {}

        for name, strategy in self.strategies.items():
            if len(strategy.performance_history) > 0:
                recent_perf = strategy.performance_history[-10:] if len(strategy.performance_history) >= 10 else strategy.performance_history
                recent_sharpe = strategy.sharpe_history[-10:] if len(strategy.sharpe_history) >= 10 else strategy.sharpe_history

                attribution[name] = {
                    'current_weight': strategy.weight,
                    'avg_return': np.mean(recent_perf) if recent_perf else 0,
                    'avg_sharpe': np.mean(recent_sharpe) if recent_sharpe else 0,
                    'contribution': strategy.weight * (np.mean(recent_perf) if recent_perf else 0),
                    'enabled': strategy.enabled
                }

        return attribution


def run_ensemble_stat_arb_backtest(data: Dict[str, pd.DataFrame],
                                 commission_bps: float = 1.0) -> Dict[str, Any]:
    """
    Main function to run ensemble statistical arbitrage backtest
    """

    strategy = EnsembleStatArbStrategy(commission_bps=commission_bps)

    if len(data['prices'].columns) < 4:
        return {'error': 'Insufficient assets for ensemble strategy'}

    # Optimize ensemble parameters
    best_params = strategy.optimize_ensemble(data)

    if not best_params:
        return {'error': 'No valid ensemble parameters found'}

    # Run final backtest with best parameters
    final_returns = strategy.run_ensemble_backtest(
        data,
        lookback_period=best_params.get('lookback_period', 90)
    )

    # Calculate final metrics
    final_metrics = strategy.calculate_performance_metrics(final_returns)

    # Get strategy attribution
    attribution = strategy.get_strategy_attribution()

    return {
        'strategy_returns': {'ensemble_stat_arb': final_returns},
        'strategy_metrics': {'ensemble_stat_arb': final_metrics},
        'best_strategy': 'ensemble_stat_arb',
        'best_sharpe': final_metrics.get('sharpe_ratio', 0),
        'best_params': best_params,
        'strategy_attribution': attribution,
        'commission_bps': commission_bps
    }