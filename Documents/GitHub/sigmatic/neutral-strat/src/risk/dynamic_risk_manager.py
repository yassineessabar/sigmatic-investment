#!/usr/bin/env python3

"""
Dynamic Risk Management System for Statistical Arbitrage
Advanced real-time risk monitoring and position management
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RiskMetrics:
    """Container for risk metrics"""
    portfolio_var: float
    expected_shortfall: float
    max_drawdown: float
    concentration_risk: float
    leverage: float
    correlation_risk: float
    liquidity_risk: float
    overall_risk_level: RiskLevel


class DynamicRiskManager:
    """
    Advanced dynamic risk management system
    Features:
    - Real-time VaR calculation
    - Dynamic position sizing
    - Drawdown control
    - Concentration limits
    - Regime-aware risk adjustment
    """

    def __init__(self,
                 initial_capital: float = 10000,
                 max_leverage: float = 10.0,
                 var_confidence: float = 0.05,
                 max_position_concentration: float = 0.3,
                 max_sector_concentration: float = 0.5,
                 max_drawdown_limit: float = 0.15,
                 daily_var_limit: float = 0.02):

        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_leverage = max_leverage
        self.var_confidence = var_confidence
        self.max_position_concentration = max_position_concentration
        self.max_sector_concentration = max_sector_concentration
        self.max_drawdown_limit = max_drawdown_limit
        self.daily_var_limit = daily_var_limit

        # Risk monitoring
        self.portfolio_history = []
        self.risk_history = []
        self.max_portfolio_value = initial_capital
        self.current_positions = {}

        # Risk adjustment factors
        self.risk_adjustment_factors = {
            RiskLevel.LOW: 1.0,
            RiskLevel.MEDIUM: 0.8,
            RiskLevel.HIGH: 0.5,
            RiskLevel.CRITICAL: 0.2
        }

    def calculate_portfolio_var(self, returns_df: pd.DataFrame, positions: pd.Series,
                               confidence_level: float = None) -> float:
        """
        Calculate portfolio Value at Risk
        """
        if confidence_level is None:
            confidence_level = self.var_confidence

        if len(returns_df) < 30:
            return 0.0

        # Calculate portfolio returns
        portfolio_returns = (positions * returns_df).sum(axis=1)

        # Remove NaN values
        portfolio_returns = portfolio_returns.dropna()

        if len(portfolio_returns) < 10:
            return 0.0

        # Calculate VaR using historical simulation
        var = np.percentile(portfolio_returns, confidence_level * 100)

        return abs(var)

    def calculate_expected_shortfall(self, returns_df: pd.DataFrame, positions: pd.Series,
                                   confidence_level: float = None) -> float:
        """
        Calculate Expected Shortfall (Conditional VaR)
        """
        if confidence_level is None:
            confidence_level = self.var_confidence

        if len(returns_df) < 30:
            return 0.0

        # Calculate portfolio returns
        portfolio_returns = (positions * returns_df).sum(axis=1)
        portfolio_returns = portfolio_returns.dropna()

        if len(portfolio_returns) < 10:
            return 0.0

        # Calculate VaR threshold
        var_threshold = np.percentile(portfolio_returns, confidence_level * 100)

        # Calculate expected shortfall
        tail_losses = portfolio_returns[portfolio_returns <= var_threshold]

        if len(tail_losses) > 0:
            expected_shortfall = abs(tail_losses.mean())
        else:
            expected_shortfall = abs(var_threshold)

        return expected_shortfall

    def calculate_concentration_risk(self, positions: pd.Series) -> float:
        """
        Calculate concentration risk score
        """
        if len(positions) == 0:
            return 0.0

        # Normalize positions
        total_exposure = positions.abs().sum()
        if total_exposure == 0:
            return 0.0

        position_weights = positions.abs() / total_exposure

        # Calculate Herfindahl-Hirschman Index
        hhi = (position_weights ** 2).sum()

        # Normalize to 0-1 scale (1/n is minimum concentration)
        min_hhi = 1.0 / len(positions)
        concentration_score = (hhi - min_hhi) / (1.0 - min_hhi) if len(positions) > 1 else 0.0

        return max(0.0, min(1.0, concentration_score))

    def calculate_correlation_risk(self, returns_df: pd.DataFrame) -> float:
        """
        Calculate correlation risk score based on average correlation
        """
        if len(returns_df.columns) < 2 or len(returns_df) < 30:
            return 0.0

        # Calculate correlation matrix
        corr_matrix = returns_df.corr()

        # Extract upper triangular matrix (excluding diagonal)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        correlations = corr_matrix.where(mask)

        # Calculate average absolute correlation
        avg_correlation = correlations.abs().stack().mean()

        return avg_correlation if not np.isnan(avg_correlation) else 0.0

    def calculate_liquidity_risk(self, volumes_df: pd.DataFrame, positions: pd.Series) -> float:
        """
        Calculate liquidity risk based on volume and position sizes
        """
        if volumes_df is None or len(volumes_df) == 0:
            return 0.5  # Default medium liquidity risk

        # Calculate average daily volume
        avg_volumes = volumes_df.rolling(20).mean().iloc[-1]

        # Calculate position turnover as fraction of daily volume
        liquidity_ratios = []
        for asset in positions.index:
            if asset in avg_volumes.index:
                position_value = abs(positions[asset]) * self.current_capital
                avg_volume_value = avg_volumes[asset] * 1  # Assuming price = 1 for simplification

                if avg_volume_value > 0:
                    liquidity_ratio = position_value / avg_volume_value
                    liquidity_ratios.append(min(liquidity_ratio, 1.0))

        if liquidity_ratios:
            return np.mean(liquidity_ratios)
        else:
            return 0.5

    def assess_overall_risk(self, risk_metrics: RiskMetrics) -> RiskLevel:
        """
        Assess overall risk level based on multiple metrics
        """
        risk_scores = []

        # VaR risk score
        var_score = min(risk_metrics.portfolio_var / self.daily_var_limit, 1.0)
        risk_scores.append(var_score)

        # Drawdown risk score
        dd_score = min(risk_metrics.max_drawdown / self.max_drawdown_limit, 1.0)
        risk_scores.append(dd_score)

        # Concentration risk score
        risk_scores.append(risk_metrics.concentration_risk)

        # Leverage risk score
        leverage_score = min(risk_metrics.leverage / self.max_leverage, 1.0)
        risk_scores.append(leverage_score)

        # Correlation risk score
        risk_scores.append(risk_metrics.correlation_risk)

        # Liquidity risk score
        risk_scores.append(risk_metrics.liquidity_risk)

        # Calculate weighted average
        overall_score = np.mean(risk_scores)

        # Determine risk level
        if overall_score >= 0.8:
            return RiskLevel.CRITICAL
        elif overall_score >= 0.6:
            return RiskLevel.HIGH
        elif overall_score >= 0.3:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW

    def calculate_risk_metrics(self, returns_df: pd.DataFrame, positions: pd.Series,
                             volumes_df: pd.DataFrame = None) -> RiskMetrics:
        """
        Calculate comprehensive risk metrics
        """
        # Portfolio VaR
        portfolio_var = self.calculate_portfolio_var(returns_df, positions)

        # Expected Shortfall
        expected_shortfall = self.calculate_expected_shortfall(returns_df, positions)

        # Current drawdown
        current_drawdown = (self.current_capital - self.max_portfolio_value) / self.max_portfolio_value

        # Concentration risk
        concentration_risk = self.calculate_concentration_risk(positions)

        # Current leverage
        total_exposure = positions.abs().sum()
        leverage = total_exposure

        # Correlation risk
        correlation_risk = self.calculate_correlation_risk(returns_df)

        # Liquidity risk
        liquidity_risk = self.calculate_liquidity_risk(volumes_df, positions)

        risk_metrics = RiskMetrics(
            portfolio_var=portfolio_var,
            expected_shortfall=expected_shortfall,
            max_drawdown=current_drawdown,
            concentration_risk=concentration_risk,
            leverage=leverage,
            correlation_risk=correlation_risk,
            liquidity_risk=liquidity_risk,
            overall_risk_level=RiskLevel.LOW  # Will be set below
        )

        # Assess overall risk
        risk_metrics.overall_risk_level = self.assess_overall_risk(risk_metrics)

        return risk_metrics

    def apply_risk_adjustments(self, target_positions: pd.Series, risk_metrics: RiskMetrics) -> pd.Series:
        """
        Apply risk adjustments to target positions
        """
        adjusted_positions = target_positions.copy()

        # Apply overall risk adjustment
        risk_factor = self.risk_adjustment_factors[risk_metrics.overall_risk_level]
        adjusted_positions *= risk_factor

        # Concentration limits
        total_exposure = adjusted_positions.abs().sum()
        if total_exposure > 0:
            position_weights = adjusted_positions.abs() / total_exposure

            # Apply individual position limits
            for asset in position_weights.index:
                if position_weights[asset] > self.max_position_concentration:
                    scale_factor = self.max_position_concentration / position_weights[asset]
                    adjusted_positions[asset] *= scale_factor

        # Leverage limits
        total_adjusted_exposure = adjusted_positions.abs().sum()
        if total_adjusted_exposure > self.max_leverage:
            leverage_scale = self.max_leverage / total_adjusted_exposure
            adjusted_positions *= leverage_scale

        # VaR limits (if historical data available)
        if len(adjusted_positions) > 0:
            # Simulate risk with adjusted positions
            test_risk = self.calculate_portfolio_var(
                pd.DataFrame(np.random.randn(100, len(adjusted_positions)) * 0.02,
                           columns=adjusted_positions.index),
                adjusted_positions
            )

            if test_risk > self.daily_var_limit:
                var_scale = self.daily_var_limit / test_risk
                adjusted_positions *= var_scale

        return adjusted_positions

    def update_portfolio_state(self, new_portfolio_value: float, positions: pd.Series):
        """
        Update portfolio state for risk monitoring
        """
        self.current_capital = new_portfolio_value
        self.max_portfolio_value = max(self.max_portfolio_value, new_portfolio_value)
        self.current_positions = positions.to_dict()

        # Update history
        self.portfolio_history.append({
            'timestamp': pd.Timestamp.now(),
            'portfolio_value': new_portfolio_value,
            'positions': positions.to_dict()
        })

        # Keep only recent history (last 1000 entries)
        if len(self.portfolio_history) > 1000:
            self.portfolio_history = self.portfolio_history[-1000:]

    def get_risk_report(self, returns_df: pd.DataFrame, positions: pd.Series,
                       volumes_df: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Generate comprehensive risk report
        """
        risk_metrics = self.calculate_risk_metrics(returns_df, positions, volumes_df)

        # Calculate additional metrics
        total_exposure = positions.abs().sum()
        net_exposure = positions.sum()

        # Position breakdown
        long_positions = positions[positions > 0]
        short_positions = positions[positions < 0]

        risk_report = {
            'timestamp': pd.Timestamp.now(),
            'overall_risk_level': risk_metrics.overall_risk_level.value,
            'portfolio_metrics': {
                'current_capital': self.current_capital,
                'max_portfolio_value': self.max_portfolio_value,
                'current_drawdown': risk_metrics.max_drawdown,
                'total_exposure': total_exposure,
                'net_exposure': net_exposure,
                'leverage': risk_metrics.leverage
            },
            'risk_metrics': {
                'portfolio_var': risk_metrics.portfolio_var,
                'expected_shortfall': risk_metrics.expected_shortfall,
                'concentration_risk': risk_metrics.concentration_risk,
                'correlation_risk': risk_metrics.correlation_risk,
                'liquidity_risk': risk_metrics.liquidity_risk
            },
            'position_breakdown': {
                'long_positions': len(long_positions),
                'short_positions': len(short_positions),
                'largest_long': long_positions.max() if len(long_positions) > 0 else 0,
                'largest_short': abs(short_positions.min()) if len(short_positions) > 0 else 0
            },
            'risk_limits': {
                'max_leverage': self.max_leverage,
                'daily_var_limit': self.daily_var_limit,
                'max_drawdown_limit': self.max_drawdown_limit,
                'max_position_concentration': self.max_position_concentration
            },
            'limit_utilization': {
                'leverage_utilization': min(risk_metrics.leverage / self.max_leverage, 1.0),
                'var_utilization': min(risk_metrics.portfolio_var / self.daily_var_limit, 1.0),
                'drawdown_utilization': min(abs(risk_metrics.max_drawdown) / self.max_drawdown_limit, 1.0)
            }
        }

        return risk_report

    def should_reduce_risk(self, risk_metrics: RiskMetrics) -> bool:
        """
        Determine if risk should be reduced immediately
        """
        return (
            risk_metrics.overall_risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL] or
            risk_metrics.portfolio_var > self.daily_var_limit or
            abs(risk_metrics.max_drawdown) > self.max_drawdown_limit or
            risk_metrics.leverage > self.max_leverage
        )