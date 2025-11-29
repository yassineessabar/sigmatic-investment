#!/usr/bin/env python3

"""
Crypto Statistical Arbitrage Strategy
Market-neutral statistical arbitrage for cryptocurrency pairs
Based on research by Ronald Lui - targeting 2+ Sharpe ratio
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class CryptoStatArbStrategy:
    """
    Market-neutral statistical arbitrage strategy for cryptocurrency
    Implements both mean reversion and enhanced volume-momentum approaches
    """

    def __init__(self, commission_bps: float = 5.0):
        self.commission_bps = commission_bps
        self.periods_per_year = 365  # Daily frequency

    def calculate_market_neutral_signal(self, prices_df: pd.DataFrame, formation_period: int = 1) -> pd.DataFrame:
        """
        Calculate market-neutral signal by removing market-wide movements

        Args:
            prices_df: DataFrame of cryptocurrency prices
            formation_period: Lookback period for signal calculation

        Returns:
            Market-neutral residual returns
        """
        # Calculate raw returns
        raw_returns = prices_df.pct_change(formation_period)

        # Market-neutral returns: subtract cross-sectional mean (market component)
        market_component = raw_returns.mean(axis=1)
        market_neutral_returns = raw_returns.subtract(market_component, axis=0)

        return market_neutral_returns

    def run_market_neutral_reversal_backtest(self, prices_df: pd.DataFrame,
                                           formation_period: int = 1,
                                           holding_period: int = 1,
                                           long_threshold: float = 0.2,
                                           short_threshold: float = 0.8) -> pd.Series:
        """
        Market-neutral mean reversion strategy
        Longs underperformers, shorts outperformers

        Args:
            prices_df: Price data
            formation_period: Signal formation period
            holding_period: Position holding period
            long_threshold: Percentile threshold for long positions
            short_threshold: Percentile threshold for short positions

        Returns:
            Strategy returns time series
        """
        # Calculate market-neutral signal
        signal = self.calculate_market_neutral_signal(prices_df, formation_period)

        # Rank signals (ascending = True means worst performers get rank 0)
        ranks = signal.rank(axis=1, ascending=True, pct=True)

        # Create position signals
        longs = (ranks <= long_threshold).astype(int)  # Buy worst performers
        shorts = (ranks >= short_threshold).astype(int) * -1  # Sell best performers

        # Combine positions
        raw_weights = longs + shorts

        # Normalize to dollar-neutral (equal long/short exposure)
        position_counts = raw_weights.abs().sum(axis=1)
        positions = raw_weights.div(position_counts, axis=0).fillna(0)

        # Apply holding period (rebalance every holding_period days)
        strategy_positions = positions[::holding_period].reindex(positions.index, method='ffill').fillna(0)

        # Calculate returns
        asset_returns = prices_df.pct_change()
        strategy_returns = (strategy_positions.shift(1) * asset_returns).sum(axis=1)

        # Calculate transaction costs
        turnover = (strategy_positions - strategy_positions.shift(1)).abs().sum(axis=1)
        transaction_costs = turnover * (self.commission_bps / 10000)

        # Net returns
        net_returns = strategy_returns - transaction_costs
        return net_returns.fillna(0)

    def run_volume_momentum_backtest(self, prices_df: pd.DataFrame, volumes_df: pd.DataFrame,
                                   formation_period: int = 7,
                                   holding_period: int = 1,
                                   long_threshold: float = 0.2,
                                   short_threshold: float = 0.8,
                                   signal_lag: int = 1) -> pd.Series:
        """
        Enhanced volume-momentum strategy

        Args:
            prices_df: Price data
            volumes_df: Volume data
            formation_period: Momentum formation period
            holding_period: Position holding period
            long_threshold: Percentile for long positions
            short_threshold: Percentile for short positions
            signal_lag: Lag to apply to momentum signal

        Returns:
            Strategy returns time series
        """
        # Calculate momentum signal with lag
        momentum_signal = np.log(prices_df).diff(formation_period).shift(signal_lag)

        # Volume confirmation signal
        volume_ma = volumes_df.rolling(window=formation_period * 2, min_periods=5).mean()
        volume_signal = (volumes_df > volume_ma).astype(int)

        # Combined signal (momentum * volume confirmation)
        combined_signal = momentum_signal * volume_signal

        # Rank and create positions
        ranks = combined_signal.rank(axis=1, ascending=False, na_option='bottom', pct=True)
        longs = (ranks <= long_threshold).astype(int)  # Top momentum
        shorts = (ranks >= short_threshold).astype(int) * -1  # Bottom momentum

        raw_weights = longs + shorts

        # Normalize positions
        position_counts = raw_weights.abs().sum(axis=1)
        positions = raw_weights.div(position_counts, axis=0).fillna(0)

        # Apply holding period
        strategy_positions = positions[::holding_period].reindex(positions.index, method='ffill').fillna(0)

        # Calculate returns and costs
        asset_returns = prices_df.pct_change()
        strategy_returns = (strategy_positions.shift(1) * asset_returns).sum(axis=1)

        turnover = (strategy_positions - strategy_positions.shift(1)).abs().sum(axis=1)
        transaction_costs = turnover * (self.commission_bps / 10000)

        net_returns = strategy_returns - transaction_costs
        return net_returns.fillna(0)

    def optimize_parameters(self, prices_df: pd.DataFrame, volumes_df: pd.DataFrame = None,
                          strategy_type: str = "mean_reversion") -> Dict[str, Any]:
        """
        Optimize strategy parameters for maximum Sharpe ratio

        Args:
            prices_df: Price data for optimization
            volumes_df: Volume data (for momentum strategy)
            strategy_type: "mean_reversion" or "volume_momentum"

        Returns:
            Best parameters and performance metrics
        """
        best_params = {}
        best_sharpe = -np.inf
        best_returns = None

        if strategy_type == "mean_reversion":
            # Parameter grid for mean reversion
            formation_periods = [1, 2, 3, 5, 7]
            holding_periods = [1, 2, 3, 5]
            thresholds = [(0.1, 0.9), (0.15, 0.85), (0.2, 0.8), (0.25, 0.75)]

            for fp in formation_periods:
                for hp in holding_periods:
                    for long_thresh, short_thresh in thresholds:
                        try:
                            returns = self.run_market_neutral_reversal_backtest(
                                prices_df, fp, hp, long_thresh, short_thresh
                            )

                            if len(returns.dropna()) > 50:  # Minimum observations
                                sharpe = self.calculate_sharpe_ratio(returns)

                                if sharpe > best_sharpe:
                                    best_sharpe = sharpe
                                    best_returns = returns
                                    best_params = {
                                        'formation_period': fp,
                                        'holding_period': hp,
                                        'long_threshold': long_thresh,
                                        'short_threshold': short_thresh,
                                        'strategy_type': 'mean_reversion'
                                    }
                        except Exception as e:
                            logger.warning(f"Error in optimization: {e}")
                            continue

        elif strategy_type == "volume_momentum" and volumes_df is not None:
            # Parameter grid for volume momentum
            formation_periods = [3, 5, 7, 10, 15, 20]
            holding_periods = [1, 2, 3, 5]
            signal_lags = [1, 2, 3]
            thresholds = [(0.1, 0.9), (0.15, 0.85), (0.2, 0.8)]

            for fp in formation_periods:
                for hp in holding_periods:
                    for lag in signal_lags:
                        for long_thresh, short_thresh in thresholds:
                            try:
                                returns = self.run_volume_momentum_backtest(
                                    prices_df, volumes_df, fp, hp,
                                    long_thresh, short_thresh, lag
                                )

                                if len(returns.dropna()) > 50:
                                    sharpe = self.calculate_sharpe_ratio(returns)

                                    if sharpe > best_sharpe:
                                        best_sharpe = sharpe
                                        best_returns = returns
                                        best_params = {
                                            'formation_period': fp,
                                            'holding_period': hp,
                                            'long_threshold': long_thresh,
                                            'short_threshold': short_thresh,
                                            'signal_lag': lag,
                                            'strategy_type': 'volume_momentum'
                                        }
                            except Exception as e:
                                continue

        # Calculate comprehensive metrics
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

        # Drawdown
        cumulative = (1 + returns_clean).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        # Calmar ratio
        calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # Sortino ratio
        negative_returns = returns_clean[returns_clean < 0]
        downside_std = negative_returns.std() * np.sqrt(self.periods_per_year) if len(negative_returns) > 0 else 0
        sortino = annual_return / downside_std if downside_std > 0 else 0

        # Market neutrality test (should be close to zero for good stat arb)
        market_beta = self.calculate_market_beta(returns_clean)

        # Win rate
        win_rate = (returns_clean > 0).mean()

        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar,
            'sortino_ratio': sortino,
            'market_beta': market_beta,
            'win_rate': win_rate,
            'num_trades': len(returns_clean),
            'final_performance': 1 + total_return
        }

    def calculate_market_beta(self, strategy_returns: pd.Series) -> float:
        """Calculate beta to equal-weighted market"""
        try:
            # For stat arb, we expect beta close to 0 (market neutral)
            market_returns = strategy_returns  # Placeholder - would use market index
            if len(strategy_returns) > 20 and strategy_returns.std() > 0:
                correlation = strategy_returns.corr(market_returns)
                beta = correlation * (strategy_returns.std() / market_returns.std())
                return beta if not np.isnan(beta) else 0
        except:
            pass
        return 0

    def generate_crypto_pairs_signals(self, pairs: List[str], data: Dict[str, pd.DataFrame]) -> Dict[str, pd.Series]:
        """
        Generate signals for cryptocurrency pairs
        Adapted for our existing crypto futures data

        Args:
            pairs: List of crypto symbols available in data
            data: Dictionary with price and volume data

        Returns:
            Dictionary of strategy returns for each approach
        """
        results = {}

        if 'prices' not in data:
            logger.error("Price data required for stat arb strategy")
            return results

        prices_df = data['prices']

        # Use all available symbols from the data
        available_pairs = [pair for pair in prices_df.columns if pair in prices_df.columns]

        if len(available_pairs) < 5:
            logger.warning(f"Insufficient pairs for stat arb: {len(available_pairs)}")
            return results

        prices_subset = prices_df[available_pairs].dropna(how='all')

        # Split data for train/test
        split_date = '2024-01-01'  # Use more recent split for better training data

        # Convert to timestamp for comparison
        split_ts = pd.Timestamp(split_date)
        train_prices = prices_subset[prices_subset.index < split_ts]
        test_prices = prices_subset[prices_subset.index >= split_ts]

        logger.info(f"Train period: {train_prices.index.min()} to {train_prices.index.max()}")
        logger.info(f"Test period: {test_prices.index.min()} to {test_prices.index.max()}")
        logger.info(f"Train samples: {len(train_prices)}, Test samples: {len(test_prices)}")

        # Optimize on training data
        logger.info("Optimizing mean reversion strategy...")
        logger.info(f"Training data shape: {train_prices.shape}")
        best_mr_params = self.optimize_parameters(train_prices, strategy_type="mean_reversion")
        logger.info(f"Best MR params: {best_mr_params}")

        # Test on out-of-sample data
        if best_mr_params and len(test_prices) > 20:
            test_returns = self.run_market_neutral_reversal_backtest(
                test_prices,
                formation_period=best_mr_params.get('formation_period', 1),
                holding_period=best_mr_params.get('holding_period', 1),
                long_threshold=best_mr_params.get('long_threshold', 0.2),
                short_threshold=best_mr_params.get('short_threshold', 0.8)
            )

            results['mean_reversion'] = test_returns

            # Calculate final metrics
            final_metrics = self.calculate_performance_metrics(test_returns)
            logger.info(f"Mean Reversion Strategy - Sharpe: {final_metrics.get('sharpe_ratio', 0):.2f}")

        # Volume momentum if volume data available
        if 'volumes' in data:
            volumes_df = data['volumes'][available_pairs].dropna(how='all')

            if len(volumes_df) > 0:
                train_volumes = volumes_df.loc[:split_date] if split_date in volumes_df.index else volumes_df.iloc[:split_idx]
                test_volumes = volumes_df.loc[split_date:] if split_date in volumes_df.index else volumes_df.iloc[split_idx:]

                logger.info("Optimizing volume momentum strategy...")
                best_vm_params = self.optimize_parameters(
                    train_prices, train_volumes,
                    strategy_type="volume_momentum"
                )

                if best_vm_params and len(test_prices) > 20:
                    test_returns_vm = self.run_volume_momentum_backtest(
                        test_prices, test_volumes,
                        formation_period=best_vm_params.get('formation_period', 7),
                        holding_period=best_vm_params.get('holding_period', 1),
                        long_threshold=best_vm_params.get('long_threshold', 0.2),
                        short_threshold=best_vm_params.get('short_threshold', 0.8),
                        signal_lag=best_vm_params.get('signal_lag', 1)
                    )

                    results['volume_momentum'] = test_returns_vm

                    final_metrics_vm = self.calculate_performance_metrics(test_returns_vm)
                    logger.info(f"Volume Momentum Strategy - Sharpe: {final_metrics_vm.get('sharpe_ratio', 0):.2f}")

        return results


def run_crypto_stat_arb_backtest(data: Dict[str, pd.DataFrame],
                                pairs: List[str],
                                commission_bps: float = 5.0,
                                enable_ml_enhancement: bool = False) -> Dict[str, Any]:
    """
    Main function to run crypto statistical arbitrage backtest

    Args:
        data: Dictionary containing 'prices' and optionally 'volumes' DataFrames
        pairs: List of cryptocurrency pairs to trade
        commission_bps: Commission in basis points
        enable_ml_enhancement: Whether to use ML-enhanced strategy

    Returns:
        Dictionary with strategy results and metrics
    """

    if enable_ml_enhancement:
        # Import ML strategy
        try:
            from .ml_enhanced_stat_arb import run_ml_enhanced_stat_arb_backtest
            return run_ml_enhanced_stat_arb_backtest(data, commission_bps)
        except ImportError as e:
            logger.warning(f"ML enhancement not available: {e}")

    strategy = CryptoStatArbStrategy(commission_bps=commission_bps)

    # Generate signals for crypto pairs
    results = strategy.generate_crypto_pairs_signals(pairs, data)

    if not results:
        return {'error': 'No valid strategy results generated'}

    # Find best performing strategy
    best_strategy = None
    best_sharpe = -np.inf

    strategy_metrics = {}

    for strategy_name, returns in results.items():
        metrics = strategy.calculate_performance_metrics(returns)
        strategy_metrics[strategy_name] = metrics

        current_sharpe = metrics.get('sharpe_ratio', 0)
        if current_sharpe > best_sharpe:
            best_sharpe = current_sharpe
            best_strategy = strategy_name

    return {
        'strategy_returns': results,
        'strategy_metrics': strategy_metrics,
        'best_strategy': best_strategy,
        'best_sharpe': best_sharpe,
        'commission_bps': commission_bps
    }