#!/usr/bin/env python3

"""
Machine Learning Enhanced Statistical Arbitrage Strategy
Advanced feature engineering and regime detection for 2+ Sharpe target
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class MLEnhancedStatArbStrategy:
    """
    Machine Learning Enhanced Statistical Arbitrage Strategy
    Features:
    - Advanced feature engineering
    - Regime detection
    - Dynamic position sizing
    - Anomaly detection
    - Ensemble predictions
    """

    def __init__(self, commission_bps: float = 1.0):
        self.commission_bps = commission_bps
        self.periods_per_year = 365
        self.scaler = StandardScaler()
        self.regime_model = KMeans(n_clusters=3, random_state=42)
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.signal_model = RandomForestRegressor(n_estimators=100, random_state=42)

    def engineer_features(self, prices_df: pd.DataFrame, volumes_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Advanced feature engineering for statistical arbitrage
        """
        features_dict = {}

        # Basic price features
        returns = prices_df.pct_change()
        log_prices = np.log(prices_df)

        for symbol in prices_df.columns:
            prefix = symbol.replace('USDT', '')

            # Price-based features
            features_dict[f'{prefix}_return_1d'] = returns[symbol]
            features_dict[f'{prefix}_return_5d'] = returns[symbol].rolling(5).sum()
            features_dict[f'{prefix}_return_20d'] = returns[symbol].rolling(20).sum()

            # Volatility features
            features_dict[f'{prefix}_vol_5d'] = returns[symbol].rolling(5).std()
            features_dict[f'{prefix}_vol_20d'] = returns[symbol].rolling(20).std()
            features_dict[f'{prefix}_vol_60d'] = returns[symbol].rolling(60).std()

            # Momentum features
            features_dict[f'{prefix}_momentum_5d'] = log_prices[symbol].diff(5)
            features_dict[f'{prefix}_momentum_20d'] = log_prices[symbol].diff(20)

            # Mean reversion features
            ma_5 = log_prices[symbol].rolling(5).mean()
            ma_20 = log_prices[symbol].rolling(20).mean()
            features_dict[f'{prefix}_deviation_ma5'] = log_prices[symbol] - ma_5
            features_dict[f'{prefix}_deviation_ma20'] = log_prices[symbol] - ma_20

            # Relative strength features
            market_return = returns.mean(axis=1)
            features_dict[f'{prefix}_relative_strength'] = returns[symbol] - market_return

            # Volatility-adjusted returns
            vol_adj_return = returns[symbol] / (returns[symbol].rolling(20).std() + 1e-6)
            features_dict[f'{prefix}_vol_adj_return'] = vol_adj_return

            # Volume features (if available)
            if volumes_df is not None and symbol in volumes_df.columns:
                volume = volumes_df[symbol]
                features_dict[f'{prefix}_volume_ma_ratio'] = volume / volume.rolling(20).mean()
                features_dict[f'{prefix}_price_volume_trend'] = returns[symbol] * np.log(volume + 1)

        # Cross-sectional features
        returns_rank = returns.rank(axis=1, pct=True)
        vol_rank = returns.rolling(20).std().rank(axis=1, pct=True)

        for i, symbol in enumerate(prices_df.columns):
            prefix = symbol.replace('USDT', '')
            features_dict[f'{prefix}_return_rank'] = returns_rank[symbol]
            features_dict[f'{prefix}_vol_rank'] = vol_rank[symbol]

        # Market-wide features
        features_dict['market_return'] = returns.mean(axis=1)
        features_dict['market_vol'] = returns.std(axis=1)
        features_dict['market_skew'] = returns.skew(axis=1)
        features_dict['market_correlation'] = returns.corrwith(returns.mean(axis=1), axis=0).mean()

        # Regime features
        market_vol_regime = returns.std(axis=1).rolling(60).mean()
        features_dict['vol_regime'] = market_vol_regime
        features_dict['vol_regime_change'] = market_vol_regime.diff()

        return pd.DataFrame(features_dict)

    def detect_market_regimes(self, features_df: pd.DataFrame) -> pd.Series:
        """
        Detect market regimes using unsupervised learning
        """
        # Select regime-relevant features
        regime_features = [
            'market_vol', 'vol_regime', 'vol_regime_change',
            'market_return', 'market_skew'
        ]

        available_features = [f for f in regime_features if f in features_df.columns]

        if len(available_features) < 3:
            return pd.Series(0, index=features_df.index)

        regime_data = features_df[available_features].dropna()

        if len(regime_data) < 60:
            return pd.Series(0, index=features_df.index)

        # Fit regime detection model
        regime_data_scaled = self.scaler.fit_transform(regime_data)
        regimes = self.regime_model.fit_predict(regime_data_scaled)

        # Create full series
        regime_series = pd.Series(0, index=features_df.index)
        regime_series.loc[regime_data.index] = regimes

        # Forward fill regime assignments
        regime_series = regime_series.replace(0, np.nan).fillna(method='ffill').fillna(0)

        return regime_series

    def generate_ml_signals(self, features_df: pd.DataFrame, prices_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate ML-enhanced signals for statistical arbitrage
        """
        signals_dict = {}

        # Calculate future returns for training
        returns = prices_df.pct_change()
        future_returns = {}

        for symbol in prices_df.columns:
            future_returns[symbol] = returns[symbol].shift(-1)  # Next period return

        # Prepare features for ML model
        feature_cols = [col for col in features_df.columns if not col.startswith('market_')]

        for symbol in prices_df.columns:
            prefix = symbol.replace('USDT', '')
            symbol_features = [col for col in feature_cols if col.startswith(prefix)]

            if len(symbol_features) < 3:
                continue

            # Prepare training data
            X = features_df[symbol_features].dropna()
            y = future_returns[symbol].loc[X.index]

            # Remove NaN targets
            valid_idx = ~y.isna()
            X = X.loc[valid_idx]
            y = y.loc[valid_idx]

            if len(X) < 100:  # Need sufficient data
                continue

            # Split for training (use first 70% for training)
            split_idx = int(len(X) * 0.7)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

            try:
                # Fit ML model
                model = RandomForestRegressor(n_estimators=50, random_state=42)
                model.fit(X_train, y_train)

                # Generate signals for all data
                predictions = model.predict(X)

                # Convert predictions to signals
                signal_series = pd.Series(0.0, index=features_df.index)
                signal_series.loc[X.index] = predictions

                signals_dict[symbol] = signal_series

            except Exception as e:
                logger.warning(f"ML signal generation failed for {symbol}: {e}")
                continue

        return pd.DataFrame(signals_dict)

    def run_ml_enhanced_backtest(self, prices_df: pd.DataFrame, volumes_df: pd.DataFrame = None,
                                lookback_period: int = 60, rebalance_frequency: int = 1,
                                position_sizing_method: str = "equal_weight") -> pd.Series:
        """
        Run ML-enhanced statistical arbitrage backtest
        """
        # Engineer features
        features_df = self.engineer_features(prices_df, volumes_df)

        # Detect market regimes
        regimes = self.detect_market_regimes(features_df)

        # Generate ML signals
        ml_signals = self.generate_ml_signals(features_df, prices_df)

        if ml_signals.empty:
            logger.warning("No ML signals generated")
            return pd.Series(0, index=prices_df.index)

        # Calculate position weights
        positions_list = []

        for i in range(lookback_period, len(prices_df), rebalance_frequency):
            current_date = prices_df.index[i]

            # Get signals for current date
            current_signals = ml_signals.loc[current_date]
            current_regime = regimes.loc[current_date]

            # Filter valid signals
            valid_signals = current_signals.dropna()

            if len(valid_signals) < 2:
                positions = pd.Series(0, index=prices_df.columns)
            else:
                # Rank signals
                signal_ranks = valid_signals.rank(ascending=False, pct=True)

                # Create long/short positions based on regime
                if current_regime == 0:  # Low volatility regime - mean reversion
                    longs = signal_ranks <= 0.2  # Bottom 20% (contrarian)
                    shorts = signal_ranks >= 0.8  # Top 20% (contrarian)
                elif current_regime == 1:  # Medium volatility regime - momentum
                    longs = signal_ranks >= 0.8  # Top 20% (momentum)
                    shorts = signal_ranks <= 0.2  # Bottom 20% (momentum)
                else:  # High volatility regime - neutral
                    longs = signal_ranks >= 0.6
                    shorts = signal_ranks <= 0.4

                # Position sizing
                if position_sizing_method == "equal_weight":
                    long_weight = 0.5 / longs.sum() if longs.sum() > 0 else 0
                    short_weight = -0.5 / shorts.sum() if shorts.sum() > 0 else 0
                elif position_sizing_method == "volatility_weighted":
                    # Use recent volatility for position sizing
                    recent_vol = prices_df.pct_change().iloc[i-20:i].std()
                    vol_weights = 1 / (recent_vol + 1e-6)
                    vol_weights = vol_weights / vol_weights.sum()

                    long_weight = 0.5 * vol_weights[longs] / vol_weights[longs].sum() if longs.sum() > 0 else 0
                    short_weight = -0.5 * vol_weights[shorts] / vol_weights[shorts].sum() if shorts.sum() > 0 else 0
                else:  # signal_weighted
                    signal_strength = np.abs(valid_signals)
                    long_weights = signal_strength[longs]
                    short_weights = signal_strength[shorts]

                    long_weight = 0.5 * long_weights / long_weights.sum() if len(long_weights) > 0 else 0
                    short_weight = -0.5 * short_weights / short_weights.sum() if len(short_weights) > 0 else 0

                # Create position vector
                positions = pd.Series(0.0, index=prices_df.columns)

                if isinstance(long_weight, (int, float)):
                    positions[longs] = long_weight
                else:
                    positions.loc[longs.index] = long_weight

                if isinstance(short_weight, (int, float)):
                    positions[shorts] = short_weight
                else:
                    positions.loc[shorts.index] = short_weight

            positions_list.append((current_date, positions))

        # Convert to position DataFrame
        position_dates = [pos[0] for pos in positions_list]
        position_data = [pos[1] for pos in positions_list]

        position_df = pd.DataFrame(position_data, index=position_dates)
        position_df = position_df.reindex(prices_df.index, method='ffill').fillna(0)

        # Calculate returns
        asset_returns = prices_df.pct_change()
        strategy_returns = (position_df.shift(1) * asset_returns).sum(axis=1)

        # Calculate transaction costs
        turnover = (position_df - position_df.shift(1)).abs().sum(axis=1)
        transaction_costs = turnover * (self.commission_bps / 10000)

        # Net returns
        net_returns = strategy_returns - transaction_costs
        return net_returns.fillna(0)

    def optimize_ml_strategy(self, prices_df: pd.DataFrame, volumes_df: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Optimize ML-enhanced strategy parameters
        """
        best_params = {}
        best_sharpe = -np.inf
        best_returns = None

        # Parameter grid
        lookback_periods = [30, 45, 60, 90]
        rebalance_frequencies = [1, 2, 3, 5]
        position_sizing_methods = ["equal_weight", "volatility_weighted", "signal_weighted"]

        # Split data
        split_idx = int(len(prices_df) * 0.7)
        train_prices = prices_df.iloc[:split_idx]
        test_prices = prices_df.iloc[split_idx:]

        if volumes_df is not None:
            train_volumes = volumes_df.iloc[:split_idx]
            test_volumes = volumes_df.iloc[split_idx:]
        else:
            train_volumes = test_volumes = None

        for lookback in lookback_periods:
            for rebal_freq in rebalance_frequencies:
                for pos_sizing in position_sizing_methods:
                    try:
                        # Test on training data
                        train_returns = self.run_ml_enhanced_backtest(
                            train_prices, train_volumes, lookback, rebal_freq, pos_sizing
                        )

                        if len(train_returns.dropna()) > 50:
                            train_sharpe = self.calculate_sharpe_ratio(train_returns)

                            if train_sharpe > 0.5:  # Filter promising candidates
                                # Test on validation data
                                test_returns = self.run_ml_enhanced_backtest(
                                    test_prices, test_volumes, lookback, rebal_freq, pos_sizing
                                )

                                test_sharpe = self.calculate_sharpe_ratio(test_returns)

                                if test_sharpe > best_sharpe:
                                    best_sharpe = test_sharpe
                                    best_returns = test_returns
                                    best_params = {
                                        'lookback_period': lookback,
                                        'rebalance_frequency': rebal_freq,
                                        'position_sizing_method': pos_sizing,
                                        'strategy_type': 'ml_enhanced_stat_arb'
                                    }

                    except Exception as e:
                        logger.warning(f"Error in ML optimization: {e}")
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

        # Additional metrics
        calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # Sortino ratio
        negative_returns = returns_clean[returns_clean < 0]
        downside_std = negative_returns.std() * np.sqrt(self.periods_per_year) if len(negative_returns) > 0 else 0
        sortino = annual_return / downside_std if downside_std > 0 else 0

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
            'win_rate': win_rate,
            'num_trades': len(returns_clean),
            'final_performance': 1 + total_return
        }


def run_ml_enhanced_stat_arb_backtest(data: Dict[str, pd.DataFrame],
                                     commission_bps: float = 1.0) -> Dict[str, Any]:
    """
    Main function to run ML-enhanced statistical arbitrage backtest
    """

    strategy = MLEnhancedStatArbStrategy(commission_bps=commission_bps)

    prices_df = data['prices']
    volumes_df = data.get('volumes', None)

    if len(prices_df.columns) < 4:
        return {'error': 'Insufficient assets for ML-enhanced stat arb strategy'}

    # Run optimization
    best_params = strategy.optimize_ml_strategy(prices_df, volumes_df)

    if not best_params:
        return {'error': 'No valid ML strategy parameters found'}

    # Run final backtest with best parameters
    final_returns = strategy.run_ml_enhanced_backtest(
        prices_df, volumes_df,
        lookback_period=best_params.get('lookback_period', 60),
        rebalance_frequency=best_params.get('rebalance_frequency', 1),
        position_sizing_method=best_params.get('position_sizing_method', 'equal_weight')
    )

    # Calculate final metrics
    final_metrics = strategy.calculate_performance_metrics(final_returns)

    return {
        'strategy_returns': {'ml_enhanced_stat_arb': final_returns},
        'strategy_metrics': {'ml_enhanced_stat_arb': final_metrics},
        'best_strategy': 'ml_enhanced_stat_arb',
        'best_sharpe': final_metrics.get('sharpe_ratio', 0),
        'best_params': best_params,
        'commission_bps': commission_bps
    }