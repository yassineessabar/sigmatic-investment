#!/usr/bin/env python3

"""
ML-Enhanced Relative Momentum Strategy
Combines traditional relative momentum with machine learning signal prediction
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


def calculate_technical_features(base_data: pd.DataFrame, alt_data: pd.DataFrame,
                                lookback: int = 20) -> pd.DataFrame:
    """Calculate technical indicators for ML features"""

    base_close = base_data['close']
    alt_close = alt_data['close']

    # Relative ratio features
    relative_ratio = base_close / alt_close
    features = pd.DataFrame(index=base_data.index)

    # 1. Momentum features
    features['ratio'] = relative_ratio
    features['ratio_ema_5'] = relative_ratio.ewm(span=5).mean()
    features['ratio_ema_10'] = relative_ratio.ewm(span=10).mean()
    features['ratio_ema_20'] = relative_ratio.ewm(span=20).mean()
    features['ratio_sma_10'] = relative_ratio.rolling(10).mean()
    features['ratio_sma_20'] = relative_ratio.rolling(20).mean()

    # 2. Momentum indicators
    features['ratio_roc_5'] = relative_ratio.pct_change(5)
    features['ratio_roc_10'] = relative_ratio.pct_change(10)
    features['ratio_momentum'] = relative_ratio / relative_ratio.shift(10) - 1

    # 3. Volatility features
    features['ratio_std_10'] = relative_ratio.rolling(10).std()
    features['ratio_std_20'] = relative_ratio.rolling(20).std()
    features['ratio_volatility'] = relative_ratio.rolling(20).std() / relative_ratio.rolling(20).mean()

    # 4. Price position features
    features['ratio_percentile_20'] = relative_ratio.rolling(20).rank(pct=True)
    features['ratio_zscore'] = (relative_ratio - relative_ratio.rolling(20).mean()) / relative_ratio.rolling(20).std()

    # 5. Individual asset features
    # Base asset features
    features['base_rsi'] = calculate_rsi(base_close, 14)
    features['base_momentum'] = base_close.pct_change(10)
    features['base_volatility'] = base_close.rolling(20).std() / base_close.rolling(20).mean()

    # Alt asset features
    features['alt_rsi'] = calculate_rsi(alt_close, 14)
    features['alt_momentum'] = alt_close.pct_change(10)
    features['alt_volatility'] = alt_close.rolling(20).std() / alt_close.rolling(20).mean()

    # 6. Cross-asset features
    features['momentum_divergence'] = features['base_momentum'] - features['alt_momentum']
    features['volatility_ratio'] = features['base_volatility'] / features['alt_volatility']
    features['rsi_divergence'] = features['base_rsi'] - features['alt_rsi']

    # 7. Regime detection features
    features['vol_regime'] = (features['ratio_std_20'] > features['ratio_std_20'].rolling(50).quantile(0.7)).astype(int)
    features['trend_strength'] = abs(features['ratio_ema_5'] - features['ratio_ema_20']) / features['ratio_ema_20']

    return features


def calculate_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """Calculate RSI indicator"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def prepare_ml_dataset(features: pd.DataFrame, target_window: int = 5) -> Tuple[pd.DataFrame, pd.Series]:
    """Prepare dataset for ML training with future returns as target"""

    # Calculate future returns for target
    future_returns = features['ratio'].pct_change(target_window).shift(-target_window)

    # Create binary target: 1 if future returns > 0, 0 otherwise
    target = (future_returns > 0).astype(int)

    # Remove target-related features from input
    X = features.drop(['ratio'], axis=1, errors='ignore')

    # Drop rows with NaN values
    valid_idx = X.dropna().index.intersection(target.dropna().index)
    X = X.loc[valid_idx]
    y = target.loc[valid_idx]

    return X, y


def train_simple_models(X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
    """Train simple statistical models for signal prediction"""

    models = {}

    try:
        # 1. Simple correlation-based model
        correlations = {}
        for col in X.columns:
            if not X[col].isna().all() and len(X[col].dropna()) > 10:
                corr = np.corrcoef(X[col].dropna(), y.loc[X[col].dropna().index])[0, 1]
                correlations[col] = corr if not np.isnan(corr) else 0

        # 2. Weighted average model based on correlations
        feature_weights = pd.Series(correlations).fillna(0)
        feature_weights = feature_weights / (feature_weights.abs().sum() + 1e-8)  # Normalize

        models['correlation_weights'] = feature_weights

        # 3. Simple moving average crossover predictor
        ma_signals = {}
        for col in X.columns:
            if not X[col].isna().all():
                ma_short = X[col].rolling(5).mean()
                ma_long = X[col].rolling(20).mean()
                ma_signal = (ma_short > ma_long).astype(int)
                ma_accuracy = (ma_signal.shift(1) == y).mean()
                ma_signals[col] = {'signal': ma_signal, 'accuracy': ma_accuracy}

        models['moving_average_signals'] = ma_signals

        # 4. Simple linear regression coefficients
        linear_coeffs = {}
        for col in X.columns:
            if not X[col].isna().all() and len(X[col].dropna()) > 20:
                x_vals = X[col].dropna().values.reshape(-1, 1)
                y_vals = y.loc[X[col].dropna().index].values

                # Simple linear regression: y = ax + b
                if len(x_vals) > 0:
                    x_mean = np.mean(x_vals)
                    y_mean = np.mean(y_vals)

                    numerator = np.sum((x_vals.flatten() - x_mean) * (y_vals - y_mean))
                    denominator = np.sum((x_vals.flatten() - x_mean) ** 2)

                    slope = numerator / (denominator + 1e-8)
                    intercept = y_mean - slope * x_mean

                    linear_coeffs[col] = {'slope': slope, 'intercept': intercept}

        models['linear_coefficients'] = linear_coeffs

        # 5. Feature importance based on variance and correlation
        feature_importance = pd.Series(index=X.columns, dtype=float)
        for col in X.columns:
            if col in correlations:
                variance_score = X[col].var() if not X[col].isna().all() else 0
                correlation_score = abs(correlations[col])
                combined_score = correlation_score * (1 + np.log1p(variance_score))
                feature_importance[col] = combined_score

        feature_importance = feature_importance.fillna(0)
        feature_importance = feature_importance / (feature_importance.sum() + 1e-8)
        models['feature_importance'] = feature_importance

    except Exception as e:
        logger.warning(f"Error training simple models: {e}")
        return {}

    return models


def get_ml_signal_confidence(models: Dict[str, Any], features: pd.DataFrame) -> float:
    """Get simple statistical ensemble prediction confidence"""

    if not models or features.empty:
        return 0.5  # Neutral signal

    try:
        # Get latest feature vector
        latest_features = features.iloc[-1].fillna(0)

        predictions = []

        # 1. Correlation-weighted prediction
        if 'correlation_weights' in models:
            weights = models['correlation_weights']
            # Weighted sum of features
            weighted_score = 0
            for col in latest_features.index:
                if col in weights:
                    weighted_score += latest_features[col] * weights[col]

            # Convert to probability (sigmoid-like function)
            prob = 1 / (1 + np.exp(-weighted_score))
            predictions.append(prob)

        # 2. Moving average signal
        if 'moving_average_signals' in models:
            ma_votes = []
            for col, ma_data in models['moving_average_signals'].items():
                if col in latest_features.index and 'accuracy' in ma_data:
                    accuracy = ma_data['accuracy']
                    if not np.isnan(accuracy) and accuracy > 0.5:  # Only use if better than random
                        # Simple threshold-based signal
                        feature_val = latest_features[col]
                        if not np.isnan(feature_val):
                            # Normalize feature and convert to vote
                            normalized_val = np.tanh(feature_val)  # Bound between -1 and 1
                            vote = (normalized_val + 1) / 2  # Convert to 0-1 range
                            ma_votes.append(vote * accuracy)  # Weight by accuracy

            if ma_votes:
                predictions.append(np.mean(ma_votes))

        # 3. Linear regression prediction
        if 'linear_coefficients' in models:
            linear_preds = []
            for col, coeffs in models['linear_coefficients'].items():
                if col in latest_features.index:
                    feature_val = latest_features[col]
                    if not np.isnan(feature_val):
                        pred = coeffs['slope'] * feature_val + coeffs['intercept']
                        # Convert to probability
                        prob = 1 / (1 + np.exp(-pred))
                        linear_preds.append(prob)

            if linear_preds:
                predictions.append(np.mean(linear_preds))

        if predictions:
            # Ensemble average
            ensemble_confidence = np.mean(predictions)
            # Bound the result
            ensemble_confidence = np.clip(ensemble_confidence, 0.1, 0.9)
            return ensemble_confidence

    except Exception as e:
        logger.warning(f"Error getting statistical prediction: {e}")

    return 0.5


def backtest_ml_enhanced_momentum(base_data: pd.DataFrame, alt_data: pd.DataFrame,
                                 ema_window: int = 10, allocation_weight: float = 0.75,
                                 fees: float = 0.0004, slippage: float = 0.0005,
                                 freq: int = 365, base_funding_data: pd.DataFrame = None,
                                 alt_funding_data: pd.DataFrame = None,
                                 ml_weight: float = 0.3, min_train_size: int = 200) -> Dict[str, Any]:
    """
    Backtest ML-enhanced relative momentum strategy

    Args:
        ml_weight: Weight of ML signal vs traditional signal (0.0 = pure traditional, 1.0 = pure ML)
        min_train_size: Minimum number of samples needed to start using ML
    """

    # Align data
    common_idx = base_data.index.intersection(alt_data.index)
    base_aligned = base_data.loc[common_idx]
    alt_aligned = alt_data.loc[common_idx]

    if len(common_idx) < max(ema_window + 20, min_train_size):
        logger.warning("Insufficient data for ML-enhanced momentum strategy")
        return {'returns': pd.Series(dtype=float), 'final_performance': 1.0, 'ema_window': ema_window}

    # Calculate technical features
    features = calculate_technical_features(base_aligned, alt_aligned)

    # Initialize variables
    base_prices = base_aligned['close']
    alt_prices = alt_aligned['close']
    relative_ratio = base_prices / alt_prices
    ema_ratio = relative_ratio.ewm(span=ema_window, adjust=False).mean()

    positions = []
    trades = []
    current_position = 0  # -1: short ratio, 0: neutral, 1: long ratio

    # Rolling ML training
    ml_models = {}

    for i in range(max(ema_window, min_train_size), len(common_idx)):
        current_idx = common_idx[i]

        # Traditional signal
        current_ratio = relative_ratio.iloc[i]
        current_ema = ema_ratio.iloc[i]
        traditional_signal = 1 if current_ratio > current_ema else -1

        # ML signal (retrain every 50 days)
        ml_signal = 0
        if i >= min_train_size and i % 50 == 0:
            # Prepare training data
            train_features = features.iloc[:i-10]  # Leave some gap to avoid lookahead
            X_train, y_train = prepare_ml_dataset(train_features)

            if len(X_train) >= min_train_size:
                ml_models = train_simple_models(X_train, y_train)

        # Get ML prediction
        if ml_models and i < len(features):
            current_features = features.iloc[:i+1]
            ml_confidence = get_ml_signal_confidence(ml_models, current_features)
            ml_signal = 1 if ml_confidence > 0.55 else -1 if ml_confidence < 0.45 else 0

        # Combined signal
        if ml_signal != 0:
            combined_signal = (1 - ml_weight) * traditional_signal + ml_weight * ml_signal
            final_signal = 1 if combined_signal > 0.2 else -1 if combined_signal < -0.2 else 0
        else:
            final_signal = traditional_signal

        # Enhanced signal with additional filters
        rsi_base = features['base_rsi'].iloc[i] if i < len(features) else 50
        rsi_alt = features['alt_rsi'].iloc[i] if i < len(features) else 50

        # RSI filter: avoid extreme conditions
        if final_signal == 1 and rsi_base > 80:
            final_signal = 0
        elif final_signal == -1 and rsi_alt > 80:
            final_signal = 0

        # Position sizing based on volatility
        vol_factor = 1.0
        if i < len(features) and not pd.isna(features['ratio_volatility'].iloc[i]):
            current_vol = features['ratio_volatility'].iloc[i]
            vol_percentile = features['ratio_volatility'].iloc[:i+1].rank(pct=True).iloc[-1]
            if vol_percentile > 0.8:  # High volatility
                vol_factor = 0.5
            elif vol_percentile < 0.2:  # Low volatility
                vol_factor = 1.2

        target_position = final_signal * allocation_weight * vol_factor

        # Record position
        positions.append({
            'date': current_idx,
            'position': target_position,
            'traditional_signal': traditional_signal,
            'ml_signal': ml_signal,
            'final_signal': final_signal,
            'ratio': current_ratio,
            'ema': current_ema
        })

        current_position = target_position

    if not positions:
        return {'returns': pd.Series(dtype=float), 'final_performance': 1.0, 'ema_window': ema_window}

    # Convert to DataFrame for easier processing
    positions_df = pd.DataFrame(positions).set_index('date')

    # Calculate returns
    returns = calculate_strategy_returns(
        positions_df, relative_ratio, base_aligned, alt_aligned,
        fees, slippage, base_funding_data, alt_funding_data
    )

    # Calculate performance metrics
    ann_return, ann_vol, sharpe, max_dd = compute_ml_metrics(returns, freq)
    final_performance = (1 + returns).prod()

    return {
        'returns': returns,
        'final_performance': final_performance,
        'ann_return': ann_return,
        'ann_vol': ann_vol,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'ema_window': ema_window,
        'positions': positions_df,
        'ml_models': ml_models,
        'optimizations_enabled': True
    }


def calculate_strategy_returns(positions_df: pd.DataFrame, relative_ratio: pd.Series,
                              base_data: pd.DataFrame, alt_data: pd.DataFrame,
                              fees: float, slippage: float,
                              base_funding_data: pd.DataFrame = None,
                              alt_funding_data: pd.DataFrame = None) -> pd.Series:
    """Calculate strategy returns from positions"""

    returns = []
    prev_position = 0

    for i in range(len(positions_df)):
        current_position = positions_df['position'].iloc[i]
        current_date = positions_df.index[i]

        # Calculate return from ratio change
        if i > 0:
            ratio_return = relative_ratio.pct_change().loc[current_date]
            position_return = prev_position * ratio_return

            # Add transaction costs
            position_change = abs(current_position - prev_position)
            transaction_cost = position_change * (fees + slippage)

            # Add funding costs if available
            funding_cost = 0
            if base_funding_data is not None and alt_funding_data is not None:
                try:
                    base_funding = base_funding_data.loc[base_funding_data.index <= current_date, 'funding_rate'].iloc[-1] if len(base_funding_data.loc[base_funding_data.index <= current_date]) > 0 else 0
                    alt_funding = alt_funding_data.loc[alt_funding_data.index <= current_date, 'funding_rate'].iloc[-1] if len(alt_funding_data.loc[alt_funding_data.index <= current_date]) > 0 else 0
                    funding_cost = abs(prev_position) * (base_funding - alt_funding) / (365 * 3)  # Daily funding
                except:
                    funding_cost = 0

            total_return = position_return - transaction_cost - funding_cost
            returns.append(total_return)
        else:
            returns.append(0)

        prev_position = current_position

    return pd.Series(returns, index=positions_df.index)


def compute_ml_metrics(returns: pd.Series, freq: int = 365) -> Tuple[float, float, float, float]:
    """Compute performance metrics for ML strategy"""

    if len(returns) == 0:
        return 0, 0, 0, 0

    # Annualized return
    ann_return = (1 + returns).prod() ** (freq / len(returns)) - 1

    # Annualized volatility
    ann_vol = returns.std() * np.sqrt(freq)

    # Sharpe ratio
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0
    sharpe = np.clip(sharpe, -10, 10)

    # Maximum drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_dd = drawdown.min()

    return ann_return, ann_vol, sharpe, max_dd


def optimize_ml_enhanced_strategy(base_data: pd.DataFrame, alt_data: pd.DataFrame,
                                 ema_windows: List[int], allocation_weight: float = 0.75,
                                 fees: float = 0.0004, slippage: float = 0.0005,
                                 freq: int = 365, base_funding_data: pd.DataFrame = None,
                                 alt_funding_data: pd.DataFrame = None,
                                 optimization_metric: str = 'sharpe') -> Dict[str, Any]:
    """Optimize ML-enhanced strategy parameters"""

    best_result = None
    best_score = -np.inf

    # Test different ML weights and EMA windows
    ml_weights = [0.0, 0.2, 0.3, 0.4, 0.5]

    for ema_window in ema_windows:
        for ml_weight in ml_weights:
            try:
                result = backtest_ml_enhanced_momentum(
                    base_data, alt_data, ema_window, allocation_weight,
                    fees, slippage, freq, base_funding_data, alt_funding_data,
                    ml_weight
                )

                if optimization_metric == 'sharpe':
                    score = result['sharpe']
                elif optimization_metric == 'calmar':
                    score = result['ann_return'] / abs(result['max_dd']) if result['max_dd'] != 0 else 0
                else:
                    score = result['sharpe']

                if score > best_score:
                    best_score = score
                    best_result = result
                    best_result['ml_weight'] = ml_weight

            except Exception as e:
                logger.warning(f"Error optimizing EMA {ema_window}, ML weight {ml_weight}: {e}")
                continue

    return best_result if best_result else {
        'returns': pd.Series(dtype=float),
        'final_performance': 1.0,
        'ema_window': ema_windows[0] if ema_windows else 10,
        'ml_weight': 0.3
    }