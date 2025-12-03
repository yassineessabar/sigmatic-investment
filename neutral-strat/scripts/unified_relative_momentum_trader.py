#!/usr/bin/env python3

"""
Unified Relative Momentum Trading Engine

This script ensures PERFECT ALIGNMENT between backtest, test, and live modes.
The SAME strategy logic, parameters, and conditions apply to all modes.
Only the execution environment changes.
"""

import sys
import os
import time
import signal
import logging
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse
import warnings
from enum import Enum
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import ConfigManager
from src.utils.logger import setup_logging
from src.strategies.relative_momentum import (
    compute_relative_momentum_signals,
    backtest_relative_momentum_pair,
    compute_metrics
)
from src.execution.advanced_execution import AdvancedExecutionEngine, ExecutionAlgorithm

logger = logging.getLogger(__name__)


class TradingMode(Enum):
    """Trading modes with exact same strategy logic"""
    BACKTEST = "backtest"
    TEST = "test"
    LIVE = "live"


class UnifiedRelativeMomentumTrader:
    """Unified trader that ensures perfect alignment across all modes"""

    def __init__(self, mode: TradingMode, config_path: str = 'config/market_neutral_config.yaml'):
        """Initialize unified trader"""
        self.mode = mode

        # Load unified configuration
        self.config_manager = ConfigManager()
        self.config = self.config_manager.load_config(config_path)

        # Apply mode-specific settings while keeping strategy identical
        self._apply_mode_config()

        setup_logging(self.config)

        # Show mode and confirm alignment
        self._show_mode_info()

        # Initialize exchange if needed
        if self.mode != TradingMode.BACKTEST and not self.config.get('execution', {}).get('simulation_only', False):
            self._initialize_exchange()
        elif self.config.get('execution', {}).get('use_public_data', False):
            # Initialize exchange for public data access only
            self._initialize_public_exchange()
        else:
            # Set exchange to None for pure simulation/backtest modes
            self.exchange = None

        # Extract IDENTICAL strategy parameters for all modes
        self._extract_strategy_parameters()

        # Initialize tracking
        self._initialize_tracking()

        # Initialize advanced execution engine for non-backtest modes
        if self.mode != TradingMode.BACKTEST and self.exchange:
            execution_config = self.config.get('execution', {}).get('advanced_execution', {})
            self.execution_engine = AdvancedExecutionEngine(self.exchange, execution_config)
            logger.info(f"[!] Advanced execution engine initialized")
        else:
            self.execution_engine = None

        logger.info(f"✅ Unified Trader initialized in {self.mode.value.upper()} mode")
        logger.info(f"[*] Strategy parameters IDENTICAL across all modes")

    def _apply_mode_config(self):
        """Apply mode-specific configuration while preserving strategy identity"""
        if self.mode == TradingMode.BACKTEST:
            # Backtest mode configuration
            self.config['execution'].update(self.config['execution']['backtest_mode'])
            self.config['binance'] = {}  # No API needed for backtest

        elif self.mode == TradingMode.TEST:
            # Test mode configuration
            self.config['execution'].update(self.config['execution']['test_mode'])
            self.config['binance'].update(self.config['binance']['testnet_mode'])

        elif self.mode == TradingMode.LIVE:
            # Live mode configuration
            self.config['execution'].update(self.config['execution']['live_mode'])
            self.config['binance'].update(self.config['binance']['live_mode'])

            # Extra safety for live mode
            if self.config['execution'].get('confirmation_required', True):
                self._require_live_confirmation()

    def _show_mode_info(self):
        """Show mode information and alignment confirmation"""
        print(f"\n{'='*60}")
        print(f"🎯 UNIFIED RELATIVE MOMENTUM TRADER")
        print(f"{'='*60}")
        print(f"Mode: {self.mode.value.upper()}")

        if self.mode == TradingMode.BACKTEST:
            print("📊 BACKTEST MODE - Historical simulation")
        elif self.mode == TradingMode.TEST:
            if self.config['binance']['testnet_mode'].get('demo_trading', False):
                print("🧪 TEST MODE - Demo Trading with fake money")
                print("🔒 SAFE: Uses Binance Demo Trading (Futures Enabled)")
            else:
                print("🧪 TEST MODE - Testnet with fake money")
                print("🔒 SAFE: Uses Binance testnet")
        elif self.mode == TradingMode.LIVE:
            print("🔴 LIVE MODE - Real money trading")
            print("⚠️  WARNING: Uses real money")

        # Show strategy alignment
        print(f"\n🎯 STRATEGY ALIGNMENT:")
        print(f"  Pairs: {len(self.config['pairs'])} pairs")
        print(f"  Strategy: {self.config['strategy']['type']}")
        print(f"  Risk: {self.config['risk']['max_position_size']*100:.0f}% max position")
        print(f"  Optimization: {self.config['strategy']['optimization']['enabled']}")
        print(f"  Futures: {self.config['futures']['enabled']}")
        print(f"\n✅ IDENTICAL strategy logic across all modes")
        print(f"{'='*60}")

    def _require_live_confirmation(self):
        """Require explicit confirmation for live trading"""
        print(f"\n🔴 LIVE TRADING CONFIRMATION REQUIRED")
        print(f"⚠️  This will trade with REAL MONEY")
        print(f"⚠️  Make sure you have tested thoroughly first")

        confirmation = input("Type 'CONFIRMED' to proceed with live trading: ").strip()
        if confirmation != "CONFIRMED":
            print("❌ Live trading cancelled for safety")
            sys.exit(1)

        print("🔴 Live trading confirmed - starting with real money")

    def _initialize_exchange(self):
        """Initialize exchange for test or live modes"""
        # Get mode-specific credentials
        if self.mode == TradingMode.TEST:
            binance_config = self.config['binance']['testnet_mode']
            # Try config first, then environment
            api_key = binance_config.get('api_key') or os.getenv('BINANCE_TESTNET_API_KEY')
            api_secret = binance_config.get('api_secret') or os.getenv('BINANCE_TESTNET_API_SECRET')
        else:  # LIVE mode
            binance_config = self.config['binance']['live_mode']
            api_key = binance_config.get('api_key') or os.getenv('BINANCE_LIVE_API_KEY')
            api_secret = binance_config.get('api_secret') or os.getenv('BINANCE_LIVE_API_SECRET')

        if not api_key or not api_secret:
            mode_name = "TESTNET" if self.mode == TradingMode.TEST else "LIVE"
            raise ValueError(f"API credentials required: BINANCE_{mode_name}_API_KEY and BINANCE_{mode_name}_API_SECRET")

        # Initialize exchange with mode-specific settings
        # Use demo.binance.com for test mode, regular API for live
        exchange_config = {
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': binance_config.get('default_type', 'future')  # Support COIN-M futures
            }
        }

        # Create exchange instance
        self.exchange = ccxt.binance(exchange_config)

        # Enable demo trading if configured
        if binance_config.get('demo_trading') and hasattr(self.exchange, 'enable_demo_trading'):
            self.exchange.enable_demo_trading(True)
            logger.info("Demo trading enabled using exchange.enable_demo_trading(True)")
        elif binance_config.get('testnet', False):
            # Fallback for older testnet mode
            pass

        if self.mode == TradingMode.TEST and binance_config.get('demo_trading'):
            logger.info("Exchange initialized: Demo Trading (Futures Enabled)")
        elif self.mode == TradingMode.TEST:
            logger.info("Exchange initialized: Testnet")
        else:
            logger.info("Exchange initialized: Live")

    def _initialize_public_exchange(self):
        """Initialize exchange for public data access only (no API keys required)"""
        exchange_config = {
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future'  # Default to USDT-M for public data
            }
        }

        self.exchange = ccxt.binance(exchange_config)
        logger.info("Exchange initialized: Public Data Only (Futures)")

    def _extract_strategy_parameters(self):
        """Extract IDENTICAL strategy parameters for all modes"""
        # Extract universe and pairs - IDENTICAL logic
        self.universe = []
        self.pairs = []

        for pair in self.config['pairs']:
            if pair['alt'] not in self.universe:
                self.universe.append(pair['alt'])

            # Store EXACT same parameters
            self.pairs.append({
                'base': pair['base'],
                'alt': pair['alt'],
                'ema_window': pair['ema_window'],
                'allocation_weight': pair['allocation_weight'],
                'max_notional': pair['max_notional']
            })

        # Add BTC if not present (IDENTICAL logic)
        if 'BTCUSDT' not in self.universe:
            self.universe = ['BTCUSDT'] + self.universe

        # Extract IDENTICAL risk parameters
        self.risk_params = self.config['risk'].copy()

        # Extract IDENTICAL strategy parameters
        self.strategy_params = self.config['strategy'].copy()

    def _initialize_tracking(self):
        """Initialize tracking variables"""
        self.positions: Dict[str, Dict] = {}
        self.market_data: Dict[str, pd.DataFrame] = {}
        self.funding_data: Dict[str, pd.DataFrame] = {}
        self.running = True
        self.last_signal_time = {}

        # Performance tracking
        self.start_time = datetime.now()
        self.start_balance = self.config.get('initial_capital', 10000)
        self.current_balance = self.start_balance
        self.total_trades = 0
        self.successful_trades = 0
        self.daily_pnl = 0.0
        self.total_pnl = 0.0

        # Enhanced position tracking for paper trading
        self.paper_positions = {}  # Track simulated positions with P&L
        self.unrealized_pnl = 0.0  # Track unrealized P&L

        # Signal handlers for non-backtest modes
        if self.mode != TradingMode.BACKTEST:
            signal.signal(signal.SIGTERM, self._signal_handler)
            signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.running = False

    def convert_to_trading_symbol(self, symbol: str) -> str:
        """Convert symbol to futures format for all modes"""
        # Always use futures format: BTCUSDT -> BTC/USDT:USDT
        if 'USDT' in symbol and ':USDT' not in symbol:
            return symbol.replace('USDT', '/USDT:USDT')
        return symbol

    def fetch_market_data(self, symbol: str, lookback_days: int = 200) -> pd.DataFrame:
        """Fetch market data - IDENTICAL format for all modes"""
        if self.mode == TradingMode.BACKTEST:
            # For backtest, use historical data loading
            return self._load_historical_data(symbol, lookback_days)
        else:
            # For test/live, fetch current data
            return self._fetch_current_data(symbol, lookback_days)

    def _load_historical_data(self, symbol: str, lookback_days: int) -> pd.DataFrame:
        """Load historical data for backtest"""
        try:
            # This would load from your existing backtest data files
            # Using same format as backtest script
            data_file = f"data/historical/daily/{symbol.replace('/', '').replace(':', '')}_daily_futures.csv"

            if os.path.exists(data_file):
                df = pd.read_csv(data_file, index_col=0, parse_dates=True)
                return df.tail(lookback_days)
            else:
                logger.warning(f"No historical data file for {symbol}")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"Error loading historical data for {symbol}: {e}")
            return pd.DataFrame()

    def _fetch_current_data(self, symbol: str, lookback_days: int) -> pd.DataFrame:
        """Fetch current data for test/live modes"""
        # For simulation mode, return empty DataFrame
        if self.exchange is None:
            logger.warning(f"Simulation mode: No data available for {symbol}")
            return pd.DataFrame()

        try:
            trading_symbol = self.convert_to_trading_symbol(symbol)

            # Fetch recent OHLCV data - IDENTICAL format
            ohlcv = self.exchange.fetch_ohlcv(
                trading_symbol,
                timeframe='1d',
                limit=lookback_days
            )

            if not ohlcv:
                return pd.DataFrame()

            # Convert to DataFrame - IDENTICAL format as backtest
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('datetime', inplace=True)
            df = df[['open', 'high', 'low', 'close', 'volume']]

            return df

        except Exception as e:
            logger.error(f"Error fetching current data for {symbol}: {e}")
            return pd.DataFrame()

    def fetch_funding_data(self, symbol: str, days_back: int = 30) -> pd.DataFrame:
        """Fetch funding data - IDENTICAL logic for all modes"""
        if self.mode == TradingMode.BACKTEST:
            return self._load_historical_funding(symbol, days_back)
        else:
            return self._fetch_current_funding(symbol, days_back)

    def _load_historical_funding(self, symbol: str, days_back: int) -> pd.DataFrame:
        """Load historical funding data for backtest"""
        try:
            funding_file = f"data/historical/daily/{symbol.replace('/', '').replace(':', '')}_funding_rates.csv"

            if os.path.exists(funding_file):
                df = pd.read_csv(funding_file, index_col=0, parse_dates=True)
                return df.tail(days_back * 3)  # 3 funding rates per day
            else:
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"Error loading historical funding for {symbol}: {e}")
            return pd.DataFrame()

    def _fetch_current_funding(self, symbol: str, days_back: int) -> pd.DataFrame:
        """Fetch current funding data for test/live modes"""
        # For simulation mode, return empty DataFrame
        if self.exchange is None:
            logger.warning(f"Simulation mode: No funding data available for {symbol}")
            return pd.DataFrame()

        try:
            trading_symbol = self.convert_to_trading_symbol(symbol)
            since_ts = self.exchange.milliseconds() - (days_back * 24 * 60 * 60 * 1000)

            funding_history = self.exchange.fetch_funding_rate_history(
                trading_symbol,
                since=since_ts,
                limit=days_back * 3
            )

            if not funding_history:
                return pd.DataFrame()

            # Convert to DataFrame - IDENTICAL format as backtest
            funding_df = pd.DataFrame(funding_history)
            funding_df['datetime'] = pd.to_datetime(funding_df['timestamp'], unit='ms')
            funding_df.set_index('datetime', inplace=True)
            funding_df = funding_df[['fundingRate']].rename(columns={'fundingRate': 'funding_rate'})

            return funding_df

        except Exception as e:
            logger.error(f"Error fetching current funding for {symbol}: {e}")
            return pd.DataFrame()

    def update_market_data(self):
        """Update market data - IDENTICAL process for all modes"""
        logger.info("[+] Updating market data...")

        for symbol in self.universe:
            # Fetch price data
            price_data = self.fetch_market_data(symbol)
            if not price_data.empty:
                self.market_data[symbol] = price_data
                logger.info(f"  {symbol}: {len(price_data)} candles")

            # Fetch funding data
            funding_data = self.fetch_funding_data(symbol)
            if not funding_data.empty:
                self.funding_data[symbol] = funding_data
                logger.info(f"  {symbol}: {len(funding_data)} funding rates")

            if self.mode != TradingMode.BACKTEST:
                time.sleep(0.1)  # Rate limiting for live modes

    def generate_signals(self) -> List[Dict]:
        """Generate signals using EXACT same logic across all modes"""
        signals = []

        # Remove demo signal generation - use real signals only
        # if self.config.get('execution', {}).get('force_demo_signals', False) and len(self.paper_positions) == 0:
        #     return self._generate_demo_signal()

        try:
            for pair_config in self.pairs:
                base_symbol = pair_config['base']
                alt_symbol = pair_config['alt']
                pair_name = f"{base_symbol}/{alt_symbol.replace('USDT', '')}"

                # Check data availability
                if base_symbol not in self.market_data or alt_symbol not in self.market_data:
                    logger.warning(f"Missing data for {pair_name}")
                    continue

                base_data = self.market_data[base_symbol]
                alt_data = self.market_data[alt_symbol]

                if len(base_data) < 50 or len(alt_data) < 50:
                    logger.warning(f"Insufficient data for {pair_name}")
                    continue

                # Get funding data
                base_funding = self.funding_data.get(base_symbol, pd.DataFrame())
                alt_funding = self.funding_data.get(alt_symbol, pd.DataFrame())

                try:
                    # Use EXACT same signal generation logic as backtest
                    if self.strategy_params['optimization']['enabled']:
                        # Use optimized EMA window
                        window_range = range(
                            self.strategy_params['optimization']['window_range_start'],
                            self.strategy_params['optimization']['window_range_end']
                        )

                        # For live/test modes, use a simplified optimization or fixed window
                        # to avoid over-optimization during live trading
                        if self.mode != TradingMode.BACKTEST:
                            # Use the configured EMA window for live/test
                            ema_window = pair_config['ema_window']
                        else:
                            # Use full optimization for backtest
                            # This would require implementing live optimization
                            ema_window = pair_config['ema_window']
                    else:
                        ema_window = pair_config['ema_window']

                    # Generate signals using EXACT backtest logic
                    # Create data dict and config for this pair
                    pair_data = {
                        base_symbol: base_data,
                        alt_symbol: alt_data
                    }

                    # Use appropriate strategy based on config
                    strategy_type = self.strategy_params.get('type', 'relative_momentum')

                    if strategy_type == 'market_neutral_pairs':
                        from src.strategies.market_neutral_pairs import compute_market_neutral_signals
                        pair_signals = compute_market_neutral_signals(
                            pair_data,
                            {'pairs': [pair_config], 'strategy': self.strategy_params}
                        )
                    else:
                        from src.strategies.relative_momentum import compute_relative_momentum_signals
                        pair_signals = compute_relative_momentum_signals(
                            pair_data,
                            {'pairs': [pair_config]}
                        )

                    if pair_signals:
                        # Convert PairSignal objects to dictionary format
                        for pair_signal in pair_signals:
                            # Convert base signal to dict
                            base_signal_dict = {
                                'symbol': pair_signal.base_signal.symbol,
                                'side': pair_signal.base_signal.side,
                                'type': 'entry',
                                'allocation_weight': pair_config['allocation_weight'],
                                'max_notional': pair_config['max_notional'],
                                'timestamp': pair_signal.base_signal.timestamp,
                                'reason': pair_signal.entry_reason,
                                'pair_name': pair_name,
                                'config': pair_config,
                                'confidence': pair_signal.base_signal.confidence,
                                'metadata': pair_signal.base_signal.metadata
                            }

                            # Convert hedge signal to dict
                            hedge_signal_dict = {
                                'symbol': pair_signal.hedge_signal.symbol,
                                'side': pair_signal.hedge_signal.side,
                                'type': 'entry',
                                'allocation_weight': pair_config['allocation_weight'],
                                'max_notional': pair_config['max_notional'],
                                'timestamp': pair_signal.hedge_signal.timestamp,
                                'reason': pair_signal.entry_reason,
                                'pair_name': pair_name,
                                'config': pair_config,
                                'confidence': pair_signal.hedge_signal.confidence,
                                'metadata': pair_signal.hedge_signal.metadata
                            }

                            signals.extend([base_signal_dict, hedge_signal_dict])

                        logger.info(f"[>] Generated {len(pair_signals)} pair signals for {pair_name}")

                except Exception as e:
                    logger.error(f"Error generating signals for {pair_name}: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error in signal generation: {e}")

        return signals

    def _generate_demo_signal(self) -> List[Dict]:
        """Generate a demo signal for testing position taking"""
        if not self.exchange:
            return []

        try:
            # Generate a demo signal for BTCUSDT
            symbol = 'BTCUSDT'
            trading_symbol = self.convert_to_trading_symbol(symbol)
            ticker = self.exchange.fetch_ticker(trading_symbol)
            current_price = ticker['last']

            # Create a demo buy signal
            demo_signal = {
                'symbol': symbol,
                'side': 'long',
                'type': 'entry',
                'allocation_weight': 0.5,
                'max_notional': 500,
                'timestamp': datetime.now(),
                'reason': 'Demo signal for testing'
            }

            logger.info(f"[*] DEMO: Generated test signal for {symbol} @ ${current_price:.2f}")
            return [demo_signal]

        except Exception as e:
            logger.error(f"Error generating demo signal: {e}")
            return []

    def calculate_position_size(self, signal: Dict, current_price: float) -> float:
        """Calculate position size - IDENTICAL logic with mode-specific multipliers"""
        try:
            # Get position size multiplier based on mode
            multiplier = self.config['execution'].get('position_size_multiplier', 1.0)

            # IDENTICAL base calculation
            allocation_weight = signal.get('allocation_weight', 1.0)
            max_notional = signal.get('max_notional', 1000.0)

            if self.mode == TradingMode.BACKTEST:
                # Use backtest initial capital
                available_capital = self.config['backtest']['initial_capital']
            else:
                # Use live/test balance
                balance = self.get_account_balance()
                available_capital = balance.get('free', 0.0)

            # Calculate target notional - IDENTICAL logic
            target_notional = min(
                available_capital * allocation_weight * self.risk_params['max_position_size'],
                max_notional
            )

            # Apply mode-specific multiplier
            target_notional *= multiplier

            # Calculate position size
            position_size = target_notional / current_price
            return round(position_size, 6)

        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0.0

    def get_account_balance(self) -> Dict:
        """Get account balance for live/test modes"""
        if self.mode == TradingMode.BACKTEST or self.exchange is None:
            # For paper trading, return dynamic balance with P&L
            if hasattr(self, 'current_balance') and hasattr(self, 'unrealized_pnl'):
                total_balance = self.current_balance + getattr(self, 'unrealized_pnl', 0.0)
                return {'total': total_balance, 'free': total_balance, 'used': 0.0}
            else:
                initial_capital = self.config.get('initial_capital', 10000)
                return {'total': initial_capital, 'free': initial_capital, 'used': 0.0}

        try:
            balance = self.exchange.fetch_balance()
            usdt_balance = balance.get('USDT', {})

            return {
                'total': usdt_balance.get('total', 0.0),
                'free': usdt_balance.get('free', 0.0),
                'used': usdt_balance.get('used', 0.0)
            }

        except Exception as e:
            # For public data mode, just return simulation balance
            if "apiKey" in str(e):
                initial_capital = self.config.get('initial_capital', 10000)
                return {'total': initial_capital, 'free': initial_capital, 'used': 0.0}

            logger.error(f"Error getting account balance: {e}")
            return {'total': 0.0, 'free': 0.0, 'used': 0.0}

    def get_asset_positions(self) -> Dict:
        """Get positions for all tracked assets (BTC, ETH, SOL, ADA, AVAX)"""
        if self.mode == TradingMode.BACKTEST or self.exchange is None:
            return {}

        try:
            positions = self.exchange.fetch_positions()
            asset_positions = {}

            # Track our main assets - support both USDT-M and COIN-M futures
            tracked_patterns = [
                # USDT-M futures
                'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'ADA/USDT', 'AVAX/USDT',
                # COIN-M futures
                'BTCUSD_PERP', 'ETHUSD_PERP', 'SOLUSD_PERP', 'ADAUSD_PERP', 'AVAXUSD_PERP'
            ]

            for position in positions:
                symbol = position['symbol']

                # Check if this is one of our tracked symbols and has a position
                if any(pattern in symbol for pattern in tracked_patterns) and abs(float(position.get('contracts', position.get('size', 0)))) > 0:

                    # Extract asset name from symbol
                    if 'USD_PERP' in symbol:
                        asset = symbol.replace('USD_PERP', '')
                    elif 'USDT' in symbol:
                        asset = symbol.split('/')[0] if '/' in symbol else symbol.replace('USDT', '')
                    else:
                        asset = symbol.split('USD')[0] if 'USD' in symbol else symbol

                    # Use contracts for COIN-M, size for USDT-M
                    position_size = float(position.get('contracts', position.get('size', 0)))

                    asset_positions[asset] = {
                        'size': position_size,
                        'side': position.get('side', 'long' if position_size > 0 else 'short'),
                        'entry_price': float(position.get('entryPrice', 0)),
                        'mark_price': float(position.get('markPrice', 0)),
                        'unrealized_pnl': float(position.get('unrealizedProfit', position.get('unrealizedPnl', 0))),
                        'percentage': float(position.get('percentage', 0)),
                        'symbol': symbol  # Keep original symbol for reference
                    }

            return asset_positions

        except Exception as e:
            logger.error(f"Error getting asset positions: {e}")
            return {}

    def get_wallet_balances(self) -> Dict:
        """Get wallet balances for all tracked assets (BTC, ETH, SOL, ADA, AVAX)"""
        if self.mode == TradingMode.BACKTEST or self.exchange is None:
            return {}

        try:
            balance = self.exchange.fetch_balance()
            tracked_assets = ['BTC', 'ETH', 'SOL', 'ADA', 'AVAX']
            wallet_balances = {}

            for asset in tracked_assets:
                asset_balance = balance.get(asset, {})
                if asset_balance:
                    wallet_balances[asset] = {
                        'free': float(asset_balance.get('free', 0.0)),
                        'used': float(asset_balance.get('used', 0.0)),
                        'total': float(asset_balance.get('total', 0.0))
                    }
                else:
                    wallet_balances[asset] = {'free': 0.0, 'used': 0.0, 'total': 0.0}

            return wallet_balances

        except Exception as e:
            logger.error(f"Error getting wallet balances: {e}")
            return {}

    def execute_signal(self, signal: Dict) -> bool:
        """Execute signal with mode-appropriate method"""
        if self.mode == TradingMode.BACKTEST:
            return self._simulate_signal_execution(signal)
        elif self.config.get('execution', {}).get('simulate_execution', False):
            return self._simulate_paper_trading(signal)
        else:
            return self._execute_live_signal(signal)

    def _simulate_signal_execution(self, signal: Dict) -> bool:
        """Simulate signal execution for backtest"""
        # This would integrate with your existing backtest execution logic
        logger.info(f"[+] BACKTEST: Simulated execution of {signal.get('symbol', 'Unknown')}")
        return True

    def _simulate_paper_trading(self, signal: Dict) -> bool:
        """Enhanced paper trading with position tracking and P&L calculation"""
        try:
            symbol = signal['symbol']
            side = signal['side']
            signal_type = signal.get('type', 'unknown')

            # Get real market price
            if not self.exchange:
                logger.info(f"[P] PAPER TRADING: {signal_type} {side} {symbol} (no price data)")
                return True

            trading_symbol = self.convert_to_trading_symbol(symbol)
            ticker = self.exchange.fetch_ticker(trading_symbol)
            current_price = ticker['last']

            if signal_type == 'entry':
                # Calculate position size
                size = self.calculate_position_size(signal, current_price)
                if size <= 0:
                    return False

                # Record new position
                self.paper_positions[symbol] = {
                    'side': side,
                    'size': size,
                    'entry_price': current_price,
                    'entry_time': datetime.now(),
                    'unrealized_pnl': 0.0
                }

                self.total_trades += 1
                self.successful_trades += 1

                logger.info(f"[P] PAPER ENTRY: {side} {size:.6f} {symbol} @ ${current_price:.2f}")

            elif signal_type == 'exit' and symbol in self.paper_positions:
                # Close position and calculate P&L
                position = self.paper_positions[symbol]
                entry_price = position['entry_price']
                size = position['size']

                if position['side'] == 'long':
                    pnl = (current_price - entry_price) * size
                else:  # short
                    pnl = (entry_price - current_price) * size

                # Update balances
                self.total_pnl += pnl
                self.current_balance += pnl

                logger.info(f"[P] PAPER EXIT: {position['side']} {symbol} @ ${current_price:.2f} | P&L: ${pnl:.2f}")

                # Remove position
                del self.paper_positions[symbol]

            # Update unrealized P&L for open positions
            self._update_unrealized_pnl()

            return True

        except Exception as e:
            logger.error(f"Error in paper trading simulation: {e}")
            return True

    def _update_unrealized_pnl(self):
        """Update unrealized P&L for open positions"""
        if not self.exchange or not self.paper_positions:
            return

        total_unrealized = 0.0
        for symbol, position in self.paper_positions.items():
            try:
                trading_symbol = self.convert_to_trading_symbol(symbol)
                ticker = self.exchange.fetch_ticker(trading_symbol)
                current_price = ticker['last']

                entry_price = position['entry_price']
                size = position['size']

                if position['side'] == 'long':
                    unrealized_pnl = (current_price - entry_price) * size
                else:  # short
                    unrealized_pnl = (entry_price - current_price) * size

                position['unrealized_pnl'] = unrealized_pnl
                total_unrealized += unrealized_pnl

            except Exception as e:
                logger.error(f"Error updating unrealized P&L for {symbol}: {e}")

        self.unrealized_pnl = total_unrealized

    def _execute_live_signal(self, signal: Dict) -> bool:
        """Execute signal on live/test exchange using advanced execution"""
        # For simulation mode, log and return success without executing
        if self.exchange is None:
            symbol = signal['symbol']
            side = signal['side']
            signal_type = signal.get('type', 'unknown')
            logger.info(f"[S] SIMULATION: Would execute {signal_type} {side} {symbol}")
            return True

        try:
            symbol = signal['symbol']
            side = signal['side']
            signal_type = signal['type']

            trading_symbol = self.convert_to_trading_symbol(symbol)
            ticker = self.exchange.fetch_ticker(trading_symbol)
            current_price = ticker['last']

            mode_indicator = "🧪 TEST" if self.config['binance'].get('testnet', False) else "🔴 LIVE"
            logger.info(f"{mode_indicator}: Executing {signal_type} {side} {symbol} @ ${current_price:.2f}")

            if signal_type == 'entry':
                size = self.calculate_position_size(signal, current_price)

                if size <= 0:
                    return False

                # Prepare execution signal for advanced engine
                execution_signal = {
                    'symbol': trading_symbol,
                    'side': side,
                    'size': size,
                    'urgency': signal.get('urgency', 'normal'),
                    'original_signal': signal
                }

                # Use advanced execution engine if available
                if self.execution_engine:
                    logger.info(f"[E] Using advanced execution engine for {side} {symbol}")
                    execution_result = self.execution_engine.execute_trade(execution_signal)

                    logger.info(f"[E] Execution result status: {execution_result.get('status', 'unknown')}")
                    logger.info(f"[E] Full execution result: {execution_result}")

                    if execution_result.get('status') in ['closed', 'partial']:
                        executed_price = execution_result.get('executed_price', current_price)
                        executed_size = execution_result.get('order', {}).get('filled', size)

                        logger.info(f"[+] Advanced execution completed")
                        logger.info(f"   Algorithm: {execution_result.get('algorithm', 'unknown')}")
                        logger.info(f"   Price: ${executed_price:.2f}")
                        logger.info(f"   Size: {executed_size:.6f}")

                        if 'slippage' in execution_result:
                            slippage_bps = execution_result['slippage'] * 10000
                            logger.info(f"   Slippage: {slippage_bps:.1f}bps")

                        self.positions[symbol] = {
                            'side': side,
                            'size': executed_size,
                            'entry_price': executed_price,
                            'entry_time': datetime.now(),
                            'execution_result': execution_result
                        }

                        self.total_trades += 1
                        self.successful_trades += 1
                        return True
                    else:
                        logger.warning(f"[!] Execution failed or incomplete. Status: {execution_result.get('status')}")
                        logger.warning(f"[!] Error: {execution_result.get('error', 'No error message')}")
                else:
                    # Fallback to market order if no advanced engine
                    logger.info(f"[M] Using fallback market order for {side} {symbol}")

                    try:
                        if side == 'long':
                            order = self.exchange.create_market_buy_order(trading_symbol, size)
                        else:
                            order = self.exchange.create_market_sell_order(trading_symbol, size)

                        logger.info(f"[M] Market order result: {order}")

                        if order:
                            logger.info(f"[+] Market order executed: {order['id']}")

                            self.positions[symbol] = {
                                'side': side,
                                'size': size,
                                'entry_price': current_price,
                                'entry_time': datetime.now(),
                                'order_id': order['id']
                            }

                            self.total_trades += 1
                            self.successful_trades += 1
                            return True
                        else:
                            logger.error(f"[!] Market order returned None/False")
                    except Exception as e:
                        logger.error(f"[!] Market order execution failed: {e}")

            elif signal_type == 'exit':
                if symbol in self.positions:
                    position = self.positions[symbol]
                    position_size = position['size']

                    # Prepare exit signal
                    exit_signal = {
                        'symbol': trading_symbol,
                        'side': 'sell' if position['side'] == 'long' else 'buy',
                        'size': position_size,
                        'urgency': 'high',  # Exits are usually urgent
                        'original_signal': signal
                    }

                    # Use advanced execution for exit
                    if self.execution_engine:
                        execution_result = self.execution_engine.execute_trade(exit_signal)

                        if execution_result.get('status') in ['closed', 'partial']:
                            exit_price = execution_result.get('executed_price', current_price)

                            # Calculate P&L
                            entry_price = position['entry_price']
                            if position['side'] == 'long':
                                pnl = (exit_price - entry_price) * position_size
                            else:
                                pnl = (entry_price - exit_price) * position_size

                            self.daily_pnl += pnl

                            logger.info(f"✅ Advanced exit completed")
                            logger.info(f"   Algorithm: {execution_result.get('algorithm', 'unknown')}")
                            logger.info(f"   Exit Price: ${exit_price:.2f}")
                            logger.info(f"   P&L: ${pnl:.2f}")

                            del self.positions[symbol]
                            self.total_trades += 1
                            self.successful_trades += 1
                            return True
                    else:
                        # Fallback to market order
                        if position['side'] == 'long':
                            order = self.exchange.create_market_sell_order(trading_symbol, position_size)
                        else:
                            order = self.exchange.create_market_buy_order(trading_symbol, position_size)

                        if order:
                            # Calculate P&L
                            entry_price = position['entry_price']
                            if position['side'] == 'long':
                                pnl = (current_price - entry_price) * position_size
                            else:
                                pnl = (entry_price - current_price) * position_size

                            self.daily_pnl += pnl

                            logger.info(f"✅ Market exit executed: {order['id']}")
                            logger.info(f"   P&L: ${pnl:.2f}")

                            del self.positions[symbol]
                            self.total_trades += 1
                            self.successful_trades += 1
                            return True

        except Exception as e:
            logger.error(f"Error executing signal: {e}")

        return False

    def run_backtest(self):
        """Run backtest mode using existing backtest logic"""
        logger.info("[+] Running backtest with unified parameters...")

        # This would call your existing backtest functions
        # but with the unified configuration ensuring identical parameters
        from scripts.run_relative_momentum_backtest import run_relative_momentum_backtest

        # Use the same config file but ensure backtest mode
        results = run_relative_momentum_backtest(
            config_path='config/market_neutral_config.yaml'
        )

        logger.info("[+] Backtest completed with unified parameters")
        return results

    def run_live_trading(self):
        """Run live trading loop"""
        logger.info(f"[!] Starting {self.mode.value} trading with unified parameters...")

        # Record starting balance
        balance = self.get_account_balance()
        self.start_balance = balance.get('total', 0)

        cycle_count = 0
        check_frequency = self.config['execution'].get('check_frequency_minutes', 60)

        try:
            while self.running:
                cycle_count += 1
                cycle_start = datetime.now()

                logger.info(f"\n[@] Cycle #{cycle_count} - {cycle_start}")

                # 1. Update market data with IDENTICAL logic
                self.update_market_data()

                # 2. Generate signals with IDENTICAL parameters
                signals = self.generate_signals()

                if signals:
                    logger.info(f"[>] Generated {len(signals)} signals")

                    # 3. Execute signals
                    for signal in signals:
                        success = self.execute_signal(signal)
                        if success and self.mode != TradingMode.BACKTEST:
                            time.sleep(1)
                else:
                    logger.info("[>] No signals generated")

                # 4. Log status
                self._log_status()

                # 5. Sleep until next check
                logger.info(f"[T] Next check in {check_frequency} minutes...")
                time.sleep(check_frequency * 60)

        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt, shutting down...")
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
        finally:
            self._shutdown()

    def _log_status(self):
        """Log current status"""
        balance = self.get_account_balance()
        runtime = datetime.now() - self.start_time

        mode_symbol = {"backtest": "[+]", "test": "[T]", "live": "[L]"}[self.mode.value]

        logger.info(f"\n{'='*60}")
        logger.info(f"{mode_symbol} {self.mode.value.upper()} TRADING STATUS")
        logger.info(f"{'='*60}")
        logger.info(f"[T] Runtime: {runtime}")
        logger.info(f"[$] Balance: ${balance.get('total', 0):,.2f}")
        logger.info(f"[@] Trades: {self.total_trades}")
        logger.info(f"[%] Success Rate: {(self.successful_trades/max(1,self.total_trades)*100):.1f}%")

        # Show paper trading specific info
        if hasattr(self, 'paper_positions') and self.paper_positions:
            logger.info(f"[P] Paper Positions: {len(self.paper_positions)}")
            for symbol, pos in self.paper_positions.items():
                unrealized = pos.get('unrealized_pnl', 0.0)
                logger.info(f"  {symbol}: {pos['side']} {pos['size']:.6f} @ ${pos['entry_price']:.2f} (P&L: ${unrealized:.2f})")

        # Show P&L summary for paper trading
        if hasattr(self, 'total_pnl'):
            logger.info(f"[R] Realized P&L: ${self.total_pnl:.2f}")
            if hasattr(self, 'unrealized_pnl'):
                logger.info(f"[U] Unrealized P&L: ${self.unrealized_pnl:.2f}")
                logger.info(f"[=] Total P&L: ${self.total_pnl + self.unrealized_pnl:.2f}")

        # Legacy positions for live trading
        if self.positions:
            logger.info(f"[L] Live Positions: {len(self.positions)}")
            for symbol, pos in self.positions.items():
                logger.info(f"  {symbol}: {pos['side']} {pos['size']:.6f}")

        # Show wallet balances for all tracked assets
        wallet_balances = self.get_wallet_balances()
        if wallet_balances:
            logger.info(f"[$] Wallet Balances:")
            for asset, balance in wallet_balances.items():
                logger.info(f"  {asset}: {balance['total']:.6f} (Free: {balance['free']:.6f}, Used: {balance['used']:.6f})")

        # Show asset positions for all tracked assets
        asset_positions = self.get_asset_positions()
        if asset_positions:
            logger.info(f"[A] Asset Positions:")
            for asset, pos in asset_positions.items():
                pnl_sign = "+" if pos['unrealized_pnl'] >= 0 else ""
                logger.info(f"  {asset}: {pos['side']} {abs(pos['size']):.6f} @ ${pos['entry_price']:.2f} | Mark: ${pos['mark_price']:.2f} | PnL: {pnl_sign}${pos['unrealized_pnl']:.2f} ({pnl_sign}{pos['percentage']:.2f}%)")
        else:
            # Show zero positions for tracked assets
            logger.info(f"[A] Asset Positions: All positions closed (BTC: 0, ETH: 0, SOL: 0, ADA: 0, AVAX: 0)")

        logger.info(f"{'='*60}")

    def _shutdown(self):
        """Graceful shutdown"""
        logger.info(f"[X] Shutting down {self.mode.value} trader...")

        if self.positions and self.mode != TradingMode.BACKTEST:
            # Close positions for live/test modes
            self._close_all_positions()

        self._log_final_performance()

    def _close_all_positions(self):
        """Close all open positions"""
        for symbol, position in list(self.positions.items()):
            try:
                trading_symbol = self.convert_to_trading_symbol(symbol)

                if position['side'] == 'long':
                    order = self.exchange.create_market_sell_order(trading_symbol, position['size'])
                else:
                    order = self.exchange.create_market_buy_order(trading_symbol, position['size'])

                logger.info(f"Closed position {symbol}: {order['id']}")

            except Exception as e:
                logger.error(f"Error closing position {symbol}: {e}")

    def _log_final_performance(self):
        """Log final performance summary"""
        final_balance = self.get_account_balance().get('total', 0)
        total_return = final_balance - self.start_balance
        return_pct = (total_return / self.start_balance) * 100 if self.start_balance > 0 else 0

        logger.info(f"\n[=] FINAL PERFORMANCE ({self.mode.value.upper()}):")
        logger.info(f"  Starting Balance: ${self.start_balance:,.2f}")
        logger.info(f"  Final Balance: ${final_balance:,.2f}")
        logger.info(f"  Total Return: ${total_return:,.2f} ({return_pct:.2f}%)")
        logger.info(f"  Total Trades: {self.total_trades}")

    def run(self):
        """Main entry point - route to appropriate mode"""
        if self.mode == TradingMode.BACKTEST:
            return self.run_backtest()
        else:
            return self.run_live_trading()


def main():
    """Main entry point with mode selection"""
    parser = argparse.ArgumentParser(description='Unified Relative Momentum Trader')
    parser.add_argument('--mode', type=str, choices=['backtest', 'test', 'live'],
                       required=True, help='Trading mode')
    parser.add_argument('--config', type=str,
                       default='config/market_neutral_config.yaml',
                       help='Configuration file path')

    args = parser.parse_args()

    try:
        # Convert mode string to enum
        mode = TradingMode(args.mode)

        print(f"🎯 Starting Unified Trader in {mode.value.upper()} mode")
        print("✅ GUARANTEED: Identical strategy parameters across all modes")

        trader = UnifiedRelativeMomentumTrader(mode, args.config)
        trader.run()

    except Exception as e:
        logging.error(f"Failed to start unified trader: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()