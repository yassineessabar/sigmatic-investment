#!/usr/bin/env python3
"""
Test the complete live trading system
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_strategy_components():
    """Test all strategy components work together"""

    print("🧪 Testing Strategy Components...")

    # Test 1: Config Manager
    try:
        from src.utils.config import ConfigManager
        config_manager = ConfigManager()
        config = config_manager.load_config('config/two_week_config.yaml')
        print("✅ Config Manager: OK")
        print(f"   Loaded {len(config.get('pairs', []))} trading pairs")
    except Exception as e:
        print(f"❌ Config Manager: FAILED - {e}")
        return False

    # Test 2: Strategy
    try:
        from src.strategies.neutral_pairs import compute_signals

        # Create mock market data
        dates = pd.date_range(start='2023-01-01', end='2023-01-10', freq='1H')

        # Generate mock price data
        btc_prices = 30000 + np.cumsum(np.random.randn(len(dates)) * 100)
        eth_prices = 2000 + np.cumsum(np.random.randn(len(dates)) * 50)

        market_data = {
            'BTCUSDT': pd.DataFrame({
                'close': btc_prices,
                'open': btc_prices * 0.999,
                'high': btc_prices * 1.001,
                'low': btc_prices * 0.998,
                'volume': np.random.randint(1000, 10000, len(dates))
            }, index=dates),
            'ETHUSDT': pd.DataFrame({
                'close': eth_prices,
                'open': eth_prices * 0.999,
                'high': eth_prices * 1.001,
                'low': eth_prices * 0.998,
                'volume': np.random.randint(1000, 10000, len(dates))
            }, index=dates)
        }

        # Generate signals
        signals = compute_signals(market_data, config)
        print("✅ Strategy: OK")
        print(f"   Generated {len(signals)} trading signals")

    except Exception as e:
        print(f"❌ Strategy: FAILED - {e}")
        return False

    # Test 3: Portfolio
    try:
        from src.backtest.portfolio import PortfolioState

        portfolio = PortfolioState(initial_capital=100000)

        # Test portfolio with mock data
        portfolio.update_positions(market_data, datetime.now())
        summary = portfolio.get_results()

        print("✅ Portfolio: OK")
        print(f"   Initial capital: ${portfolio.initial_capital:,.2f}")
        print(f"   Current equity: ${portfolio.total_equity:,.2f}")

    except Exception as e:
        print(f"❌ Portfolio: FAILED - {e}")
        return False

    # Test 4: Data Client (mock test)
    try:
        from data.binance_client import BinanceDataClient

        # Test without actual API calls
        print("✅ Data Client: OK (structure)")
        print("   Binance client class loaded successfully")

    except Exception as e:
        print(f"❌ Data Client: FAILED - {e}")
        return False

    print("\n🎉 All Strategy Components Working!")
    return True

def test_live_trading_simulation():
    """Simulate a few live trading cycles"""

    print("\n🚀 Testing Live Trading Simulation...")

    try:
        from src.utils.config import ConfigManager
        from src.strategies.neutral_pairs import compute_signals
        from src.backtest.portfolio import PortfolioState

        # Load config
        config_manager = ConfigManager()
        config = config_manager.load_config('config/two_week_config.yaml')

        # Initialize portfolio
        portfolio = PortfolioState(initial_capital=100000)

        print(f"📊 Starting simulation with ${portfolio.initial_capital:,.2f}")

        # Simulate 5 trading cycles
        for cycle in range(5):
            print(f"\n--- Cycle {cycle + 1} ---")

            # Generate mock market data for this cycle
            current_time = datetime.now() + timedelta(hours=cycle)
            dates = pd.date_range(start=current_time - timedelta(hours=100),
                                end=current_time, freq='1H')

            # More realistic price simulation
            np.random.seed(cycle)  # For reproducible results
            btc_base = 30000
            eth_base = 2000

            btc_returns = np.random.normal(0.0001, 0.02, len(dates))  # Small drift, 2% volatility
            eth_returns = np.random.normal(0.0001, 0.025, len(dates))  # Slightly higher vol

            btc_prices = btc_base * np.cumprod(1 + btc_returns)
            eth_prices = eth_base * np.cumprod(1 + eth_returns)

            market_data = {
                'BTCUSDT': pd.DataFrame({
                    'close': btc_prices,
                    'open': btc_prices * 0.9995,
                    'high': btc_prices * 1.005,
                    'low': btc_prices * 0.995,
                    'volume': np.random.randint(1000, 10000, len(dates))
                }, index=dates),
                'ETHUSDT': pd.DataFrame({
                    'close': eth_prices,
                    'open': eth_prices * 0.9995,
                    'high': eth_prices * 1.005,
                    'low': eth_prices * 0.995,
                    'volume': np.random.randint(1000, 10000, len(dates))
                }, index=dates)
            }

            # Generate signals
            signals = compute_signals(market_data, config)

            print(f"   📈 BTC: ${btc_prices[-1]:,.2f} | ETH: ${eth_prices[-1]:,.2f}")
            print(f"   🎯 Generated {len(signals)} signals")

            # Apply signals to portfolio
            if signals:
                trades = portfolio.compute_trades(signals, config)
                portfolio.apply_trades(trades, market_data, config)
                print(f"   💼 Executed {len(trades)} trades")

            # Update portfolio with current prices
            portfolio.update_positions(market_data, current_time)

            # Show current status
            print(f"   💰 Portfolio Value: ${portfolio.total_equity:,.2f}")
            print(f"   📊 P&L: ${portfolio.total_equity - portfolio.initial_capital:,.2f}")
            print(f"   🔢 Positions: {len(portfolio.positions)}")

        # Final results
        final_results = portfolio.get_results()

        print(f"\n🏁 Final Results After 5 Cycles:")
        print(f"   💰 Final Portfolio Value: ${portfolio.total_equity:,.2f}")
        print(f"   📈 Total Return: {final_results.total_return:.2%}")
        print(f"   🎯 Total Trades: {final_results.total_trades}")
        print(f"   ✅ Win Rate: {final_results.win_rate:.1%}")
        print(f"   📊 Sharpe Ratio: {final_results.sharpe_ratio:.2f}")
        print(f"   📉 Max Drawdown: {final_results.max_drawdown:.2%}")

        print("\n✅ Live Trading Simulation Completed Successfully!")
        return True

    except Exception as e:
        print(f"❌ Live Trading Simulation: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("=" * 60)
    print("🧪 SIGMATIC NEUTRAL STRATEGY - LIVE TRADING TEST")
    print("=" * 60)

    # Test 1: Component tests
    if not test_strategy_components():
        print("❌ Component tests failed!")
        return False

    # Test 2: Live trading simulation
    if not test_live_trading_simulation():
        print("❌ Live trading simulation failed!")
        return False

    print("\n" + "=" * 60)
    print("🎉 ALL TESTS PASSED - LIVE TRADING SYSTEM READY!")
    print("=" * 60)
    print("\n📋 Next Steps:")
    print("1. Run: ./scripts/start_24_7.sh start")
    print("2. Monitor: ./scripts/start_24_7.sh status")
    print("3. View logs: ./scripts/start_24_7.sh follow")
    print("4. Check health: ./scripts/start_24_7.sh health")

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)