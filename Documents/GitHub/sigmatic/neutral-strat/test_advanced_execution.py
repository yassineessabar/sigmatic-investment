#!/usr/bin/env python3

"""
Test Advanced Execution Engine with Different Algorithms
"""

import sys
import os
import time
import ccxt
from datetime import datetime

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import ConfigManager
from src.execution.advanced_execution import AdvancedExecutionEngine, ExecutionAlgorithm

def test_advanced_execution():
    """Test different execution algorithms"""
    print("🚀 Testing Advanced Execution Engine")
    print("="*50)

    # Load config
    config_manager = ConfigManager()
    config = config_manager.load_config('config/market_neutral_config.yaml')

    # Get execution config
    execution_config = config['execution']['advanced_execution']
    print(f"📋 Execution Config:")
    print(f"  Algorithm: {execution_config['execution_algorithm']}")
    print(f"  Max Slippage: {execution_config['max_position_slippage']*10000:.0f}bps")
    print(f"  TWAP Duration: {execution_config['twap_duration']}s")

    # Initialize exchange
    binance_config = config['binance']['testnet_mode']
    exchange = ccxt.binance({
        'apiKey': binance_config['api_key'],
        'secret': binance_config['api_secret'],
        'enableRateLimit': True,
        'options': {'defaultType': 'delivery'}
    })
    exchange.enable_demo_trading(True)
    print("✅ Demo trading enabled")

    # Initialize execution engine
    execution_engine = AdvancedExecutionEngine(exchange, execution_config)
    print("✅ Advanced execution engine initialized")

    # Test different algorithms
    algorithms_to_test = [
        ExecutionAlgorithm.SMART_LIMIT,
        ExecutionAlgorithm.ADAPTIVE,
        # ExecutionAlgorithm.TWAP,  # Skip TWAP for quick test
        # ExecutionAlgorithm.MARKET  # For comparison
    ]

    symbol = 'BTC/USDT:USDT'
    # Calculate minimum size based on $100 notional requirement
    ticker = exchange.fetch_ticker(symbol)
    btc_price = ticker['last']
    min_notional = 100  # $100 minimum
    test_size = (min_notional / btc_price) * 1.1  # 10% buffer
    print(f"📏 BTC Price: ${btc_price:.2f}, Min Size: {test_size:.6f} BTC (~${test_size * btc_price:.2f})")

    for algorithm in algorithms_to_test:
        print(f"\n🧪 Testing {algorithm.value.upper()} Algorithm")
        print("-" * 40)

        try:
            # Create test signal
            test_signal = {
                'symbol': symbol,
                'side': 'buy',
                'size': test_size,
                'urgency': 'normal'
            }

            # Execute with specific algorithm
            result = execution_engine.execute_trade(test_signal, algorithm)

            print(f"📊 Execution Result:")
            print(f"  Status: {result.get('status', 'unknown')}")
            print(f"  Algorithm: {result.get('algorithm', 'unknown')}")

            if 'executed_price' in result:
                print(f"  Executed Price: ${result['executed_price']:.2f}")

            if 'execution_time' in result:
                print(f"  Execution Time: {result['execution_time']:.2f}s")

            if 'slippage' in result:
                slippage_bps = result['slippage'] * 10000
                print(f"  Slippage: {slippage_bps:.1f}bps")

            if 'price_improvement' in result:
                improvement_bps = result['price_improvement'] * 10000
                print(f"  Price Improvement: {improvement_bps:.1f}bps")

            if result.get('status') == 'closed':
                print(f"  ✅ Execution successful")

                # Close the position immediately
                print(f"  🔄 Closing position...")
                close_signal = {
                    'symbol': symbol,
                    'side': 'sell',
                    'size': test_size,
                    'urgency': 'high'
                }

                close_result = execution_engine.execute_trade(close_signal, ExecutionAlgorithm.SMART_LIMIT)
                if close_result.get('status') == 'closed':
                    print(f"  ✅ Position closed successfully")
                else:
                    print(f"  ⚠️ Position close failed")

            else:
                print(f"  ❌ Execution failed or partial")

        except Exception as e:
            print(f"  ❌ Error testing {algorithm.value}: {e}")

        # Wait between tests
        if algorithm != algorithms_to_test[-1]:
            print(f"  ⏰ Waiting 10 seconds before next test...")
            time.sleep(10)

    # Show execution summary
    print(f"\n📈 Execution Performance Summary")
    print("="*50)
    summary = execution_engine.get_execution_summary()
    metrics = summary['metrics']

    print(f"Total Orders: {execution_engine.metrics.total_orders}")
    print(f"Success Rate: {metrics['success_rate']*100:.1f}%")
    print(f"Average Slippage: {metrics['avg_slippage']*10000:.1f}bps")
    print(f"Average Execution Time: {metrics['avg_execution_time']:.2f}s")

    if metrics['avg_price_improvement'] != 0:
        print(f"Average Price Improvement: {metrics['avg_price_improvement']*10000:.1f}bps")

    print(f"\n🎯 Advanced Execution Test Complete!")

def test_market_analysis():
    """Test market condition analysis"""
    print(f"\n🔍 Testing Market Analysis")
    print("-" * 30)

    # Load config and initialize
    config_manager = ConfigManager()
    config = config_manager.load_config('config/market_neutral_config.yaml')
    execution_config = config['execution']['advanced_execution']

    binance_config = config['binance']['testnet_mode']
    exchange = ccxt.binance({
        'apiKey': binance_config['api_key'],
        'secret': binance_config['api_secret'],
        'enableRateLimit': True,
        'options': {'defaultType': 'delivery'}
    })
    exchange.enable_demo_trading(True)

    execution_engine = AdvancedExecutionEngine(exchange, execution_config)

    # Analyze market conditions
    symbol = 'BTC/USDT:USDT'
    market_analysis = execution_engine._analyze_market_conditions(symbol)

    print(f"📊 Market Analysis for {symbol}:")
    print(f"  Spread: {market_analysis['spread_bps']:.1f}bps")
    print(f"  Bid Liquidity: {market_analysis['bid_liquidity']:.3f}")
    print(f"  Ask Liquidity: {market_analysis['ask_liquidity']:.3f}")
    print(f"  Total Liquidity: {market_analysis['total_liquidity']:.3f}")
    print(f"  Volatility: {market_analysis['volatility']*100:.2f}%")
    print(f"  Recent Volume: {market_analysis['recent_volume']:.3f}")
    print(f"  Best Bid: ${market_analysis['best_bid']:.2f}")
    print(f"  Best Ask: ${market_analysis['best_ask']:.2f}")

    # Test adaptive algorithm choice
    test_signal = {
        'symbol': symbol,
        'side': 'buy',
        'size': 0.001,
        'urgency': 'normal'
    }

    chosen_algorithm = execution_engine._choose_adaptive_algorithm(test_signal, market_analysis)
    print(f"  🎯 Adaptive Algorithm Choice: {chosen_algorithm.value}")

if __name__ == "__main__":
    try:
        test_market_analysis()
        test_advanced_execution()
    except KeyboardInterrupt:
        print("\n⏹️ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()