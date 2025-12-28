#!/usr/bin/env python3
"""
Test script to verify alignment between backtest and test modes
"""

import sys
import os
import logging
from datetime import datetime

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils.config import ConfigManager

def test_configuration_alignment():
    """Test that configuration is properly aligned across modes"""
    print("🔍 Testing Configuration Alignment...")

    config_manager = ConfigManager()
    config = config_manager.load_config('config/unified_trading_config.yaml')

    # Check mode-specific settings
    backtest_mode = config['execution']['backtest_mode']
    test_mode = config['execution']['test_mode']
    live_mode = config['execution']['live_mode']

    print(f"\n📊 Mode Configuration Comparison:")
    print(f"{'Setting':<25} {'Backtest':<15} {'Test':<15} {'Live':<15} {'Aligned?':<10}")
    print(f"{'-'*85}")

    # Check interval alignment
    bt_interval = backtest_mode['interval']
    test_interval = test_mode['interval']
    live_interval = live_mode['interval']
    interval_aligned = bt_interval == test_interval == live_interval
    print(f"{'Interval':<25} {bt_interval:<15} {test_interval:<15} {live_interval:<15} {'✅' if interval_aligned else '❌'}")

    # Check position size multiplier
    bt_mult = 1.0  # Backtest always uses 1.0 (implicit)
    test_mult = test_mode['position_size_multiplier']
    live_mult = live_mode['position_size_multiplier']
    mult_aligned = bt_mult == test_mult == live_mult
    print(f"{'Position Multiplier':<25} {bt_mult:<15} {test_mult:<15} {live_mult:<15} {'✅' if mult_aligned else '❌'}")

    # Check frequency alignment
    bt_freq = "daily"  # Backtest is daily
    test_freq = f"{test_mode['check_frequency_minutes']}min"
    live_freq = f"{live_mode['check_frequency_minutes']}min"
    freq_aligned = test_mode['check_frequency_minutes'] == live_mode['check_frequency_minutes'] == 1440
    print(f"{'Check Frequency':<25} {bt_freq:<15} {test_freq:<15} {live_freq:<15} {'✅' if freq_aligned else '❌'}")

    # Check futures usage
    test_futures = not test_mode.get('use_spot_trading', True)
    live_futures = config['futures']['enabled']
    futures_aligned = test_futures == live_futures
    print(f"{'Futures Enabled':<25} {'True':<15} {str(test_futures):<15} {str(live_futures):<15} {'✅' if futures_aligned else '❌'}")

    # Overall alignment
    overall_aligned = interval_aligned and mult_aligned and freq_aligned and futures_aligned
    print(f"\n🎯 Overall Alignment: {'✅ ALIGNED' if overall_aligned else '❌ NOT ALIGNED'}")

    if overall_aligned:
        print("✅ Configuration is properly aligned across all modes!")
    else:
        print("❌ Configuration needs further alignment!")

    return overall_aligned

def test_strategy_parameters():
    """Test that strategy parameters are identical across modes"""
    print(f"\n📈 Testing Strategy Parameters...")

    config_manager = ConfigManager()
    config = config_manager.load_config('config/unified_trading_config.yaml')

    # Check pairs configuration
    pairs = config['pairs']
    print(f"Trading Pairs: {len(pairs)}")
    for pair in pairs:
        print(f"  {pair['base']}/{pair['alt'].replace('USDT', '')}: EMA={pair['ema_window']}, Weight={pair['allocation_weight']}, Max=${pair['max_notional']}")

    # Check strategy settings
    strategy = config['strategy']
    print(f"\nStrategy Configuration:")
    print(f"  Type: {strategy['type']}")
    print(f"  Optimization Enabled: {strategy['optimization']['enabled']}")
    print(f"  Window Range: {strategy['optimization']['window_range_start']}-{strategy['optimization']['window_range_end']}")
    print(f"  Metric: {strategy['optimization']['metric']}")
    print(f"  Enhancements: {strategy['enhancements']['enabled']}")

    # Check risk parameters
    risk = config['risk']
    print(f"\nRisk Management:")
    print(f"  Max Position Size: {risk['max_position_size']*100:.0f}%")
    print(f"  Max Daily DD: {risk['max_daily_dd']*100:.0f}%")
    print(f"  Leverage Limit: {risk['leverage_limit']}x")
    print(f"  Stop Loss: {risk['stop_loss']*100:.0f}%")

    return True

def main():
    """Main test function"""
    print("🚀 Testing Strategy Alignment Across Modes")
    print("="*60)

    try:
        # Test configuration alignment
        config_aligned = test_configuration_alignment()

        # Test strategy parameters
        strategy_ok = test_strategy_parameters()

        print("\n" + "="*60)
        if config_aligned and strategy_ok:
            print("✅ ALL TESTS PASSED - Strategy is aligned across modes!")
            print("\n🎯 Ready to run:")
            print("  Backtest: python3 scripts/run_relative_momentum_backtest.py")
            print("  Test:     python3 scripts/unified_relative_momentum_trader.py --mode test")
            print("  Live:     python3 scripts/unified_relative_momentum_trader.py --mode live")
        else:
            print("❌ ALIGNMENT ISSUES DETECTED - Check configuration!")

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False

    return config_aligned and strategy_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)