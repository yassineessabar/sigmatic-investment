#!/usr/bin/env python3

"""
Quick Test Script to Verify Position Entry/Exit in Demo Mode
"""

import sys
import os
import time
import ccxt
from datetime import datetime

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import ConfigManager

def test_demo_execution():
    """Test demo execution with a small BTC position"""
    print("🧪 Testing Demo Execution - Entry and Exit")
    print("="*50)

    # Load config
    config_manager = ConfigManager()
    config = config_manager.load_config('config/market_neutral_config.yaml')

    # Get test mode credentials
    binance_config = config['binance']['testnet_mode']
    api_key = binance_config['api_key']
    api_secret = binance_config['api_secret']

    # Initialize exchange
    exchange = ccxt.binance({
        'apiKey': api_key,
        'secret': api_secret,
        'enableRateLimit': True,
        'options': {
            'defaultType': 'delivery'  # COIN-M futures
        }
    })

    # Enable demo trading
    exchange.enable_demo_trading(True)
    print("✅ Demo trading enabled")

    # Check balance before
    balance_before = exchange.fetch_balance()
    print(f"\n📊 Balance Before:")
    print(f"  BTC: {balance_before.get('BTC', {}).get('free', 0):.6f}")

    # Check current price - auto-detect the correct symbol
    symbol = None
    current_price = None

    # Get current positions - check all positions for COIN-M
    positions_before = exchange.fetch_positions()
    existing_position = None
    print(f"\n📊 Checking existing positions...")

    for pos in positions_before:
        symbol_check = pos.get('symbol', '')
        contracts = float(pos.get('contracts', pos.get('size', 0)))

        if ('BTC' in symbol_check and abs(contracts) > 0):
            existing_position = pos
            symbol = symbol_check  # Use the actual symbol from the position
            print(f"  Found position in {symbol_check}: {contracts} contracts")

            # Get current price for this symbol
            ticker = exchange.fetch_ticker(symbol)
            current_price = ticker['last']
            print(f"📈 Current {symbol} Price: ${current_price:,.2f}")
            break

    if not existing_position:
        print(f"  No BTC positions found")
        # Default to USDT-M if no position found
        symbol = 'BTC/USDT:USDT'
        ticker = exchange.fetch_ticker(symbol)
        current_price = ticker['last']
        print(f"📈 Current {symbol} Price: ${current_price:,.2f}")

    if existing_position:
        print(f"📍 Existing Position Found:")
        contracts = float(existing_position.get('contracts', 0))
        entry_price = float(existing_position.get('entryPrice', 0))
        mark_price = float(existing_position.get('markPrice', 0))
        unrealized_pnl = float(existing_position.get('unrealizedPnl', 0))

        print(f"  Size: {abs(contracts):.6f} BTC")
        print(f"  Side: {'Long' if contracts > 0 else 'Short'}")
        print(f"  Entry Price: ${entry_price:,.2f}")
        print(f"  Mark Price: ${mark_price:,.2f}")
        print(f"  Unrealized P&L: ${unrealized_pnl:.2f}")

        # Test closing existing position
        print(f"\n🔄 Testing Position Exit...")
        try:
            if contracts > 0:
                # Close long position with sell
                order = exchange.create_market_sell_order(symbol, abs(contracts))
            else:
                # Close short position with buy
                order = exchange.create_market_buy_order(symbol, abs(contracts))

            print(f"✅ Exit Order Placed: {order['id']}")
            print(f"   Type: {order['type']}")
            print(f"   Side: {order['side']}")
            print(f"   Amount: {order['amount']}")

            time.sleep(2)  # Wait for execution

            # Check new balance and positions
            balance_after = exchange.fetch_balance()
            positions_after = exchange.fetch_positions([symbol])

            print(f"\n📊 After Exit:")
            print(f"  BTC Balance: {balance_after.get('BTC', {}).get('free', 0):.6f}")

            active_positions = [p for p in positions_after if abs(float(p.get('contracts', 0))) > 0]
            if active_positions:
                print(f"  Remaining Positions: {len(active_positions)}")
            else:
                print(f"  ✅ All positions closed")

        except Exception as e:
            print(f"❌ Error closing position: {e}")

    else:
        # Test opening a small position
        print(f"📍 No existing position found. Testing position entry...")

        # Calculate minimum position size (1 contract for COIN-M futures)
        position_size = 1

        try:
            # Place a small long order
            print(f"\n🔄 Testing Position Entry...")
            print(f"   Opening LONG {position_size} contract(s) at market price")

            order = exchange.create_market_buy_order(symbol, position_size)

            print(f"✅ Entry Order Placed: {order['id']}")
            print(f"   Type: {order['type']}")
            print(f"   Side: {order['side']}")
            print(f"   Amount: {order['amount']}")

            time.sleep(2)  # Wait for execution

            # Check new positions
            positions_after = exchange.fetch_positions([symbol])
            for pos in positions_after:
                if abs(float(pos.get('contracts', 0))) > 0:
                    contracts = float(pos.get('contracts', 0))
                    entry_price = float(pos.get('entryPrice', 0))
                    mark_price = float(pos.get('markPrice', 0))
                    unrealized_pnl = float(pos.get('unrealizedPnl', 0))

                    print(f"\n📍 New Position Created:")
                    print(f"   Size: {abs(contracts):.6f} BTC")
                    print(f"   Side: {'Long' if contracts > 0 else 'Short'}")
                    print(f"   Entry Price: ${entry_price:,.2f}")
                    print(f"   Mark Price: ${mark_price:,.2f}")
                    print(f"   Unrealized P&L: ${unrealized_pnl:.2f}")

                    # Immediately close the position
                    print(f"\n🔄 Testing Immediate Position Exit...")
                    close_order = exchange.create_market_sell_order(symbol, abs(contracts))

                    print(f"✅ Close Order Placed: {close_order['id']}")

                    time.sleep(2)

                    # Final check
                    final_positions = exchange.fetch_positions([symbol])
                    final_active = [p for p in final_positions if abs(float(p.get('contracts', 0))) > 0]

                    if not final_active:
                        print(f"✅ Position successfully closed!")
                    else:
                        print(f"⚠️  Position still open: {len(final_active)} remaining")

                    break

        except Exception as e:
            print(f"❌ Error with position entry: {e}")

    print(f"\n🎯 Demo Execution Test Complete!")
    print(f"="*50)

if __name__ == "__main__":
    test_demo_execution()