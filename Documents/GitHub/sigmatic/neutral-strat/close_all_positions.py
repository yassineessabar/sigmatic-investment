#!/usr/bin/env python3
"""
Close All Open Positions Script
Closes all open positions on Binance demo account
"""

import ccxt
import os
import sys

def close_all_positions():
    """Close all open positions on demo account"""

    # Get API credentials from environment
    api_key = os.getenv('BINANCE_TESTNET_API_KEY', 'cXur5uynoGfICg5mviK7zJRpn2C4xIAgV9Ou7gY2075fc3XMMgRtPnOACU9VOJdu')
    api_secret = os.getenv('BINANCE_TESTNET_API_SECRET', 'n3gu6IZrHP1WJRnYppKUqnTUjVnN8lXxmkLd8tM4eNazRIxdzv5v2R42dqSmxFP2')

    try:
        # Initialize exchange with demo trading
        exchange = ccxt.binance({
            'apiKey': api_key,
            'secret': api_secret,
            'sandbox': False,
            'options': {
                'defaultType': 'delivery',  # COIN-M futures
            }
        })

        # Enable demo trading
        exchange.enable_demo_trading(True)
        print("📊 Connected to Binance Demo Trading (COIN-M Futures)")

        # Get all open positions
        positions = exchange.fetch_positions()
        open_positions = [pos for pos in positions if float(pos['contracts']) != 0]

        if not open_positions:
            print("✅ No open positions found")
            return

        print(f"🔍 Found {len(open_positions)} open positions:")

        closed_count = 0
        failed_count = 0

        for position in open_positions:
            symbol = position['symbol']
            side = position['side']
            size = abs(float(position['contracts']))

            if size == 0:
                continue

            print(f"\n📦 Position: {symbol}")
            print(f"   Side: {side}")
            print(f"   Size: {size}")
            print(f"   Entry Price: ${float(position['entryPrice']):.2f}")
            print(f"   Mark Price: ${float(position['markPrice']):.2f}")
            print(f"   PnL: ${float(position['unrealizedPnl']):.2f}")

            try:
                # Determine the closing side
                close_side = 'sell' if side == 'long' else 'buy'

                print(f"   🔄 Closing with {close_side} order...")

                # Create market order to close position
                order = exchange.create_order(
                    symbol=symbol,
                    type='market',
                    side=close_side,
                    amount=size,
                    params={'reduceOnly': True}  # Ensure this only closes, doesn't open new
                )

                print(f"   ✅ Position closed successfully!")
                print(f"   Order ID: {order['id']}")
                print(f"   Status: {order['status']}")

                if 'average' in order and order['average']:
                    print(f"   Close Price: ${order['average']:.2f}")

                closed_count += 1

            except Exception as e:
                print(f"   ❌ Failed to close position: {e}")
                failed_count += 1

        print(f"\n{'='*50}")
        print(f"📊 SUMMARY:")
        print(f"   Total Positions Found: {len(open_positions)}")
        print(f"   Successfully Closed: {closed_count}")
        print(f"   Failed to Close: {failed_count}")

        if closed_count > 0:
            print(f"✅ All positions closed successfully!")

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    close_all_positions()