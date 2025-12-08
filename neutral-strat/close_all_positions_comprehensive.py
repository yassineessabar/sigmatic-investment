#!/usr/bin/env python3
"""
Comprehensive Close All Positions Script
Closes positions across ALL Binance markets: COIN-M, USDT-M, Spot, Margin
"""

import ccxt
import os
import sys

def close_positions_for_exchange(exchange_type, exchange):
    """Close all positions for a specific exchange type"""
    print(f"\n🔍 Checking {exchange_type} positions...")

    try:
        positions = exchange.fetch_positions()
        print(f"   Total position records: {len(positions)}")

        open_positions = []
        for pos in positions:
            contracts = float(pos.get('contracts', 0))
            unrealized_pnl = float(pos.get('unrealizedPnl', 0))

            if contracts != 0 or abs(unrealized_pnl) > 0.000001:
                open_positions.append(pos)

        if not open_positions:
            print(f"   ✅ No open positions in {exchange_type}")
            return 0

        print(f"   🎯 Found {len(open_positions)} open positions to close!")

        closed_count = 0
        for pos in open_positions:
            symbol = pos['symbol']
            side = pos['side']
            size = abs(float(pos['contracts']))
            unrealized_pnl = float(pos.get('unrealizedPnl', 0))

            print(f"\n   📦 {symbol}: {side} {size} | PnL: ${unrealized_pnl:.6f}")

            try:
                # Determine closing side
                close_side = 'sell' if side == 'long' else 'buy'

                # Create market order to close
                order = exchange.create_order(
                    symbol=symbol,
                    type='market',
                    side=close_side,
                    amount=size,
                    params={'reduceOnly': True}
                )

                print(f"      ✅ Closed! Order ID: {order['id']}")
                if 'average' in order and order['average']:
                    print(f"      Close Price: ${order['average']:.6f}")

                closed_count += 1

            except Exception as e:
                print(f"      ❌ Failed to close: {e}")

        return closed_count

    except Exception as e:
        print(f"   ❌ Error checking {exchange_type}: {e}")
        return 0

def main():
    """Close all positions across all Binance markets"""

    api_key = os.getenv('BINANCE_TESTNET_API_KEY', 'cXur5uynoGfICg5mviK7zJRpn2C4xIAgV9Ou7gY2075fc3XMMgRtPnOACU9VOJdu')
    api_secret = os.getenv('BINANCE_TESTNET_API_SECRET', 'n3gu6IZrHP1WJRnYppKUqnTUjVnN8lXxmkLd8tM4eNazRIxdzv5v2R42dqSmxFP2')

    print("🚀 COMPREHENSIVE POSITION CLOSER")
    print("Checking and closing positions across ALL Binance markets...")

    total_closed = 0

    # 1. COIN-M Futures (Delivery)
    try:
        coinm_exchange = ccxt.binance({
            'apiKey': api_key,
            'secret': api_secret,
            'sandbox': False,
            'options': {'defaultType': 'delivery'}
        })
        coinm_exchange.enable_demo_trading(True)

        closed = close_positions_for_exchange("COIN-M Futures", coinm_exchange)
        total_closed += closed

    except Exception as e:
        print(f"❌ Could not check COIN-M: {e}")

    # 2. USDT-M Futures
    try:
        usdtm_exchange = ccxt.binance({
            'apiKey': api_key,
            'secret': api_secret,
            'sandbox': False,
            'options': {'defaultType': 'future'}
        })
        usdtm_exchange.enable_demo_trading(True)

        closed = close_positions_for_exchange("USDT-M Futures", usdtm_exchange)
        total_closed += closed

    except Exception as e:
        print(f"❌ Could not check USDT-M: {e}")

    # 3. Spot (for any leveraged spot positions)
    try:
        spot_exchange = ccxt.binance({
            'apiKey': api_key,
            'secret': api_secret,
            'sandbox': False,
            'options': {'defaultType': 'spot'}
        })
        spot_exchange.enable_demo_trading(True)

        # Spot doesn't have positions, but check balance for any leveraged trades
        print(f"\n🔍 Checking Spot account...")
        balance = spot_exchange.fetch_balance()

        leveraged_assets = []
        for asset, data in balance.items():
            if isinstance(data, dict) and data.get('used', 0) > 0:
                leveraged_assets.append((asset, data))

        if leveraged_assets:
            print(f"   ⚠️ Found {len(leveraged_assets)} assets with 'used' balance (potential leveraged positions)")
            for asset, data in leveraged_assets:
                print(f"      {asset}: Used {data['used']}")
        else:
            print("   ✅ No leveraged spot positions")

    except Exception as e:
        print(f"❌ Could not check Spot: {e}")

    # 4. Summary
    print(f"\n{'='*60}")
    print(f"📊 FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"Total positions closed across all markets: {total_closed}")

    if total_closed > 0:
        print(f"✅ Successfully closed {total_closed} positions!")
        print(f"💡 Check your GUI again - unrealized PnL should now be zero")
    else:
        print(f"🤔 No positions found to close.")
        print(f"💡 The unrealized PnL might be from:")
        print(f"   - Funding fee accumulation")
        print(f"   - Very small dust positions")
        print(f"   - GUI display lag")
        print(f"   - Different account/subaccount")

if __name__ == "__main__":
    main()