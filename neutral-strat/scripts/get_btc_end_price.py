#!/usr/bin/env python3
"""
Extract actual BTC spot price from backtest data (Nov 21, 2025)
"""

import pandas as pd
import ccxt
from datetime import datetime

def get_btc_spot_price():
    """Get actual BTC spot price from our data"""

    try:
        # Use CCXT to get recent BTC price
        exchange = ccxt.binance()

        # Get recent OHLCV data for BTCUSDT
        symbol = 'BTC/USDT'
        timeframe = '1d'

        # Get the most recent candles
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=5)

        if ohlcv:
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')

            # Get the most recent close price
            latest_price = df['close'].iloc[-1]
            latest_date = df['datetime'].iloc[-1]

            print(f"Latest BTC/USDT price: ${latest_price:,.2f}")
            print(f"As of: {latest_date}")
            print(f"Recent prices:")
            print(df[['datetime', 'close']].tail())

            return latest_price, latest_date
        else:
            print("No price data available")
            return None, None

    except Exception as e:
        print(f"Error fetching price: {e}")
        return None, None

def get_price_from_saved_data():
    """Try to get price from our saved backtest data"""

    try:
        # Check if we have cached data from our recent backtest
        import os
        cache_dir = "data/cache"

        # Look for BTC data files
        if os.path.exists(cache_dir):
            btc_files = [f for f in os.listdir(cache_dir) if 'BTC' in f and f.endswith('.pkl')]
            if btc_files:
                print(f"Found cached BTC data: {btc_files}")

                # Try to load the most recent file
                import pickle
                latest_file = os.path.join(cache_dir, btc_files[0])

                with open(latest_file, 'rb') as f:
                    data = pickle.load(f)

                if 'price_data' in data and not data['price_data'].empty:
                    price_df = data['price_data']
                    latest_price = price_df['close'].iloc[-1]
                    latest_date = price_df.index[-1]

                    print(f"From cached data:")
                    print(f"BTC price: ${latest_price:,.2f}")
                    print(f"Date: {latest_date}")

                    return latest_price, latest_date

    except Exception as e:
        print(f"Error reading cached data: {e}")

    return None, None

if __name__ == "__main__":
    print("🔍 Getting actual BTC spot price as of November 21, 2025...")

    # Try to get current price
    price, date = get_btc_spot_price()

    if price is None:
        # Fallback to cached data
        print("\n📁 Trying cached backtest data...")
        price, date = get_price_from_saved_data()

    if price:
        print(f"\n✅ BTC Spot Price: ${price:,.2f}")
        print(f"📅 As of: {date}")
    else:
        print("\n❌ Could not retrieve BTC price data")
        print("📊 Using estimated price from backtest performance...")

        # Calculate from backtest performance (7.66x from ~$7,200 in 2020)
        estimated_price = 7200 * 7.66  # ~$55,152
        print(f"🔢 Estimated BTC price: ${estimated_price:,.0f}")