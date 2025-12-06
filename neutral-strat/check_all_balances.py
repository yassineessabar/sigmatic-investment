#!/usr/bin/env python3
"""
Comprehensive Balance Checker for Both USDT-M and COIN-M Futures
Get actual balances from Binance API for both futures markets
"""

import ccxt
import json

# ---------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------

API_KEY = "cXur5uynoGfICg5mviK7zJRpn2C4xIAgV9Ou7gY2075fc3XMMgRtPnOACU9VOJdu"
API_SECRET = "n3gu6IZrHP1WJRnYppKUqnTUjVnN8lXxmkLd8tM4eNazRIxdzv5v2R42dqSmxFP2"

# ---------------------------------------------------
# Helper printing function
# ---------------------------------------------------
def line(title):
    print("\n" + "="*60)
    print(title)
    print("="*60)

def print_balance_summary(balance_data, title):
    """Print formatted balance summary"""
    print(f"\n{title}")
    print("-" * len(title))

    if 'USDT' in balance_data:
        usdt = balance_data['USDT']
        print(f"💰 USDT Balance:")
        print(f"   Total: ${usdt.get('total', 0):,.2f}")
        print(f"   Free:  ${usdt.get('free', 0):,.2f}")
        print(f"   Used:  ${usdt.get('used', 0):,.2f}")

    # Show other significant balances
    crypto_balances = []
    for symbol, data in balance_data.items():
        if symbol not in ['info', 'free', 'used', 'total'] and isinstance(data, dict):
            total = data.get('total', 0)
            if total > 0:
                crypto_balances.append((symbol, total, data.get('free', 0), data.get('used', 0)))

    if crypto_balances:
        print(f"\n🪙 Crypto Balances:")
        crypto_balances.sort(key=lambda x: x[1], reverse=True)  # Sort by total desc
        for symbol, total, free, used in crypto_balances[:10]:  # Show top 10
            print(f"   {symbol}: {total:.6f} (Free: {free:.6f}, Used: {used:.6f})")

# ---------------------------------------------------
# MAIN BALANCE CHECKING
# ---------------------------------------------------

def main():
    line("COMPREHENSIVE BALANCE CHECKER")
    print("Checking balances across ALL Binance markets...")

    # ---------------------------------------------------
    # 1. SPOT BALANCE
    # ---------------------------------------------------
    line("1. SPOT BALANCE")
    try:
        spot_exchange = ccxt.binance({
            'apiKey': API_KEY,
            'secret': API_SECRET,
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        })
        spot_exchange.enable_demo_trading(True)

        spot_balance = spot_exchange.fetch_balance()
        print_balance_summary(spot_balance, "SPOT ACCOUNT BALANCE")

        # Get raw USDT info
        spot_usdt = spot_balance.get('USDT', {})
        print(f"\n📊 SPOT USDT Details: {spot_usdt}")

    except Exception as e:
        print(f"❌ Error getting spot balance: {e}")

    # ---------------------------------------------------
    # 2. USDT-M FUTURES BALANCE (future)
    # ---------------------------------------------------
    line("2. USDT-M FUTURES BALANCE")
    try:
        usdtm_exchange = ccxt.binance({
            'apiKey': API_KEY,
            'secret': API_SECRET,
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}  # USDT-M futures
        })
        usdtm_exchange.enable_demo_trading(True)

        usdtm_balance = usdtm_exchange.fetch_balance()
        print_balance_summary(usdtm_balance, "USDT-M FUTURES BALANCE")

        # Get raw account info
        print(f"\n📊 USDT-M Raw Info Keys: {list(usdtm_balance.get('info', {}).keys())}")

        # Try to get account info directly
        try:
            fapi_account = usdtm_exchange.fapiPrivateGetAccount()
            print(f"\n💼 USDT-M Account Info:")
            print(f"   Total Wallet Balance: {fapi_account.get('totalWalletBalance', 'N/A')}")
            print(f"   Total Cross Balance: {fapi_account.get('totalCrossWalletBalance', 'N/A')}")
            print(f"   Available Balance: {fapi_account.get('availableBalance', 'N/A')}")
        except Exception as e:
            print(f"⚠️ Could not get USDT-M account info: {e}")

    except Exception as e:
        print(f"❌ Error getting USDT-M balance: {e}")

    # ---------------------------------------------------
    # 3. COIN-M FUTURES BALANCE (delivery)
    # ---------------------------------------------------
    line("3. COIN-M FUTURES BALANCE")
    try:
        coinm_exchange = ccxt.binance({
            'apiKey': API_KEY,
            'secret': API_SECRET,
            'enableRateLimit': True,
            'options': {'defaultType': 'delivery'}  # COIN-M futures
        })
        coinm_exchange.enable_demo_trading(True)

        coinm_balance = coinm_exchange.fetch_balance()
        print_balance_summary(coinm_balance, "COIN-M FUTURES BALANCE")

        # Get raw account info
        print(f"\n📊 COIN-M Raw Info Keys: {list(coinm_balance.get('info', {}).keys())}")

        # Try to get account info directly
        try:
            dapi_account = coinm_exchange.dapiPrivateGetAccount()
            print(f"\n💼 COIN-M Account Info:")

            # Extract key balance information
            assets = dapi_account.get('assets', [])
            total_value = 0
            available_value = 0

            MAIN_TOKENS = ["BTC", "ETH", "SOL", "ADA", "AVAX", "BNB", "DOT", "LINK"]

            print(f"\n🏦 COIN-M Asset Balances:")
            for asset in assets:
                symbol = asset.get('asset', '')
                wallet_balance = float(asset.get('walletBalance', 0))
                margin_balance = float(asset.get('marginBalance', 0))
                available_balance = float(asset.get('availableBalance', 0))

                if wallet_balance > 0 or margin_balance > 0:
                    print(f"   {symbol}:")
                    print(f"     Wallet: {wallet_balance:.6f}")
                    print(f"     Margin: {margin_balance:.6f}")
                    print(f"     Available: {available_balance:.6f}")

                    if symbol in MAIN_TOKENS:
                        try:
                            # Get price to calculate USDT value
                            ticker = coinm_exchange.fetch_ticker(f'{symbol}/USDT')
                            price = ticker['last']
                            value_usdt = margin_balance * price
                            total_value += value_usdt
                            available_value += available_balance * price
                            print(f"     USDT Value: ${value_usdt:.2f}")
                        except:
                            pass

            print(f"\n💰 COIN-M Total Value Summary:")
            print(f"   Estimated Total Value: ${total_value:.2f}")
            print(f"   Estimated Available: ${available_value:.2f}")

        except Exception as e:
            print(f"⚠️ Could not get COIN-M account info: {e}")

    except Exception as e:
        print(f"❌ Error getting COIN-M balance: {e}")

    # ---------------------------------------------------
    # 4. MARGIN BALANCE
    # ---------------------------------------------------
    line("4. MARGIN BALANCE")
    try:
        margin_exchange = ccxt.binance({
            'apiKey': API_KEY,
            'secret': API_SECRET,
            'enableRateLimit': True,
            'options': {'defaultType': 'margin'}
        })
        margin_exchange.enable_demo_trading(True)

        margin_balance = margin_exchange.fetch_balance()
        print_balance_summary(margin_balance, "MARGIN ACCOUNT BALANCE")

    except Exception as e:
        print(f"❌ Error getting margin balance: {e}")

    # ---------------------------------------------------
    # 5. SUMMARY AND RECOMMENDATIONS
    # ---------------------------------------------------
    line("5. SUMMARY & RECOMMENDATIONS")

    print("📋 Balance Summary:")
    print("   SPOT: Check USDT balance above")
    print("   USDT-M: Check totalWalletBalance above")
    print("   COIN-M: Check estimated total value above")
    print("   MARGIN: Check margin balance above")

    print("\n💡 For Trading System:")
    print("   1. SPOT USDT: Use for general trading capital")
    print("   2. USDT-M: Use totalWalletBalance for USDT futures")
    print("   3. COIN-M: Use marginBalance * prices for crypto futures")
    print("   4. Choose the account with sufficient balance for your strategy")

    line("BALANCE CHECK COMPLETE")

if __name__ == "__main__":
    main()