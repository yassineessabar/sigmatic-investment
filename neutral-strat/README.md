# Unified Relative Momentum Trading System

A unified cryptocurrency trading system that ensures **identical strategy parameters** across backtesting, demo testing, and live trading modes.

## Core Features

- **Perfect Strategy Alignment**: Same relative momentum logic across all modes
- **Futures Trading**: Full futures support with funding cost integration
- **Three Trading Modes**: Backtest, Demo Test, and Live trading
- **Risk Management**: Built-in position sizing and risk controls
- **24/7 Operation**: Automated signal generation and execution

## Installation and Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Setup API Credentials
Create a `.env` file:
```bash
# Demo Trading Credentials (for testing)
BINANCE_TESTNET_API_KEY=your_demo_api_key
BINANCE_TESTNET_API_SECRET=your_demo_secret

# Live Trading Credentials
BINANCE_LIVE_API_KEY=your_live_api_key
BINANCE_LIVE_API_SECRET=your_live_secret
```

### 3. Run the System

**Backtest (Historical Analysis):**
```bash
python3 scripts/run_relative_momentum_backtest.py --config config/unified_trading_config.yaml --start-date 2023-10-01 --end-date 2024-11-29
```

**Demo Trading (Test with Fake Money):**
```bash
python3 scripts/unified_relative_momentum_trader.py --mode test --config config/unified_trading_config.yaml
```

**Live Trading (Real Money):**
```bash
python3 scripts/unified_relative_momentum_trader.py --mode live --config config/unified_trading_config.yaml
```

## Strategy Overview

The system implements a **relative momentum strategy** across cryptocurrency futures pairs:
- **Pairs**: BTC/AVAX, BTC/ETH, BTC/SOL, BTC/ADA
- **Signal Generation**: EMA-based relative momentum detection
- **Risk Management**: 20% max position size, automatic stop-loss
- **Optimization**: Dynamic EMA window optimization (1-30 days)

## Strategy Alignment

All three modes use identical:
- Strategy parameters and logic
- Risk management rules
- Position sizing calculations
- Signal generation algorithms
- Futures trading mechanics

Only the execution environment changes (historical data vs demo vs live).

## Project Structure

```
neutral-strat/
├── scripts/
│   ├── run_relative_momentum_backtest.py    # Backtest runner
│   ├── unified_relative_momentum_trader.py  # Unified trader
│   └── start_24_7.sh                       # 24/7 startup script
├── config/
│   └── unified_trading_config.yaml         # Unified configuration
├── src/
│   ├── strategies/                          # Strategy implementations
│   ├── utils/                              # Utilities and helpers
│   └── execution/                          # Trading execution
├── data/                                   # Historical data storage
├── logs/                                   # Trading logs
├── results/                               # Backtest results
└── .env                                   # API credentials
```

## Configuration

The `config/unified_trading_config.yaml` file controls all settings:

- **Pairs**: Trading pair definitions and parameters
- **Strategy**: Relative momentum strategy configuration
- **Risk**: Risk management and position sizing
- **Execution**: Mode-specific execution settings
- **Binance**: API endpoints and credentials

## Security

- Demo trading uses Binance's demo environment (fake money)
- Live trading requires explicit confirmation
- API credentials stored in `.env` file (not committed)
- Built-in safety checks and position limits

## Performance Tracking

The system provides:
- Real-time P&L tracking
- Win rate and success metrics
- Sharpe ratio and risk-adjusted returns
- Maximum drawdown monitoring
- Detailed execution logs

## Technical Specifications

- **Language**: Python 3.8+
- **Key Libraries**: ccxt, pandas, numpy, matplotlib
- **Exchange**: Binance (spot and futures)
- **Data**: Real-time and historical OHLCV + funding rates

## Operations

- Check logs in `logs/` directory for troubleshooting
- Verify API credentials in Binance account
- Ensure sufficient balance for position sizes
- Monitor system resources for 24/7 operation