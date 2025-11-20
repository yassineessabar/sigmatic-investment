# Relative Momentum Strategy Guide

## Quick Start

### 1. Setup Environment
```bash
cd neutral-strat
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install ccxt
```

### 2. Run Backtest
```bash
# Activate environment
source venv/bin/activate

# Run backtest with specific dates
python scripts/run_relative_momentum_backtest.py --start-date 2024-01-01 --end-date 2024-06-30

# Run full backtest (uses config dates)
python scripts/run_relative_momentum_backtest.py

# With custom config
python scripts/run_relative_momentum_backtest.py --config my_config.yaml --start-date 2023-01-01
```

### 3. Paper Trading Mode
```bash
# Activate environment
source venv/bin/activate

# Run paper trading (no real money, just signal generation)
python scripts/run_relative_momentum_signals.py
```

## What Each Command Does

### Backtest (`run_relative_momentum_backtest.py`)
- Fetches historical data for BTC vs ALT pairs (AVAX, ETH, SOL, ADA, MATIC)
- Optimizes EMA windows (1-30 days) for each pair
- Calculates performance metrics (Sharpe, Calmar, Max DD)
- Creates equal-weight and volatility-scaled portfolios
- Generates performance plots and correlation heatmap
- Saves results to `relative_momentum_results.csv`

**Sample Output:**
```
BTCUSDT/AVAX: Best EMA window 6d | Sharpe 0.95 | 51.39% annual return
Equal-Weight Portfolio: Sharpe 2.18 | 53.29% annual return
```

### Paper Trading (`run_relative_momentum_signals.py`)
- Monitors live market data
- Generates signals based on EMA crossovers of BTC/ALT ratios
- Logs signals without executing real trades
- Updates every interval (configurable: 1h, 1d, etc.)

**Sample Output:**
```
Signal: BTCUSDT_AVAXUSDT | Base: long 0.0156 BTCUSDT @ 97543.21 |
Alt: short 0.0234 AVAXUSDT @ 42.15 | Confidence: 1.45 |
Reason: Relative momentum: BTCUSDT/AVAXUSDT ratio 2314.567 > EMA 2301.234
```

## Configuration

Edit `config/relative_momentum.yaml` to customize:

```yaml
pairs:
  - base: BTCUSDT
    alt: AVAXUSDT
    ema_window: 10        # EMA window (optimized during backtest)
    allocation_weight: 0.75  # Position size (75% allocation)
    max_notional: 1000    # Maximum position value

execution:
  mode: "paper"         # "paper" | "live"
  interval: "1d"        # Trading frequency
  fees: 0.0006         # Trading fees

backtest:
  start_date: "2023-10-01"  # Backtest start date
```

## Strategy Logic

**Relative Momentum Strategy:**
1. Calculate BTC/ALT price ratio
2. Apply exponential moving average (EMA) to ratio
3. When ratio > EMA: favor BTC (long BTC, short ALT)
4. When ratio < EMA: favor ALT (long ALT, short BTC)
5. Position sizes: ±75% allocation weight

**Key Features:**
- Automatic EMA window optimization (1-30 days)
- Portfolio construction (equal-weight vs volatility-scaled)
- Risk management integration
- Paper trading mode for safe testing

## Files Created
- `src/strategies/relative_momentum.py` - Core strategy logic
- `config/relative_momentum.yaml` - Configuration
- `scripts/run_relative_momentum_backtest.py` - Backtest runner
- `scripts/run_relative_momentum_signals.py` - Live signal generator

## Next Steps

1. **Review backtest results** in CSV file and plots
2. **Tune parameters** in config file if needed
3. **Test paper trading** to understand signal generation
4. **Add API keys** for live data (optional):
   ```bash
   export BINANCE_API_KEY="your_key"
   export BINANCE_API_SECRET="your_secret"
   ```

## Troubleshooting

**"python: command not found"** → Use `python3`

**"Module not found"** → Activate virtual environment:
```bash
source venv/bin/activate
```

**API errors** → Strategy works without API keys in simulation mode

**No signals generated** → Check if market is open and data is available