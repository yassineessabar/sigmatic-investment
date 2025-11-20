# Neutral Strategy Infrastructure

A scalable, production-ready systematic market-neutral crypto trading infrastructure built with Python. This monorepo supports research, backtesting, signal generation, live execution, and risk management for pairs trading strategies.

## 🏗️ Architecture Overview

### Clean Architecture Principles

- **Config-driven strategy**: All parameters configurable via YAML files
- **Pure functions for signals**: No side effects in strategy logic
- **Single source of truth**: Centralized position and PnL management
- **Unified codebase**: Same strategy code for backtesting AND live trading
- **Paper trading mode**: Internal simulation for testing
- **Modular design**: Each component independently testable and scalable

### Core Components

```
/neutral-strat
├── data/              # Data loaders, Binance API utilities, collectors
├── research/          # Jupyter notebooks and experimental strategies
├── config/            # YAML/JSON configuration files
├── src/
│   ├── strategies/    # Signal generation logic (pure functions)
│   ├── backtest/      # Backtesting engine with event loop
│   ├── execution/     # Live trading engine + order routing
│   ├── risk/          # Position sizing, circuit breakers, limits
│   └── utils/         # Shared utilities (logging, config, enums)
├── tests/             # Unit tests for each layer
├── scripts/           # Utility scripts and runners
├── docker-compose.yml # Container orchestration
└── requirements.txt   # Python dependencies
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- Binance API credentials (for live trading)

### Installation

1. **Clone and setup environment**:
```bash
git clone <repository-url>
cd neutral-strat
pip install -r requirements.txt
```

2. **Configure environment variables**:
```bash
export BINANCE_API_KEY="your_api_key"
export BINANCE_API_SECRET="your_api_secret"
```

3. **Run in paper mode** (safe for testing):
```bash
python src/strategies/run_signals.py
```

4. **Run full system with Docker**:
```bash
docker-compose up -d
```

## ⚙️ Configuration

### Example Strategy Config (`config/neutral_pairs.yaml`)

```yaml
pairs:
  - base: BTCUSDT
    hedge: ETHUSDT
    lookback: 100
    entry_z: 2.0      # Z-score threshold for entry
    exit_z: 0.5       # Z-score threshold for exit
    max_notional: 1000

risk:
  max_daily_dd: 0.05      # 5% daily drawdown limit
  max_total_dd: 0.10      # 10% total drawdown limit
  leverage_limit: 2.0     # 2x maximum leverage
  max_position_size: 0.1  # 10% max position per symbol

execution:
  mode: "paper"           # "paper" | "live"
  interval: "1h"          # Trading frequency
  slippage: 0.001         # 0.1% slippage assumption
  fees: 0.001            # 0.1% commission rate
```

## 📊 Strategy Logic

### Market-Neutral Pairs Trading

The strategy implements a **spread reversion** approach:

1. **Spread Calculation**: Monitors price spread between correlated pairs (e.g., BTC/ETH)
2. **Z-Score Normalization**: Calculates z-score over rolling lookback window
3. **Signal Generation**:
   - Entry when |z-score| > entry_threshold
   - Exit when |z-score| < exit_threshold
4. **Position Sizing**: Risk-adjusted sizing based on volatility and confidence

### Signal Flow

```python
# Pure function - no side effects
def compute_signals(data: Dict[str, pd.DataFrame], config: Dict) -> List[PairSignal]:
    for pair in config['pairs']:
        zscore = calculate_spread_zscore(base_prices, hedge_prices, lookback)

        if abs(zscore) >= entry_z:
            # Generate opposing positions
            if zscore > entry_z:
                base_signal = Signal(symbol=base, side='short', ...)
                hedge_signal = Signal(symbol=hedge, side='long', ...)
            else:
                base_signal = Signal(symbol=base, side='long', ...)
                hedge_signal = Signal(symbol=hedge, side='short', ...)
```

## 🔄 Execution Modes

### Paper Trading Mode
- **Internal simulator** with realistic slippage and fees
- **Real market data** but simulated execution
- **Safe testing** environment for strategies
- **Identical code path** as live trading

### Live Trading Mode
- **Real Binance API** execution
- **Risk management** with circuit breakers
- **Order routing** with retry logic
- **Production monitoring**

## 📈 Backtesting Engine

### Event-Driven Architecture

```python
def run_backtest(data_loader, strategy_fn, config):
    portfolio = PortfolioState(initial_capital=100000)

    for timestamp, data_slice in data_loader.create_backtest_iterator():
        # Update positions with current market data
        portfolio.update_positions(data_slice, timestamp)

        # Generate signals using pure strategy function
        signals = strategy_fn(data_slice, config)
        validated_signals = validate_signals(signals, config)

        # Compute required trades
        trades = portfolio.compute_trades(validated_signals, config)

        # Apply trades with slippage and fees
        portfolio.apply_trades(trades, data_slice, config)

    return portfolio.get_results()
```

### Key Features

- **Walk-forward analysis** support
- **Realistic slippage and fees** modeling
- **Risk metrics calculation** (Sharpe, drawdown, etc.)
- **Position-level tracking** with P&L attribution

## 🛡️ Risk Management

### Multi-Layer Risk Controls

1. **Portfolio Level**:
   - Maximum daily drawdown limits
   - Total leverage constraints
   - Correlation exposure limits

2. **Position Level**:
   - Maximum position size per symbol
   - Single order size limits
   - Position concentration rules

3. **Signal Level**:
   - Signal validation and filtering
   - Confidence-based position sizing
   - Volatility-adjusted sizing

### Example Risk Implementation

```python
def enforce_risk_limits(portfolio_state, config):
    if portfolio_state.daily_dd > config["risk"]["max_daily_dd"]:
        return "STOP_TRADING"

    leverage = portfolio_state.gross_exposure / portfolio_state.total_equity
    if leverage > config["risk"]["leverage_limit"]:
        return "REDUCE_POSITION"

    return "OK"
```

## 🐳 Docker Deployment

### Multi-Service Architecture

```yaml
# docker-compose.yml
services:
  data-collector:      # Market data collection
    command: python data/collector.py

  signal-engine:       # Signal generation
    command: python src/strategies/run_signals.py

  execution-engine:    # Order routing and execution
    command: python src/execution/order_router.py

  research-notebook:   # Jupyter for research
    image: jupyter/scipy-notebook

  redis:              # Data caching and messaging
  monitoring:         # Grafana dashboards
```

### Deployment Commands

```bash
# Full production stack
docker-compose up -d

# Research environment only
docker-compose --profile research up

# Backtesting
docker-compose --profile backtest run backtest-runner

# Monitoring
docker-compose --profile monitoring up grafana
```

## 🧪 Testing

### Comprehensive Test Suite

```bash
# Run all tests
pytest tests/

# Test specific modules
pytest tests/test_strategy.py -v
pytest tests/test_backtest.py -v
pytest tests/test_risk.py -v

# Coverage report
pytest --cov=src tests/
```

### Test Categories

- **Strategy Tests**: Signal generation logic validation
- **Backtest Tests**: Portfolio mechanics and P&L calculation
- **Risk Tests**: Risk limit enforcement and position sizing
- **Integration Tests**: End-to-end workflow validation

## 📊 Usage Examples

### Running a Backtest

```bash
python scripts/run_backtest.py \
  --start-date 2023-01-01 \
  --end-date 2023-12-31 \
  --config config/neutral_pairs.yaml \
  --output results/backtest_2023.json
```

### Live Signal Generation

```bash
# Paper mode (safe)
export TRADING_MODE=paper
python src/strategies/run_signals.py

# Live mode (requires API keys)
export TRADING_MODE=live
export BINANCE_API_KEY=your_key
export BINANCE_API_SECRET=your_secret
python src/execution/order_router.py
```

### Research Notebook

```bash
# Start Jupyter environment
docker-compose --profile research up

# Access at http://localhost:8888
# Notebooks located in /research directory
```

## 🔧 Extending the System

### Adding New Strategies

1. **Create strategy module** in `src/strategies/`
2. **Implement signal function** following the interface:
   ```python
   def compute_signals(data: Dict[str, pd.DataFrame], config: Dict) -> List[Signal]
   ```
3. **Add configuration** in `config/`
4. **Write tests** in `tests/`

### Custom Risk Rules

1. **Extend risk checks** in `src/risk/risk_checks.py`
2. **Update configuration** schema
3. **Add validation** in signal processing

### Data Sources

1. **Implement data loader** in `data/`
2. **Follow DataLoader interface**
3. **Add caching and error handling**

## 📈 Performance Monitoring

### Metrics Tracked

- **Returns**: Total return, Sharpe ratio, Calmar ratio
- **Risk**: Maximum drawdown, VaR, volatility
- **Trading**: Win rate, profit factor, trade frequency
- **Execution**: Slippage, fill rates, latency

### Logging Structure

```
logs/
├── trading.log      # Main application log
├── signals.log      # Signal generation events
├── trades.log       # Trade execution records
└── risk.log         # Risk management alerts
```

## 🚨 Production Considerations

### Security

- **API keys** stored as environment variables
- **No hardcoded credentials** in configuration
- **Rate limiting** on API calls
- **Error handling** and retry logic

### Reliability

- **Circuit breakers** for risk management
- **Graceful degradation** on errors
- **Health checks** for all services
- **Data validation** at every layer

### Scalability

- **Horizontal scaling** with Docker Swarm/Kubernetes
- **Redis caching** for data sharing
- **Async processing** for high-frequency data
- **Modular architecture** for easy extension

## 📚 Additional Resources

### Documentation
- [Strategy Development Guide](docs/strategy_development.md)
- [Risk Management Configuration](docs/risk_configuration.md)
- [Production Deployment](docs/production_deployment.md)

### Support
- **Issues**: GitHub Issues
- **Documentation**: In-code docstrings and README
- **Examples**: `/research` notebook examples

---

**⚡ Built for systematic traders who demand production-grade infrastructure with clean, testable, and scalable code.**