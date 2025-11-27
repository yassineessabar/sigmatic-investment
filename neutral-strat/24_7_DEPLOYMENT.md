# 🚀 24/7 Paper Trading Deployment Guide

Complete setup for running your neutral strategy continuously with monitoring and auto-restart.

## ⚡ Quick Start (5 minutes)

### 1. **Basic 24/7 Setup**
```bash
# Start with monitoring and auto-restart
./scripts/start_24_7.sh monitor
```

### 2. **Check Status**
```bash
# Check if running
./scripts/start_24_7.sh status

# View logs
./scripts/start_24_7.sh follow
```

## 🛠️ Complete Setup Options

### Option A: Manual Control
```bash
# Start trader
./scripts/start_24_7.sh start

# Stop trader
./scripts/start_24_7.sh stop

# Restart trader
./scripts/start_24_7.sh restart

# Check health
./scripts/start_24_7.sh health
```

### Option B: Full Monitoring (Recommended)
```bash
# Start with automatic monitoring and restart
./scripts/start_24_7.sh monitor

# In another terminal, monitor system health
python scripts/monitor_24_7.py
```

### Option C: System Service (VPS/Linux)
```bash
# Install as system service (runs on boot)
sudo cp setup/sigmatic-trader.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable sigmatic-trader
sudo systemctl start sigmatic-trader

# Check service status
sudo systemctl status sigmatic-trader
```

## 📊 What's Running

Your 24/7 setup includes:

### **Core Trading Engine**
- **Script**: `scripts/live_trader.py`
- **Config**: `config/two_week_config.yaml`
- **Frequency**: Hourly trading cycles
- **Pairs**: BTC/ETH, ADA/DOT
- **Mode**: Paper trading (no real money)

### **Performance Monitoring**
- **Script**: `scripts/live_monitor.py`
- **Tracking**: Strategy vs BTC performance
- **Reports**: Daily performance reports
- **Exports**: CSV data exports

### **System Monitoring**
- **Script**: `scripts/monitor_24_7.py`
- **Features**: Auto-restart, health checks, alerts
- **Monitoring**: CPU, memory, disk, network
- **Recovery**: Automatic process restart

## 📁 File Structure

```
neutral-strat/
├── scripts/
│   ├── start_24_7.sh         # Main control script
│   ├── live_trader.py        # Trading engine
│   ├── live_monitor.py       # Performance monitor
│   └── monitor_24_7.py       # System monitor
├── config/
│   └── two_week_config.yaml  # Trading configuration
├── logs/                     # All log files
├── reports/                  # Daily reports
└── exports/                  # CSV exports
```

## 📈 Monitoring & Alerts

### **Real-time Monitoring**
```bash
# Follow trading logs
tail -f logs/trader_24_7.log

# Follow monitor logs
tail -f logs/monitor_24_7.log

# Check performance
python scripts/live_monitor.py
```

### **Health Checks**
```bash
# System health
./scripts/start_24_7.sh health

# Detailed health check
python scripts/monitor_24_7.py check
```

### **Performance Reports**
- **Location**: `reports/daily_report_YYYYMMDD.txt`
- **Frequency**: Daily at midnight
- **Content**: Strategy performance vs BTC, risk metrics, alerts

## 🔧 Configuration

### **Trading Parameters** (`config/two_week_config.yaml`)
```yaml
risk:
  max_daily_dd: 0.03      # 3% max daily drawdown
  max_total_dd: 0.08      # 8% max total drawdown
  leverage_limit: 1.5     # Conservative leverage

execution:
  mode: "paper"           # Paper trading
  interval: "1h"          # Hourly cycles

monitoring:
  daily_reports: true     # Generate daily reports
  alert_thresholds:
    daily_loss_pct: 0.025 # Alert at 2.5% daily loss
```

### **Monitor Configuration** (`config/monitoring.json`)
```json
{
  "monitoring": {
    "check_interval": 60,           // Check every minute
    "health_check_interval": 300,   // Full check every 5 min
    "max_memory_mb": 1024,         // Memory limit
    "max_cpu_percent": 80          // CPU limit
  },
  "alerts": {
    "enabled": true,               // Enable alerts
    "email": {
      "enabled": false             // Email alerts (configure if needed)
    }
  }
}
```

## 🚨 Safety Features

### **Automatic Protections**
- ✅ **Process monitoring** - Auto-restart if crashes
- ✅ **Resource limits** - Stop if using too much CPU/memory
- ✅ **Network checks** - Verify Binance API connectivity
- ✅ **Risk management** - Built-in drawdown limits
- ✅ **Paper trading** - No real money at risk

### **Emergency Stops**
```bash
# Emergency stop everything
./scripts/start_24_7.sh stop
pkill -f live_trader.py

# Kill all related processes
sudo systemctl stop sigmatic-trader  # If using service
```

## 📊 Performance Tracking

### **Daily Metrics**
- Strategy return vs BTC return
- Sharpe ratio, volatility, max drawdown
- Win rate, position count
- Correlation with Bitcoin

### **Data Exports**
```bash
# Export current performance data
python scripts/live_monitor.py

# View CSV exports
ls -la exports/performance_data_*.csv
```

### **Success Criteria** (2-week test)
- ✅ **Positive Sharpe ratio**
- ✅ **Max drawdown < 8%**
- ✅ **Outperform or match BTC**
- ✅ **Stable operation (no crashes)**

## 🔧 Troubleshooting

### **Common Issues**

**1. Process not starting:**
```bash
# Check logs
./scripts/start_24_7.sh logs

# Verify config file
python -c "import yaml; yaml.safe_load(open('config/two_week_config.yaml'))"

# Check permissions
ls -la scripts/start_24_7.sh
```

**2. No trading signals:**
```bash
# Check market data connection
python -c "from data.binance_client import BinanceDataClient; print('OK')"

# Verify API keys (if using real API)
echo $BINANCE_API_KEY
```

**3. High resource usage:**
```bash
# Check process stats
./scripts/start_24_7.sh status

# Monitor resources
htop
```

### **Log Locations**
- **Trading**: `logs/trader_24_7.log`
- **Monitor**: `logs/monitor_24_7.log`
- **Performance**: `logs/live_performance.json`
- **System Service**: `logs/systemd.log`

## 🎯 Next Steps

### **Week 1: Validation**
1. ✅ Verify 24/7 operation
2. ✅ Monitor performance metrics
3. ✅ Check for any crashes/errors
4. ✅ Review daily reports

### **Week 2: Optimization**
1. ✅ Analyze performance patterns
2. ✅ Fine-tune parameters if needed
3. ✅ Prepare for real money (if successful)

### **After 2 Weeks**
Based on performance:
- 🌟 **Excellent** → Deploy with real money (small amount)
- ✅ **Good** → Continue paper trading, optimize
- ⚠️ **Poor** → Back to backtesting and strategy improvement

---

## 🚀 Start Your 24/7 Trading Now!

```bash
# One command to start everything
./scripts/start_24_7.sh monitor
```

Your strategy will now run continuously with:
- ✅ Automatic restarts
- ✅ Performance tracking
- ✅ Risk management
- ✅ Daily reporting
- ✅ Complete monitoring

**Monitor progress:**
```bash
./scripts/start_24_7.sh status
tail -f logs/trader_24_7.log
```