# 🚀 VPS Forex Quick Start Guide

## Step 1: Connect to Your VPS

```bash
# SSH into your VPS Forex server
ssh root@your-vps-ip

# Or use your VPS control panel's web terminal
```

## Step 2: One-Command Deployment

```bash
# Download and run the deployment script
curl -sSL https://raw.githubusercontent.com/yassineessabar/sigmatic-investment/main/neutral-strat/scripts/vps_deploy.sh | bash

# Or manually:
wget https://raw.githubusercontent.com/yassineessabar/sigmatic-investment/main/neutral-strat/scripts/vps_deploy.sh
chmod +x vps_deploy.sh
./vps_deploy.sh
```

## Step 3: Configure API Credentials

```bash
cd /root/sigmatic-investment/neutral-strat

# Edit the configuration file
nano .env

# Replace these lines:
BINANCE_API_KEY=your_actual_api_key
BINANCE_API_SECRET=your_actual_secret
BINANCE_TESTNET=true  # Keep true for initial testing
```

## Step 4: Test the Setup

```bash
# Test the monitoring system
python3 scripts/live_monitor.py

# This should:
# ✅ Fetch BTC price successfully
# ✅ Create log files
# ✅ Show "Insufficient data" (normal for first run)
```

## Step 5: Start Paper Trading

```bash
# Start strategy in screen session
./start_strategy.sh

# View the running strategy
screen -r strategy

# Detach from screen: Ctrl+A, then D
```

## Step 6: Monitor Performance

### Real-time Dashboard
```bash
# View live dashboard (refreshes every 5 seconds)
python3 scripts/dashboard.py
```

### Daily Reports
```bash
# View today's performance report
cat reports/daily_report_$(date +%Y%m%d).txt

# View live performance log
tail -f logs/live_monitor.log
```

### Performance Data
```bash
# View raw performance data
cat logs/live_performance.json | jq '.' || cat logs/live_performance.json

# Export to CSV for analysis
python3 -c "
from scripts.live_monitor import LivePerformanceMonitor
monitor = LivePerformanceMonitor()
monitor.export_performance_csv()
"
```

## Step 7: Key Monitoring Commands

```bash
# Check if strategy is running
screen -list

# Restart strategy if needed
screen -S strategy -X quit
./start_strategy.sh

# View recent logs
tail -50 logs/live_monitor.log

# Check system resources
htop

# Monitor network connectivity to Binance
ping api.binance.com
```

## 📊 Success Indicators (Week 1)

After running for 7 days, you should see:

✅ **Technical Health:**
- Strategy runs without errors
- Daily logs are generated
- BTC price data is fetched successfully
- No API rate limit issues

✅ **Performance Tracking:**
- Daily performance vs BTC is calculated
- Sharpe ratio is computed (should be >0.5 for good performance)
- Drawdown is tracked (should stay <15%)
- Position data is logged

✅ **Risk Management:**
- Alerts are triggered if drawdown >10%
- Strategy shows market neutrality (BTC correlation <0.8)
- No single position >50% of portfolio

## 🚨 Red Flags - Stop If:

❌ **Technical Issues:**
- Frequent API errors or disconnections
- Strategy crashes repeatedly
- Data fetching failures

❌ **Performance Issues:**
- Sharpe ratio <0.2 for 30+ days
- Drawdown >25% at any time
- Highly correlated with BTC (>0.9) for 7+ days

❌ **Risk Issues:**
- Excessive position sizes
- Frequent trading (>2 rebalances/day)
- Unexplained portfolio value changes

## 📈 Scaling Path

### Week 1-2: Paper Trading
- Keep `BINANCE_TESTNET=true`
- Monitor all metrics
- Verify system stability

### Week 3-4: Micro Live Trading
- Set `BINANCE_TESTNET=false`
- Set `INITIAL_CAPITAL=100` (start small!)
- Monitor closely for any issues

### Month 2+: Progressive Scaling
- Increase capital gradually:
  - Month 2: $500
  - Month 3: $1,000
  - Month 4+: Scale based on performance

## 🛠️ Troubleshooting

### Common Issues:

**"No module named 'ccxt'"**
```bash
pip3 install ccxt
```

**"Permission denied" errors**
```bash
chmod +x scripts/*.py
```

**Strategy not generating signals**
```bash
# Check if testnet is accessible
python3 -c "import ccxt; print(ccxt.binance({'sandbox': True}).fetch_ticker('BTC/USDT'))"
```

**High memory usage**
```bash
# Restart strategy to clear memory
screen -S strategy -X quit
./start_strategy.sh
```

## 📞 Support Commands

```bash
# Generate diagnostic report
cat > diagnostic_report.txt << EOF
Date: $(date)
Disk Space: $(df -h /)
Memory: $(free -h)
Python Version: $(python3 --version)
Strategy Logs (last 10 lines):
$(tail -10 logs/live_monitor.log)
Performance Data:
$(ls -la logs/ reports/)
EOF

# Send this report if you need help
```

## 🎯 Success Metrics Summary

**Daily Monitoring:**
- [ ] Strategy running without errors
- [ ] Daily performance calculated
- [ ] BTC comparison working
- [ ] Risk metrics computed

**Weekly Review:**
- [ ] Consistent outperformance vs BTC
- [ ] Sharpe ratio >0.8
- [ ] Maximum drawdown <15%
- [ ] System stability maintained

**Monthly Scaling Decision:**
- [ ] All technical metrics green
- [ ] Performance meets expectations
- [ ] Ready for capital increase

Remember: **Start small, monitor closely, scale gradually!** 🚀