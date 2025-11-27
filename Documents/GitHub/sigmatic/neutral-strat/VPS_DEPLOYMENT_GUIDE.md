# VPS Deployment Guide - Sigmatic Neutral Pairs Strategy

## 2-Week Live Paper Trading Setup

This guide provides step-by-step instructions to deploy your neutral pairs strategy on your VPS for 2-week live paper trading.

### 📋 Prerequisites

- VPS access credentials
- Binance API keys (testnet recommended for paper trading)
- SSH access configured
- Local machine with git and ssh

### 🖥️ VPS Information

- **Hostname**: VM23193134392700842.forexvps.net
- **IP Address**: 188.119.102.184
- **OS**: Ubuntu/Debian (assumed)

---

## 🚀 Step-by-Step Deployment

### Step 1: Initial VPS Setup

First, prepare your VPS environment:

```bash
# From your local machine
cd /path/to/sigmatic/neutral-strat
./vps/setup_vps.sh
```

This script will:
- Install Python 3.11 and required packages
- Create necessary users and directories
- Setup firewall and security
- Configure system monitoring
- Setup log rotation and backups

### Step 2: Configure API Keys

Create your environment file:

```bash
# Copy the template
cp vps/.env.template vps/.env

# Edit with your actual API keys
nano vps/.env
```

**Required Configuration:**
```env
BINANCE_API_KEY=your_testnet_api_key
BINANCE_API_SECRET=your_testnet_api_secret
TRADING_MODE=paper
INITIAL_CAPITAL=100000
```

**⚠️ IMPORTANT**: Use Binance testnet API keys for paper trading:
- Get testnet keys from: https://testnet.binance.vision/
- Never use mainnet keys for testing

### Step 3: Deploy Application

Deploy the strategy to your VPS:

```bash
# Deploy using the 2-week configuration
./vps/deploy.sh two_week_config.yaml
```

This will:
- Transfer all necessary files to VPS
- Setup Python virtual environment
- Install dependencies
- Configure systemd service
- Setup monitoring

### Step 4: Start Trading Bot

```bash
# SSH into your VPS
ssh root@188.119.102.184

# Start the trading service
systemctl start sigmatic-trader

# Check status
systemctl status sigmatic-trader
```

### Step 5: Monitor Your Bot

Use the monitoring script to check status:

```bash
# From your local machine
./vps/monitor.sh status     # Basic status
./vps/monitor.sh full       # Comprehensive check
./vps/monitor.sh logs       # Follow live logs
./vps/monitor.sh stats      # Trading statistics
```

---

## 📊 Configuration Details

### 2-Week Trading Parameters

The `two_week_config.yaml` includes:

**Pairs:**
- BTC/ETH (conservative, $800 max)
- ADA/DOT (aggressive, $400 max)

**Risk Management:**
- Max daily drawdown: 3%
- Max total drawdown: 8%
- Stop loss: 1.5%
- Lower leverage: 1.5x

**Execution:**
- Paper trading mode
- 1-hour intervals
- Conservative position sizing

### Monitoring Features

- Real-time performance tracking
- Daily reports at midnight UTC
- Automatic health checks
- Resource usage monitoring
- Trade logging

---

## 🛠️ Management Commands

### Service Management

```bash
# Start/stop/restart the bot
systemctl start sigmatic-trader
systemctl stop sigmatic-trader
systemctl restart sigmatic-trader

# View service status
systemctl status sigmatic-trader

# View logs
journalctl -u sigmatic-trader -f
```

### Direct Log Access

```bash
# Trading logs
tail -f /var/log/sigmatic/trading_2week.log

# System logs
tail -f /var/log/sigmatic/system_monitor.log
```

### Emergency Stop

```bash
# Stop trading immediately
systemctl stop sigmatic-trader

# Disable auto-start
systemctl disable sigmatic-trader
```

---

## 📈 Monitoring and Performance

### Key Metrics to Watch

1. **Total Equity**: Should grow steadily
2. **Unrealized PnL**: Current position profits/losses
3. **Drawdown**: Maximum loss from peak
4. **Trade Success Rate**: Percentage of profitable trades
5. **System Resources**: Memory and CPU usage

### Daily Monitoring Routine

1. **Morning Check** (9 AM UTC):
   ```bash
   ./vps/monitor.sh stats
   ```

2. **Evening Review** (9 PM UTC):
   ```bash
   ./vps/monitor.sh full
   ```

3. **Weekly Deep Dive** (Sundays):
   - Review all logs
   - Check performance metrics
   - Verify system health

---

## 🚨 Troubleshooting

### Common Issues

1. **Service Won't Start**
   ```bash
   # Check logs for errors
   journalctl -u sigmatic-trader --no-pager -l

   # Verify configuration
   python3 -c "import yaml; print(yaml.safe_load(open('/opt/sigmatic/config/two_week_config.yaml')))"
   ```

2. **No Trading Signals**
   - Check Binance API connectivity
   - Verify market data is updating
   - Review strategy parameters

3. **High Memory Usage**
   ```bash
   # Monitor process
   top -p $(pgrep -f live_trader.py)

   # Restart if needed
   systemctl restart sigmatic-trader
   ```

4. **Network Issues**
   ```bash
   # Test Binance connectivity
   curl -s https://testnet.binance.vision/api/v3/ping
   ```

### Emergency Procedures

**Complete System Reset:**
```bash
# Stop service
systemctl stop sigmatic-trader

# Backup current state
/opt/sigmatic/backup.sh

# Redeploy if needed
./vps/deploy.sh two_week_config.yaml
```

---

## 📊 Expected Performance

### 2-Week Goals

- **Target Return**: 2-5%
- **Maximum Drawdown**: <5%
- **Sharpe Ratio**: >1.0
- **Trade Frequency**: 5-15 trades per week

### Success Metrics

- ✅ Positive total return
- ✅ Drawdown within limits
- ✅ System uptime >99%
- ✅ No critical errors

---

## 🔐 Security Considerations

1. **API Keys**: Use testnet only
2. **Firewall**: Only necessary ports open
3. **User Permissions**: Non-root execution
4. **Log Security**: No sensitive data in logs
5. **Backups**: Encrypted if containing sensitive data

---

## 📞 Support and Maintenance

### Log Files Location

- **Trading Logs**: `/var/log/sigmatic/trading_2week.log`
- **System Logs**: `/var/log/sigmatic/system_monitor.log`
- **Service Logs**: `journalctl -u sigmatic-trader`

### Backup Schedule

- **Hourly**: State snapshots
- **Daily**: Full system backup
- **Weekly**: Log archives

### End of 2-Week Period

The bot will automatically stop after 2 weeks (configured end date). To extend:

1. Update `deployment.end_date` in config
2. Restart service: `systemctl restart sigmatic-trader`

---

## 🎯 Quick Start Summary

1. **Setup VPS**: `./vps/setup_vps.sh`
2. **Configure API keys**: Edit `vps/.env`
3. **Deploy**: `./vps/deploy.sh two_week_config.yaml`
4. **Monitor**: `./vps/monitor.sh status`
5. **Check daily**: Review performance and logs

**That's it! Your neutral pairs strategy is now running live on your VPS for 2-week paper trading.**

---

*For issues or questions, check the troubleshooting section or review the log files for detailed error information.*