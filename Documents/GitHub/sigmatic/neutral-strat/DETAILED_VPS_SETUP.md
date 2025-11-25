# 🚀 DETAILED VPS Forex Setup Guide - Step by Step

## 📋 Prerequisites Checklist

Before starting, ensure you have:
- [ ] VPS Forex account with SSH access
- [ ] VPS IP address and login credentials
- [ ] Binance account with API keys
- [ ] Basic command line knowledge
- [ ] SSH client (Terminal on Mac/Linux, PuTTY on Windows)

---

## STEP 1: GET YOUR VPS INFORMATION 📝

### 1.1 Find Your VPS Details
From your VPS Forex control panel, note down:
```
VPS IP Address: ___.___.___.___ (example: 192.168.1.100)
Username: root (usually)
Password: _____________ (or SSH key)
OS: Ubuntu/CentOS/Debian
```

### 1.2 Test SSH Connection
```bash
# On Mac/Linux Terminal or Windows WSL:
ssh root@YOUR_VPS_IP

# If using password, enter it when prompted
# If using SSH key, make sure it's configured

# First login might ask to accept fingerprint - type "yes"
```

**Expected Result:**
```
Welcome to Ubuntu 20.04.3 LTS (GNU/Linux 5.4.0-74-generic x86_64)
root@vps-name:~#
```

---

## STEP 2: PREPARE VPS ENVIRONMENT 🔧

### 2.1 Update System (CRITICAL)
```bash
# Update package lists
apt update

# Upgrade all packages (this may take 5-10 minutes)
apt upgrade -y

# Verify updates completed successfully
echo "System updated successfully"
```

### 2.2 Install Essential Tools
```bash
# Install required packages
apt install -y curl wget git python3 python3-pip screen htop nano

# Verify installations
python3 --version    # Should show Python 3.x.x
pip3 --version       # Should show pip version
git --version        # Should show git version
```

### 2.3 Create Working Directory
```bash
# Go to root directory
cd /root

# Create project directory
mkdir -p sigmatic-trading
cd sigmatic-trading

# Verify you're in the right place
pwd  # Should show /root/sigmatic-trading
```

---

## STEP 3: GET BINANCE API CREDENTIALS 🔑

### 3.1 Create Binance API Key
1. **Log into Binance.com**
2. **Go to Account → API Management**
3. **Create API Key:**
   - Label: "Sigmatic Trading VPS"
   - Enable: "Futures Trading" ✅
   - Enable: "Read" ✅
   - Restrict IP: Add your VPS IP (IMPORTANT for security)

### 3.2 Save Credentials Securely
```bash
# Create a temporary note file (we'll move this later)
nano api_credentials.txt
```

**Type in nano:**
```
BINANCE_API_KEY=your_actual_api_key_here
BINANCE_API_SECRET=your_actual_secret_here
```

**Save and exit nano:**
- Press `Ctrl + X`
- Press `Y` to confirm
- Press `Enter` to save

### 3.3 Test API Connection
```bash
# Install ccxt first
pip3 install ccxt

# Test API connection
python3 -c "
import ccxt
exchange = ccxt.binance({
    'apiKey': 'your_api_key_here',
    'secret': 'your_secret_here',
    'sandbox': True
})
print('✅ API connection successful!')
print('Account info:', exchange.fetch_balance())
"
```

**Expected Result:** Should show account balance info without errors

---

## STEP 4: DOWNLOAD AND DEPLOY STRATEGY 📦

### 4.1 Download Deployment Script
```bash
# Method 1: Direct download
curl -sSL https://raw.githubusercontent.com/yassineessabar/sigmatic-investment/main/neutral-strat/scripts/vps_deploy.sh -o vps_deploy.sh

# OR Method 2: Clone entire repository
git clone https://github.com/yassineessabar/sigmatic-investment.git
```

### 4.2 Make Script Executable and Run
```bash
# If you downloaded the script directly:
chmod +x vps_deploy.sh
./vps_deploy.sh

# OR if you cloned the repository:
cd sigmatic-investment/neutral-strat
chmod +x scripts/vps_deploy.sh
./scripts/vps_deploy.sh
```

### 4.3 Verify Deployment
```bash
# Check if deployment created the right structure
ls -la /root/sigmatic-investment/neutral-strat/

# Should see:
# - scripts/ directory
# - config/ directory
# - .env file
# - logs/ directory
# - start_strategy.sh file
```

---

## STEP 5: CONFIGURE API CREDENTIALS 🔐

### 5.1 Navigate to Strategy Directory
```bash
cd /root/sigmatic-investment/neutral-strat
pwd  # Verify you're in the right place
```

### 5.2 Edit Configuration File
```bash
# Open the environment file
nano .env
```

### 5.3 Update .env File
**Replace the template with your actual credentials:**

**BEFORE (template):**
```
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_secret_here
BINANCE_TESTNET=true
```

**AFTER (your actual values):**
```
BINANCE_API_KEY=K8Jh2v7XpQm9RtYuI3dF5gH6jK8L2nM4  # Your actual key
BINANCE_API_SECRET=x7VbN2mP9qR3sT6uY8zA4eF1gH5jK9L2  # Your actual secret
BINANCE_TESTNET=true  # Keep as true for testing

# Strategy Configuration
INITIAL_CAPITAL=10000
MAX_POSITION_SIZE=0.2
MAX_DRAWDOWN=0.25

# Monitoring
ALERT_EMAIL=your_email@gmail.com  # Optional
TZ=UTC
```

**Save and exit:**
- Press `Ctrl + X`
- Press `Y`
- Press `Enter`

### 5.4 Secure the Credentials File
```bash
# Make file readable only by root
chmod 600 .env

# Verify permissions
ls -la .env
# Should show: -rw------- 1 root root ... .env
```

---

## STEP 6: TEST THE SETUP 🧪

### 6.1 Install Python Dependencies
```bash
# Install required packages
pip3 install pandas numpy ccxt python-dateutil PyYAML matplotlib seaborn

# Verify installation
python3 -c "
import pandas, numpy, ccxt, yaml
print('✅ All packages installed successfully!')
"
```

### 6.2 Test Strategy Components
```bash
# Test 1: API Connection
python3 -c "
import sys, os
sys.path.append('.')
import ccxt
exchange = ccxt.binance({
    'enableRateLimit': True,
    'sandbox': True,  # Testnet
    'options': {'defaultType': 'future'}
})
price = exchange.fetch_ticker('BTC/USDT')
print(f'✅ BTC Price: \${price[\"last\"]:,.2f}')
"

# Test 2: Monitor System
python3 scripts/live_monitor.py

# Expected: Should run without errors and show "Insufficient data" message
```

### 6.3 Create Test Run
```bash
# Create a test script to verify everything works
cat > test_setup.py << 'EOF'
#!/usr/bin/env python3
import sys
import os
sys.path.append('.')

try:
    # Test imports
    import pandas as pd
    import numpy as np
    import ccxt
    print("✅ Imports successful")

    # Test exchange connection
    exchange = ccxt.binance({'sandbox': True, 'enableRateLimit': True})
    btc_price = exchange.fetch_ticker('BTC/USDT')
    print(f"✅ BTC Price: ${btc_price['last']:,.2f}")

    # Test directory structure
    required_dirs = ['logs', 'scripts', 'config', 'reports', 'exports']
    for directory in required_dirs:
        if os.path.exists(directory):
            print(f"✅ Directory {directory} exists")
        else:
            print(f"❌ Directory {directory} missing")
            os.makedirs(directory, exist_ok=True)
            print(f"✅ Created {directory}")

    # Test live monitor
    from scripts.live_monitor import LivePerformanceMonitor
    monitor = LivePerformanceMonitor()
    print("✅ Live monitor initialized successfully")

    print("\n🎉 SETUP TEST COMPLETED SUCCESSFULLY!")
    print("Ready to start paper trading!")

except Exception as e:
    print(f"❌ Error: {e}")
    print("Please check the error and fix before proceeding")
EOF

python3 test_setup.py
```

---

## STEP 7: START PAPER TRADING 📊

### 7.1 Start Strategy in Screen Session
```bash
# Make start script executable
chmod +x start_strategy.sh

# Start the strategy
./start_strategy.sh

# Verify it started
screen -list
# Should show: There is a screen on: [number].strategy
```

### 7.2 View Running Strategy
```bash
# Attach to the strategy screen
screen -r strategy

# You should see the strategy running
# To detach without stopping: Press Ctrl+A, then D
```

### 7.3 Monitor Logs
```bash
# Open another SSH session to monitor logs
ssh root@YOUR_VPS_IP
cd /root/sigmatic-investment/neutral-strat

# View live logs
tail -f logs/live_monitor.log

# Should show regular updates like:
# 2025-11-25 12:00:00 - INFO - Daily Performance: Strategy: 0.00%, BTC: 0.00%
```

---

## STEP 8: MONITOR PERFORMANCE 📈

### 8.1 Daily Dashboard
```bash
# View real-time dashboard (updates every 5 seconds)
python3 scripts/dashboard.py

# Expected display:
# ============================================
# 🚀 SIGMATIC STRATEGY LIVE DASHBOARD
# ============================================
# Last Updated: 2025-11-25 12:00:00 UTC
#
# 📊 PERFORMANCE OVERVIEW:
#    Portfolio Value: $10,000.00
#    Strategy Return: +0.00%
#    BTC Return: +0.00%
#    Outperformance: +0.00%
```

### 8.2 Check Performance Data
```bash
# View raw performance data
cat logs/live_performance.json

# Expected initial content:
# {
#   "start_date": "2025-11-25T12:00:00",
#   "initial_capital": 10000,
#   "daily_performance": [],
#   "btc_start_price": null
# }
```

### 8.3 Generate Reports
```bash
# Generate daily report
python3 -c "
from scripts.live_monitor import LivePerformanceMonitor
monitor = LivePerformanceMonitor()
monitor.generate_daily_report()
"

# View the report
ls reports/
cat reports/daily_report_$(date +%Y%m%d).txt
```

---

## STEP 9: TROUBLESHOOTING 🔧

### 9.1 Common Issues and Fixes

**Issue: "Command not found: python3"**
```bash
# Fix: Install Python
apt update
apt install python3 python3-pip -y
```

**Issue: "No module named 'ccxt'"**
```bash
# Fix: Install missing packages
pip3 install ccxt pandas numpy
```

**Issue: "Permission denied"**
```bash
# Fix: Make files executable
chmod +x scripts/*.py
chmod +x start_strategy.sh
```

**Issue: "API key invalid"**
```bash
# Fix: Check your .env file
cat .env | grep BINANCE_API_KEY
# Verify the key is correct and has futures permissions
```

**Issue: Strategy not running**
```bash
# Check if screen session exists
screen -list

# If not running, restart
./start_strategy.sh

# Check logs for errors
tail -20 logs/live_monitor.log
```

### 9.2 Health Check Commands
```bash
# System resources
htop  # Press 'q' to exit

# Disk space
df -h

# Network connectivity to Binance
ping api.binance.com  # Press Ctrl+C to stop

# Python packages
pip3 list | grep -E "(ccxt|pandas|numpy)"

# Strategy process
ps aux | grep python
```

---

## STEP 10: SUCCESS VERIFICATION ✅

### 10.1 Week 1 Checklist

After running for 7 days, verify:

**Technical Health:**
- [ ] Strategy runs continuously without crashes
- [ ] Daily logs are being generated
- [ ] BTC price data is fetched every day
- [ ] No repeated API errors in logs

**Performance Tracking:**
- [ ] Daily performance vs BTC is calculated
- [ ] Portfolio value is tracked accurately
- [ ] Sharpe ratio is computed (target: >0.5)
- [ ] Drawdown monitoring is active (should stay <15%)

**System Stability:**
- [ ] VPS has sufficient disk space (check with `df -h`)
- [ ] Memory usage is stable (check with `free -h`)
- [ ] No network connectivity issues
- [ ] Screen sessions remain active

### 10.2 Performance Targets

**Daily Targets:**
- Strategy generates signals without errors
- Performance data is logged consistently
- Dashboard displays current metrics

**Weekly Targets:**
- Average daily Sharpe ratio >0.5
- Maximum drawdown <15%
- System uptime >95%
- BTC correlation <0.8 (market neutral)

### 10.3 Ready for Live Trading

**Only proceed to real money if ALL of the following are true:**
- [ ] Paper trading runs flawlessly for 14+ days
- [ ] Strategy shows positive Sharpe ratio
- [ ] No technical issues or crashes
- [ ] You understand all monitoring commands
- [ ] Risk management alerts are working
- [ ] You're comfortable with the system

---

## 🚨 EMERGENCY PROCEDURES

### Stop Strategy Immediately
```bash
# Kill strategy screen session
screen -S strategy -X quit

# Verify it stopped
screen -list  # Should show "No Sockets found"
```

### Backup Important Data
```bash
# Create backup of performance data
cp logs/live_performance.json backups/emergency_backup_$(date +%Y%m%d).json

# Backup all logs
tar -czf emergency_logs_$(date +%Y%m%d).tar.gz logs/ reports/
```

### Emergency Diagnostic
```bash
# Generate emergency diagnostic report
cat > emergency_diagnostic.txt << EOF
EMERGENCY DIAGNOSTIC REPORT
Generated: $(date)

System Status:
$(uptime)

Disk Space:
$(df -h)

Memory:
$(free -h)

Python Processes:
$(ps aux | grep python)

Recent Logs:
$(tail -20 logs/live_monitor.log)

Performance Data:
$(ls -la logs/ reports/)

Network Test:
$(ping -c 3 api.binance.com)
EOF

cat emergency_diagnostic.txt
```

---

## 🎯 NEXT STEPS

### Week 1-2: Paper Trading Phase
- Run in testnet mode (`BINANCE_TESTNET=true`)
- Monitor all systems and performance
- Verify strategy behavior matches expectations

### Week 3-4: Micro Live Phase
- Switch to live mode (`BINANCE_TESTNET=false`)
- Start with tiny capital ($100-200)
- Monitor very closely for any differences

### Month 2+: Scaling Phase
- Gradually increase capital based on performance
- Continue monitoring and optimization
- Scale up only after consistent positive results

**Remember: Start small, monitor closely, scale gradually!** 🚀

---

## 📞 SUPPORT INFORMATION

If you encounter issues:

1. **Check logs first**: `tail -20 logs/live_monitor.log`
2. **Run diagnostic**: Execute the emergency diagnostic script above
3. **Verify API connectivity**: Test Binance API connection
4. **Check system resources**: Ensure VPS has adequate CPU/memory/disk

The strategy has been tested extensively and should run smoothly with proper setup! 💪