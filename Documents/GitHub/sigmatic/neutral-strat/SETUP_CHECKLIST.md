# ✅ VPS Setup Checklist - Quick Reference

## 📋 PREPARATION PHASE

### Prerequisites
- [ ] VPS Forex account with SSH access
- [ ] VPS IP address: ________________
- [ ] VPS login: root / ________________
- [ ] Binance account with API access
- [ ] SSH client ready (Terminal/PuTTY)

### Get Binance API Credentials
- [ ] Log into Binance.com
- [ ] Go to Account → API Management
- [ ] Create new API key labeled "Sigmatic VPS"
- [ ] Enable "Futures Trading" permission
- [ ] Restrict to VPS IP for security
- [ ] Copy API Key: ________________
- [ ] Copy Secret: ________________

---

## 🚀 DEPLOYMENT PHASE

### 1. Connect to VPS
```bash
ssh root@YOUR_VPS_IP
```
- [ ] SSH connection successful
- [ ] Can access command line

### 2. System Update
```bash
apt update && apt upgrade -y
apt install -y curl wget git python3 python3-pip screen htop nano
```
- [ ] System updated without errors
- [ ] All packages installed successfully

### 3. Deploy Strategy
```bash
cd /root
curl -sSL https://raw.githubusercontent.com/yassineessabar/sigmatic-investment/main/neutral-strat/scripts/vps_deploy.sh | bash
```
- [ ] Deployment script completed successfully
- [ ] No error messages during installation

### 4. Configure Credentials
```bash
cd /root/sigmatic-investment/neutral-strat
nano .env
```
**Edit these lines:**
```
BINANCE_API_KEY=your_actual_api_key
BINANCE_API_SECRET=your_actual_secret
BINANCE_TESTNET=true
```
- [ ] API credentials entered correctly
- [ ] File saved with Ctrl+X, Y, Enter

---

## 🧪 TESTING PHASE

### 5. Test Setup
```bash
python3 test_setup.py
```
**Should see:**
- [ ] ✅ Imports successful
- [ ] ✅ BTC Price: $XX,XXX.XX
- [ ] ✅ All directories exist
- [ ] ✅ Live monitor initialized
- [ ] 🎉 SETUP TEST COMPLETED SUCCESSFULLY!

### 6. Start Paper Trading
```bash
./start_strategy.sh
screen -list
```
- [ ] Strategy started in screen session
- [ ] Screen session shows: [number].strategy

### 7. Verify Logs
```bash
tail -f logs/live_monitor.log
```
- [ ] Logs are being generated
- [ ] BTC price fetched successfully
- [ ] No error messages

---

## 📊 MONITORING PHASE

### 8. Dashboard Access
```bash
python3 scripts/dashboard.py
```
**Should display:**
- [ ] Dashboard loads without errors
- [ ] Shows current portfolio value
- [ ] Updates every 5 seconds
- [ ] Exit with Ctrl+C works

### 9. Daily Reports
```bash
ls reports/
cat reports/daily_report_$(date +%Y%m%d).txt
```
- [ ] Reports directory exists
- [ ] Daily report generated
- [ ] Report shows performance metrics

### 10. Performance Data
```bash
cat logs/live_performance.json
```
- [ ] JSON file exists and readable
- [ ] Contains initial_capital: 10000
- [ ] Shows BTC start price

---

## 🚦 WEEK 1 SUCCESS CRITERIA

### Daily Checks (Days 1-7)
- [ ] **Day 1:** Strategy runs without crashes
- [ ] **Day 2:** Daily logs generated consistently
- [ ] **Day 3:** BTC price data fetched daily
- [ ] **Day 4:** Performance calculations working
- [ ] **Day 5:** Dashboard displays correctly
- [ ] **Day 6:** No API rate limit issues
- [ ] **Day 7:** System stable for full week

### Technical Health
- [ ] No strategy crashes or restarts
- [ ] Daily performance vs BTC calculated
- [ ] Sharpe ratio computed (target: >0.5)
- [ ] Drawdown tracking active (<15%)
- [ ] System resources adequate (disk/memory)

### Performance Health
- [ ] Strategy shows market neutrality (BTC correlation <0.8)
- [ ] Position data logged correctly
- [ ] No excessive trading frequency
- [ ] Risk alerts functioning (if triggered)

---

## 🔧 TROUBLESHOOTING QUICK FIXES

### Common Issues:

**"No module named 'ccxt'"**
```bash
pip3 install ccxt pandas numpy
```

**"Permission denied"**
```bash
chmod +x scripts/*.py start_strategy.sh
```

**Strategy not running**
```bash
screen -S strategy -X quit  # Stop old session
./start_strategy.sh         # Start new session
```

**API errors**
```bash
# Test API manually
python3 -c "import ccxt; print(ccxt.binance({'sandbox': True}).fetch_ticker('BTC/USDT'))"
```

**High memory usage**
```bash
# Check resources
htop
# Restart if needed
screen -S strategy -X quit
./start_strategy.sh
```

---

## 🎯 GO/NO-GO DECISION MATRIX

### ✅ PROCEED TO LIVE TRADING IF:
- [ ] **ALL** technical health criteria met
- [ ] **ALL** performance health criteria met
- [ ] Strategy ran successfully for 14+ days
- [ ] No unresolved issues or concerns
- [ ] You understand all monitoring commands
- [ ] Risk management alerts tested and working

### 🚨 DO NOT PROCEED IF:
- [ ] Any technical issues unresolved
- [ ] Strategy crashes or frequent restarts
- [ ] High correlation with BTC (>0.9)
- [ ] Drawdown >25% at any time
- [ ] Frequent API errors or connectivity issues
- [ ] System resource constraints

---

## 📈 SCALING PATHWAY

### Phase 1: Paper Trading (Week 1-2)
- **Capital:** Virtual $10,000
- **Mode:** `BINANCE_TESTNET=true`
- **Focus:** System stability and monitoring

### Phase 2: Micro Live (Week 3-4)
- **Capital:** Real $100-200
- **Mode:** `BINANCE_TESTNET=false`
- **Focus:** Live performance validation

### Phase 3: Progressive Scaling (Month 2+)
- **Capital:** Increase gradually based on performance
- **Targets:**
  - Month 2: $500
  - Month 3: $1,000
  - Month 4+: Scale based on proven results

---

## 🆘 EMERGENCY COMMANDS

### Stop Everything
```bash
screen -S strategy -X quit  # Stop strategy
```

### Backup Data
```bash
cp logs/live_performance.json backups/emergency_$(date +%Y%m%d).json
```

### System Check
```bash
df -h          # Disk space
free -h        # Memory
htop           # CPU/processes
```

### Get Help
```bash
# Generate diagnostic report
cat logs/live_monitor.log | tail -20
ls -la logs/ reports/
python3 --version
```

---

**Remember: This is a systematic approach. Complete each phase before moving to the next. Start small, monitor closely, scale gradually!** 🚀

## 📞 Support Checklist

Before asking for help, ensure you have:
- [ ] Followed all steps in sequence
- [ ] Checked the troubleshooting section
- [ ] Generated diagnostic output
- [ ] Identified the specific error message
- [ ] Noted what step failed

**Success comes from patience and systematic execution!** ✨