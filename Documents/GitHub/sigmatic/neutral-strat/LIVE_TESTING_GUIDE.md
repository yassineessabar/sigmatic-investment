# 🚀 Live Strategy Testing Guide - VPS Forex Setup

## Overview

This guide shows how to safely test your neutral momentum strategy live on VPS Forex, starting with paper trading and gradually moving to small real positions.

## 🔧 VPS Forex Setup

### 1. **VPS Requirements**
```bash
# Minimum VPS specs for crypto trading
- CPU: 2+ cores
- RAM: 4GB+
- Storage: 20GB SSD
- OS: Ubuntu 20.04+ or CentOS 8+
- Network: Low latency to Binance servers
```

### 2. **Initial VPS Setup**
```bash
# Connect to your VPS
ssh root@your-vps-ip

# Update system
apt update && apt upgrade -y

# Install Python and dependencies
apt install python3 python3-pip git screen htop -y

# Install pip packages
pip3 install pandas numpy ccxt python-dateutil PyYAML matplotlib seaborn
```

### 3. **Clone and Setup Repository**
```bash
# Clone your repository
cd /root
git clone https://github.com/yassineessabar/sigmatic-investment.git
cd sigmatic-investment/neutral-strat

# Create secure credentials file
cat > .env << EOF
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_secret_here
BINANCE_TESTNET=true  # Start with testnet
EOF

chmod 600 .env
```

## 📊 Live Monitoring Framework

### 1. **Create Live Monitor Script**
```python
# File: scripts/live_monitor.py
#!/usr/bin/env python3
"""
Live Strategy Performance Monitor for VPS
Tracks strategy performance vs BTC in real-time
"""

import sys
import os
import time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List
import pandas as pd
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/live_monitor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class LivePerformanceMonitor:
    def __init__(self, log_file='logs/live_performance.json'):
        self.log_file = log_file
        self.performance_data = self.load_performance_log()

    def load_performance_log(self):
        """Load existing performance data"""
        try:
            with open(self.log_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {
                'start_date': datetime.now().isoformat(),
                'daily_performance': [],
                'positions': [],
                'metrics': {}
            }

    def save_performance_log(self):
        """Save performance data"""
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        with open(self.log_file, 'w') as f:
            json.dump(self.performance_data, f, indent=2)

    def log_daily_performance(self, strategy_pnl: float, btc_pnl: float,
                             portfolio_value: float, positions: Dict):
        """Log daily performance vs BTC"""
        today = datetime.now().date().isoformat()

        daily_data = {
            'date': today,
            'strategy_pnl': strategy_pnl,
            'btc_pnl': btc_pnl,
            'portfolio_value': portfolio_value,
            'outperformance': strategy_pnl - btc_pnl,
            'positions': positions,
            'timestamp': datetime.now().isoformat()
        }

        self.performance_data['daily_performance'].append(daily_data)
        self.save_performance_log()

        logger.info(f"Daily Performance: Strategy PnL: {strategy_pnl:.2%}, "
                   f"BTC PnL: {btc_pnl:.2%}, Outperformance: {strategy_pnl-btc_pnl:.2%}")

    def calculate_live_metrics(self) -> Dict:
        """Calculate live strategy metrics"""
        if len(self.performance_data['daily_performance']) < 2:
            return {}

        df = pd.DataFrame(self.performance_data['daily_performance'])
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)

        # Calculate returns
        strategy_returns = df['strategy_pnl'].diff()
        btc_returns = df['btc_pnl'].diff()

        # Calculate metrics
        days_trading = len(df)
        total_strategy_return = df['strategy_pnl'].iloc[-1]
        total_btc_return = df['btc_pnl'].iloc[-1]

        strategy_vol = strategy_returns.std() * np.sqrt(365) if len(strategy_returns) > 1 else 0
        strategy_sharpe = (total_strategy_return / strategy_vol * np.sqrt(365/days_trading)) if strategy_vol > 0 else 0

        metrics = {
            'days_trading': days_trading,
            'total_strategy_return': total_strategy_return,
            'total_btc_return': total_btc_return,
            'outperformance': total_strategy_return - total_btc_return,
            'strategy_sharpe': strategy_sharpe,
            'win_rate': (strategy_returns > 0).mean() if len(strategy_returns) > 1 else 0,
            'current_portfolio_value': df['portfolio_value'].iloc[-1]
        }

        self.performance_data['metrics'] = metrics
        return metrics

    def generate_daily_report(self):
        """Generate daily performance report"""
        metrics = self.calculate_live_metrics()

        if not metrics:
            logger.info("Insufficient data for metrics calculation")
            return

        report = f"""
========================================
LIVE STRATEGY PERFORMANCE REPORT
========================================
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Days Trading: {metrics['days_trading']}

Portfolio Value: ${metrics['current_portfolio_value']:,.2f}

Performance:
  Strategy Return: {metrics['total_strategy_return']:.2%}
  BTC Return: {metrics['total_btc_return']:.2%}
  Outperformance: {metrics['outperformance']:.2%}

Risk Metrics:
  Strategy Sharpe: {metrics['strategy_sharpe']:.2f}
  Win Rate: {metrics['win_rate']:.1%}

Status: {'✅ OUTPERFORMING' if metrics['outperformance'] > 0 else '⚠️ UNDERPERFORMING'}
========================================
"""
        logger.info(report)

        # Save report to file
        with open(f"reports/daily_report_{datetime.now().strftime('%Y%m%d')}.txt", 'w') as f:
            f.write(report)

# Example usage in main script
if __name__ == "__main__":
    monitor = LivePerformanceMonitor()

    # Example daily logging (integrate with your strategy)
    monitor.log_daily_performance(
        strategy_pnl=0.015,  # 1.5% daily return
        btc_pnl=0.008,       # 0.8% BTC return
        portfolio_value=10150,
        positions={'BTCUSDT/AVAX': {'weight': 0.75, 'pnl': 0.02}}
    )

    monitor.generate_daily_report()
```

### 2. **Create Position Tracker**
```python
# File: scripts/position_tracker.py
#!/usr/bin/env python3
"""
Live Position Tracking and Risk Management
"""

import ccxt
import json
import time
from datetime import datetime
import logging

class PositionTracker:
    def __init__(self, api_key, api_secret, testnet=True):
        self.exchange = ccxt.binance({
            'apiKey': api_key,
            'secret': api_secret,
            'sandbox': testnet,
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        })

    def get_current_positions(self) -> Dict:
        """Get all current positions"""
        try:
            positions = self.exchange.fetch_positions()
            active_positions = {p['symbol']: p for p in positions if p['size'] != 0}
            return active_positions
        except Exception as e:
            logger.error(f"Error fetching positions: {e}")
            return {}

    def calculate_portfolio_pnl(self) -> float:
        """Calculate total portfolio PnL"""
        positions = self.get_current_positions()
        total_pnl = sum(pos['unrealizedPnl'] for pos in positions.values())
        return total_pnl

    def check_risk_limits(self) -> Dict:
        """Check if positions exceed risk limits"""
        positions = self.get_current_positions()

        # Risk checks
        total_exposure = sum(abs(pos['notional']) for pos in positions.values())
        max_single_position = max([abs(pos['notional']) for pos in positions.values()] + [0])

        portfolio_value = 10000  # Your starting capital

        risk_status = {
            'total_exposure_pct': total_exposure / portfolio_value,
            'max_position_pct': max_single_position / portfolio_value,
            'position_count': len(positions),
            'warnings': []
        }

        # Risk warnings
        if risk_status['total_exposure_pct'] > 2.0:  # 200% exposure limit
            risk_status['warnings'].append("Total exposure exceeds 200%")

        if risk_status['max_position_pct'] > 0.5:  # 50% single position limit
            risk_status['warnings'].append("Single position exceeds 50%")

        if risk_status['position_count'] > 10:
            risk_status['warnings'].append("Too many open positions")

        return risk_status
```

## 🎯 Safe Testing Progression

### **Phase 1: Paper Trading (Week 1-2)**
```bash
# Test with Binance Testnet
export BINANCE_TESTNET=true
export INITIAL_CAPITAL=10000  # Virtual money

# Run paper trading
screen -S strategy
python3 scripts/run_relative_momentum_signals.py --paper-trading
```

### **Phase 2: Micro Real Testing (Week 3-4)**
```bash
# Switch to real account with tiny amounts
export BINANCE_TESTNET=false
export INITIAL_CAPITAL=100    # Start with $100

# Run with micro positions
python3 scripts/run_relative_momentum_backtest.py --live --capital 100
```

### **Phase 3: Progressive Scaling (Month 2+)**
```bash
# Gradually increase if performance is good
# Week 5-6: $500
# Week 7-8: $1,000
# Month 2+: Scale based on performance
```

## 📈 Key Performance Indicators (KPIs)

### **Daily Monitoring Checklist**
```bash
# Check these metrics daily:

1. Strategy PnL vs BTC PnL
   Target: Strategy >= BTC - 1%

2. Sharpe Ratio (rolling 30-day)
   Target: >= 0.8

3. Maximum Drawdown
   Alert: > 15%
   Stop: > 25%

4. Position Count
   Normal: 2-4 pairs
   Alert: > 6 pairs

5. Trading Frequency
   Normal: 1-3 rebalances per week
   Alert: > 1 rebalance per day

6. API Rate Limits
   Monitor: < 80% of limits used
```

## 🚨 Risk Management & Alerts

### **Automated Alert System**
```python
# File: scripts/alert_system.py
def check_strategy_health():
    """Daily health check with alerts"""

    # Performance check
    if daily_pnl < -0.05:  # -5% daily loss
        send_alert("⚠️ DAILY LOSS EXCEEDS 5%")

    # Drawdown check
    if current_drawdown < -0.15:  # -15% drawdown
        send_alert("🚨 DRAWDOWN EXCEEDS 15%")

    # Position size check
    if max_position_size > 0.5:  # 50% in single position
        send_alert("⚠️ POSITION SIZE TOO LARGE")

    # Correlation check
    if strategy_btc_correlation > 0.8:  # High correlation
        send_alert("⚠️ STRATEGY NOT MARKET NEUTRAL")

def send_alert(message):
    """Send alert via email/Telegram/Discord"""
    # Implement your preferred notification method
    print(f"ALERT: {message}")

    # Example: Send to Telegram
    # requests.post(f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
    #              data={'chat_id': CHAT_ID, 'text': message})
```

## 🛠️ VPS Maintenance Commands

### **Daily Operations**
```bash
# Check strategy status
screen -r strategy

# View recent logs
tail -f logs/live_monitor.log

# Check system resources
htop

# Backup performance data
cp logs/live_performance.json backups/performance_$(date +%Y%m%d).json

# Check internet connectivity to Binance
ping api.binance.com
```

### **Weekly Health Check**
```bash
# Generate weekly performance report
python3 scripts/generate_weekly_report.py

# Check for software updates
git pull origin main
pip3 install --upgrade ccxt pandas numpy

# Restart strategy (if needed)
screen -S strategy -X quit
screen -S strategy
python3 scripts/live_trading.py
```

## 📊 Success Metrics

### **Month 1 Goals (Paper Trading)**
- ✅ Strategy runs without errors for 30 days
- ✅ Generates signals as expected
- ✅ Monitoring system works correctly
- ✅ Risk management triggers properly

### **Month 2 Goals (Micro Live)**
- ✅ Positive risk-adjusted returns vs BTC
- ✅ Sharpe ratio > 0.8
- ✅ Maximum drawdown < 15%
- ✅ No significant technical issues

### **Month 3+ Goals (Scaling)**
- ✅ Consistent outperformance vs BTC
- ✅ Sharpe ratio > 1.0
- ✅ Smooth scaling without issues
- ✅ Automated monitoring functioning

## ⚠️ Red Flags - Stop Trading If:

1. **Performance**: Underperforming BTC by >10% over 30 days
2. **Risk**: Drawdown exceeds 25%
3. **Technical**: Strategy errors or API failures
4. **Market**: Strategy becomes highly correlated with BTC (>0.8)
5. **Capital**: Total losses exceed 20% of initial capital

Remember: **Start small, monitor closely, and scale gradually!** 🚀