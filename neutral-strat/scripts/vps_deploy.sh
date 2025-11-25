#!/bin/bash
# VPS Deployment Script for Sigmatic Strategy
# Run this on your VPS Forex server

set -e  # Exit on any error

echo "🚀 Sigmatic Strategy VPS Deployment Starting..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
REPO_URL="https://github.com/yassineessabar/sigmatic-investment.git"
PROJECT_DIR="/root/sigmatic-investment"
STRATEGY_DIR="$PROJECT_DIR/neutral-strat"

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    print_error "Please run as root (sudo bash vps_deploy.sh)"
    exit 1
fi

print_status "Starting VPS setup for crypto trading..."

# 1. System Updates
print_status "Updating system packages..."
apt update && apt upgrade -y

# 2. Install Required Packages
print_status "Installing required packages..."
apt install -y \
    python3 \
    python3-pip \
    git \
    screen \
    htop \
    curl \
    wget \
    nano \
    cron \
    supervisor

# 3. Install Python Dependencies
print_status "Installing Python packages..."
pip3 install --upgrade pip
pip3 install \
    pandas>=1.5.0 \
    numpy>=1.21.0 \
    ccxt>=3.0.0 \
    python-dateutil>=2.8.0 \
    PyYAML>=6.0 \
    matplotlib>=3.5.0 \
    seaborn>=0.11.0 \
    requests

# 4. Clone Repository
print_status "Setting up project repository..."
if [ -d "$PROJECT_DIR" ]; then
    print_warning "Repository already exists, pulling latest changes..."
    cd "$PROJECT_DIR"
    git pull origin main
else
    print_status "Cloning repository..."
    git clone "$REPO_URL" "$PROJECT_DIR"
fi

cd "$STRATEGY_DIR"

# 5. Create Directory Structure
print_status "Creating directory structure..."
mkdir -p {logs,reports,exports,backups,data/historical/daily}

# 6. Set Permissions
chmod +x scripts/*.py
chmod 755 "$STRATEGY_DIR"

# 7. Create Environment Configuration
print_status "Creating environment configuration..."
cat > .env << 'EOF'
# Binance API Configuration
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_secret_here
BINANCE_TESTNET=true

# Strategy Configuration
INITIAL_CAPITAL=10000
MAX_POSITION_SIZE=0.2
MAX_DRAWDOWN=0.25

# Monitoring Configuration
ALERT_EMAIL=your_email@example.com
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_telegram_chat_id

# Timezone
TZ=UTC
EOF

print_warning "Please edit .env file with your actual API credentials!"

# 8. Create Systemd Service for Auto-restart
print_status "Creating systemd service..."
cat > /etc/systemd/system/sigmatic-strategy.service << EOF
[Unit]
Description=Sigmatic Market Neutral Strategy
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=$STRATEGY_DIR
Environment=PYTHONPATH=$STRATEGY_DIR
ExecStart=/usr/bin/python3 scripts/live_monitor.py
Restart=always
RestartSec=10
StandardOutput=append:$STRATEGY_DIR/logs/systemd.log
StandardError=append:$STRATEGY_DIR/logs/systemd_error.log

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload

# 9. Create Cron Jobs for Monitoring
print_status "Setting up automated monitoring..."
cat > /tmp/cron_strategy << EOF
# Sigmatic Strategy Monitoring
# Daily performance report at 9 AM UTC
0 9 * * * cd $STRATEGY_DIR && /usr/bin/python3 scripts/live_monitor.py >> logs/cron.log 2>&1

# Hourly health check
0 * * * * cd $STRATEGY_DIR && /usr/bin/python3 -c "
import sys
sys.path.append('.')
from scripts.live_monitor import LivePerformanceMonitor
monitor = LivePerformanceMonitor()
metrics = monitor.calculate_live_metrics()
if metrics and metrics.get('current_drawdown', 0) < -0.15:
    print('ALERT: High drawdown detected')
" >> logs/health_check.log 2>&1

# Weekly backup
0 2 * * 0 cd $STRATEGY_DIR && tar -czf backups/backup_\$(date +\%Y\%m\%d).tar.gz logs/ exports/ reports/
EOF

crontab /tmp/cron_strategy
rm /tmp/cron_strategy

# 10. Create Monitoring Dashboard Script
print_status "Creating monitoring dashboard..."
cat > scripts/dashboard.py << 'EOF'
#!/usr/bin/env python3
"""
Simple terminal dashboard for strategy monitoring
"""
import os
import sys
import time
import json
from datetime import datetime

def clear_screen():
    os.system('clear' if os.name == 'posix' else 'cls')

def load_performance_data():
    try:
        with open('logs/live_performance.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

def display_dashboard():
    while True:
        clear_screen()
        print("="*60)
        print("🚀 SIGMATIC STRATEGY LIVE DASHBOARD")
        print("="*60)
        print(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print()

        data = load_performance_data()
        if not data or not data.get('daily_performance'):
            print("❌ No performance data available")
            print("   Run the strategy first to see live data")
        else:
            latest = data['daily_performance'][-1]
            metrics = data.get('metrics', {})

            print(f"📊 PERFORMANCE OVERVIEW:")
            print(f"   Portfolio Value: ${latest.get('strategy_portfolio_value', 0):,.2f}")
            print(f"   Strategy Return: {latest.get('strategy_performance', 0):+.2%}")
            print(f"   BTC Return: {latest.get('btc_performance', 0):+.2%}")
            print(f"   Outperformance: {latest.get('outperformance', 0):+.2%}")
            print()

            if metrics:
                print(f"📈 RISK METRICS:")
                print(f"   Sharpe Ratio: {metrics.get('strategy_sharpe', 0):.2f}")
                print(f"   Max Drawdown: {metrics.get('max_drawdown', 0):.1%}")
                print(f"   Current Drawdown: {metrics.get('current_drawdown', 0):.1%}")
                print(f"   Win Rate: {metrics.get('win_rate', 0):.1%}")
                print(f"   Days Trading: {metrics.get('days_trading', 0)}")
                print()

            # Status indicator
            drawdown = metrics.get('current_drawdown', 0) if metrics else 0
            if drawdown < -0.15:
                status = "🔴 CRITICAL"
            elif drawdown < -0.10:
                status = "🟡 WARNING"
            else:
                status = "🟢 HEALTHY"

            print(f"🎯 STRATEGY STATUS: {status}")

        print()
        print("="*60)
        print("Press Ctrl+C to exit dashboard")

        try:
            time.sleep(5)  # Refresh every 5 seconds
        except KeyboardInterrupt:
            print("\nDashboard closed.")
            break

if __name__ == "__main__":
    display_dashboard()
EOF

chmod +x scripts/dashboard.py

# 11. Create Quick Start Script
cat > start_strategy.sh << 'EOF'
#!/bin/bash
# Quick start script for strategy

echo "🚀 Starting Sigmatic Strategy..."

# Check if .env is configured
if grep -q "your_api_key_here" .env; then
    echo "❌ Please configure your API keys in .env file first!"
    echo "   Edit .env and replace your_api_key_here with actual credentials"
    exit 1
fi

# Start in screen session
screen -dmS strategy python3 scripts/live_monitor.py

echo "✅ Strategy started in screen session 'strategy'"
echo "   View with: screen -r strategy"
echo "   View dashboard: python3 scripts/dashboard.py"
echo "   View logs: tail -f logs/live_monitor.log"
EOF

chmod +x start_strategy.sh

# 12. Final Setup
print_status "Final setup steps..."

# Create initial log files
touch logs/live_monitor.log
touch logs/health_check.log
touch logs/cron.log

# Set timezone to UTC
timedatectl set-timezone UTC

# 13. Print Setup Summary
print_status "VPS deployment completed successfully! 🎉"
echo
echo "📋 SETUP SUMMARY:"
echo "=================="
echo "✅ System packages installed"
echo "✅ Python dependencies installed"
echo "✅ Repository cloned to: $STRATEGY_DIR"
echo "✅ Directory structure created"
echo "✅ Systemd service created (not started)"
echo "✅ Cron jobs configured"
echo "✅ Monitoring dashboard created"
echo
echo "🔧 NEXT STEPS:"
echo "=============="
echo "1. Edit .env file with your Binance API credentials:"
echo "   nano $STRATEGY_DIR/.env"
echo
echo "2. Test the strategy in paper trading mode:"
echo "   cd $STRATEGY_DIR"
echo "   python3 scripts/live_monitor.py"
echo
echo "3. Start the strategy service:"
echo "   ./start_strategy.sh"
echo
echo "4. Monitor with dashboard:"
echo "   python3 scripts/dashboard.py"
echo
echo "5. View logs:"
echo "   tail -f logs/live_monitor.log"
echo
echo "⚠️  IMPORTANT REMINDERS:"
echo "======================="
echo "- Start with TESTNET mode (BINANCE_TESTNET=true in .env)"
echo "- Use small amounts for initial real trading"
echo "- Monitor daily performance vs BTC"
echo "- Set stop-loss at 25% drawdown"
echo
echo "🔗 USEFUL COMMANDS:"
echo "=================="
echo "  View strategy status: screen -r strategy"
echo "  Start dashboard: python3 scripts/dashboard.py"
echo "  View performance: cat logs/live_performance.json | jq"
echo "  Generate report: python3 scripts/live_monitor.py"
echo "  Restart service: systemctl restart sigmatic-strategy"
echo
print_status "Setup complete! Happy trading! 🚀"