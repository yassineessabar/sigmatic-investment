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
import ccxt

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Setup logging
os.makedirs('logs', exist_ok=True)
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
    def __init__(self, log_file='logs/live_performance.json', initial_capital=10000):
        self.log_file = log_file
        self.initial_capital = initial_capital
        self.performance_data = self.load_performance_log()

        # Initialize exchange for BTC price tracking
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        })

    def load_performance_log(self):
        """Load existing performance data"""
        try:
            with open(self.log_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {
                'start_date': datetime.now().isoformat(),
                'initial_capital': self.initial_capital,
                'daily_performance': [],
                'positions': [],
                'metrics': {},
                'btc_start_price': None
            }

    def save_performance_log(self):
        """Save performance data"""
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        with open(self.log_file, 'w') as f:
            json.dump(self.performance_data, f, indent=2, default=str)

    def get_btc_price(self) -> float:
        """Get current BTC price"""
        try:
            ticker = self.exchange.fetch_ticker('BTC/USDT')
            return ticker['last']
        except Exception as e:
            logger.error(f"Error fetching BTC price: {e}")
            return None

    def calculate_btc_performance(self, current_btc_price: float) -> float:
        """Calculate BTC buy & hold performance"""
        if not self.performance_data.get('btc_start_price'):
            self.performance_data['btc_start_price'] = current_btc_price
            self.save_performance_log()
            return 0.0

        start_price = self.performance_data['btc_start_price']
        return (current_btc_price - start_price) / start_price

    def log_daily_performance(self, strategy_portfolio_value: float,
                             positions: Dict, notes: str = ""):
        """Log daily performance vs BTC"""
        today = datetime.now().date().isoformat()

        # Get current BTC price and calculate performance
        btc_price = self.get_btc_price()
        if btc_price is None:
            logger.error("Could not fetch BTC price, skipping performance log")
            return

        btc_performance = self.calculate_btc_performance(btc_price)
        strategy_performance = (strategy_portfolio_value - self.initial_capital) / self.initial_capital

        daily_data = {
            'date': today,
            'strategy_portfolio_value': strategy_portfolio_value,
            'strategy_performance': strategy_performance,
            'btc_price': btc_price,
            'btc_performance': btc_performance,
            'outperformance': strategy_performance - btc_performance,
            'positions': positions,
            'notes': notes,
            'timestamp': datetime.now().isoformat()
        }

        # Remove any existing entry for today and add new one
        self.performance_data['daily_performance'] = [
            d for d in self.performance_data['daily_performance']
            if d['date'] != today
        ]
        self.performance_data['daily_performance'].append(daily_data)
        self.save_performance_log()

        logger.info(f"Daily Performance: Strategy: {strategy_performance:.2%}, "
                   f"BTC: {btc_performance:.2%}, Outperformance: {strategy_performance-btc_performance:.2%}")

    def calculate_live_metrics(self) -> Dict:
        """Calculate live strategy metrics"""
        if len(self.performance_data['daily_performance']) < 2:
            return {}

        df = pd.DataFrame(self.performance_data['daily_performance'])
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df = df.sort_index()

        # Calculate daily returns
        df['strategy_daily_return'] = df['strategy_performance'].diff()
        df['btc_daily_return'] = df['btc_performance'].diff()

        # Remove first NaN row
        df = df.dropna()

        if len(df) == 0:
            return {}

        # Calculate metrics
        days_trading = len(df)
        total_strategy_return = df['strategy_performance'].iloc[-1]
        total_btc_return = df['btc_performance'].iloc[-1]

        # Annualized metrics
        if days_trading > 1:
            strategy_vol = df['strategy_daily_return'].std() * np.sqrt(365)
            strategy_sharpe = (total_strategy_return * 365 / days_trading) / strategy_vol if strategy_vol > 0 else 0

            # Rolling metrics (last 7 days if available)
            if len(df) >= 7:
                recent_df = df.tail(7)
                recent_outperformance = recent_df['outperformance'].mean()
            else:
                recent_outperformance = df['outperformance'].mean()
        else:
            strategy_vol = 0
            strategy_sharpe = 0
            recent_outperformance = 0

        # Drawdown calculation
        df['strategy_cumulative'] = (1 + df['strategy_daily_return']).cumprod()
        df['peak'] = df['strategy_cumulative'].cummax()
        df['drawdown'] = (df['strategy_cumulative'] - df['peak']) / df['peak']
        max_drawdown = df['drawdown'].min()

        metrics = {
            'days_trading': days_trading,
            'total_strategy_return': total_strategy_return,
            'total_btc_return': total_btc_return,
            'outperformance': total_strategy_return - total_btc_return,
            'recent_outperformance': recent_outperformance,
            'strategy_sharpe': strategy_sharpe,
            'strategy_volatility': strategy_vol,
            'max_drawdown': max_drawdown,
            'current_drawdown': df['drawdown'].iloc[-1],
            'win_rate': (df['strategy_daily_return'] > 0).mean(),
            'btc_correlation': df['strategy_daily_return'].corr(df['btc_daily_return']) if len(df) > 3 else 0,
            'current_portfolio_value': df['strategy_portfolio_value'].iloc[-1],
            'current_btc_price': df['btc_price'].iloc[-1]
        }

        self.performance_data['metrics'] = metrics
        return metrics

    def check_alerts(self, metrics: Dict) -> List[str]:
        """Check for performance alerts"""
        alerts = []

        if metrics.get('current_drawdown', 0) < -0.15:
            alerts.append("🚨 CRITICAL: Drawdown exceeds 15%")
        elif metrics.get('current_drawdown', 0) < -0.10:
            alerts.append("⚠️  WARNING: Drawdown exceeds 10%")

        if metrics.get('recent_outperformance', 0) < -0.05:
            alerts.append("⚠️  WARNING: Underperforming BTC by >5% recently")

        if metrics.get('btc_correlation', 0) > 0.8:
            alerts.append("⚠️  WARNING: High correlation with BTC - strategy not market neutral")

        if metrics.get('days_trading', 0) >= 30:
            if metrics.get('outperformance', 0) < -0.10:
                alerts.append("🚨 CRITICAL: Underperforming BTC by >10% over 30+ days")

        return alerts

    def generate_daily_report(self):
        """Generate daily performance report"""
        metrics = self.calculate_live_metrics()

        if not metrics:
            logger.info("Insufficient data for metrics calculation")
            return

        alerts = self.check_alerts(metrics)
        alert_section = "\n".join(alerts) if alerts else "✅ No alerts"

        report = f"""
========================================
LIVE STRATEGY PERFORMANCE REPORT
========================================
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Days Trading: {metrics['days_trading']}

💰 PORTFOLIO STATUS:
  Current Value: ${metrics['current_portfolio_value']:,.2f}
  Initial Capital: ${self.initial_capital:,.2f}

📈 PERFORMANCE:
  Strategy Return: {metrics['total_strategy_return']:+.2%}
  BTC Return: {metrics['total_btc_return']:+.2%}
  Outperformance: {metrics['outperformance']:+.2%}
  Recent 7-day Outperformance: {metrics.get('recent_outperformance', 0):+.2%}

⚡ RISK METRICS:
  Sharpe Ratio: {metrics['strategy_sharpe']:.2f}
  Volatility: {metrics['strategy_volatility']:.1%}
  Max Drawdown: {metrics['max_drawdown']:.1%}
  Current Drawdown: {metrics['current_drawdown']:.1%}
  Win Rate: {metrics['win_rate']:.1%}
  BTC Correlation: {metrics['btc_correlation']:.2f}

🚨 ALERTS:
{alert_section}

📊 MARKET DATA:
  Current BTC Price: ${metrics['current_btc_price']:,.2f}

Status: {'🟢 HEALTHY' if not alerts else '🔴 NEEDS ATTENTION' if any('CRITICAL' in a for a in alerts) else '🟡 MONITORING'}
========================================
"""
        print(report)

        # Save report to file
        os.makedirs('reports', exist_ok=True)
        report_file = f"reports/daily_report_{datetime.now().strftime('%Y%m%d')}.txt"
        with open(report_file, 'w') as f:
            f.write(report)

        return report, alerts

    def export_performance_csv(self):
        """Export performance data to CSV for analysis"""
        if not self.performance_data['daily_performance']:
            logger.info("No performance data to export")
            return

        df = pd.DataFrame(self.performance_data['daily_performance'])
        df['date'] = pd.to_datetime(df['date'])

        # Add some calculated columns
        df['strategy_pnl_usd'] = df['strategy_portfolio_value'] - self.initial_capital
        df['btc_pnl_usd'] = df['btc_performance'] * self.initial_capital
        df['outperformance_usd'] = df['strategy_pnl_usd'] - df['btc_pnl_usd']

        os.makedirs('exports', exist_ok=True)
        filename = f"exports/performance_data_{datetime.now().strftime('%Y%m%d')}.csv"
        df.to_csv(filename, index=False)
        logger.info(f"Performance data exported to {filename}")

        return filename

# Example usage functions
def simulate_trading_day(monitor: LivePerformanceMonitor, portfolio_value: float, positions: Dict, notes: str = ""):
    """Simulate logging a trading day"""
    monitor.log_daily_performance(portfolio_value, positions, notes)
    report, alerts = monitor.generate_daily_report()

    if alerts:
        logger.warning(f"ALERTS DETECTED: {alerts}")

    return report, alerts

def main():
    """Example usage of the monitor"""
    print("🚀 Starting Live Strategy Monitor...")

    # Initialize monitor
    monitor = LivePerformanceMonitor(initial_capital=10000)

    # Example: Log today's performance
    example_positions = {
        'BTCUSDT/AVAX': {'side': 'long_btc_short_avax', 'notional': 7500, 'pnl_pct': 0.02},
        'BTCUSDT/ETH': {'side': 'short_btc_long_eth', 'notional': 2500, 'pnl_pct': -0.005}
    }

    # Calculate portfolio value (example)
    portfolio_value = 10000 + (7500 * 0.02) + (2500 * -0.005)

    # Log performance
    report, alerts = simulate_trading_day(
        monitor,
        portfolio_value,
        example_positions,
        "Example trading day with 2 active pairs"
    )

    # Export data
    monitor.export_performance_csv()

    print("✅ Monitor update complete!")

if __name__ == "__main__":
    main()