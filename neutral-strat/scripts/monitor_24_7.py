#!/usr/bin/env python3
"""
24/7 Trading Monitor
Comprehensive monitoring for continuous paper trading operation
"""

import os
import sys
import time
import json
import psutil
import logging
import smtplib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import subprocess

# Add project paths
sys.path.append(str(Path(__file__).parent.parent))

class TradingMonitor:
    def __init__(self, config_path="config/monitoring.json"):
        self.config = self.load_config(config_path)
        self.setup_logging()
        self.project_dir = Path(__file__).parent.parent

        # Monitoring state
        self.last_check = None
        self.alerts_sent = {}
        self.health_history = []

    def load_config(self, config_path: str) -> Dict:
        """Load monitoring configuration"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            # Default configuration
            return {
                "trader": {
                    "script_path": "scripts/live_trader.py",
                    "pid_file": "logs/trader.pid",
                    "log_file": "logs/trader_24_7.log",
                    "max_restart_attempts": 5,
                    "restart_delay": 60
                },
                "monitoring": {
                    "check_interval": 60,
                    "health_check_interval": 300,
                    "log_activity_threshold": 300,
                    "max_memory_mb": 1024,
                    "max_cpu_percent": 80
                },
                "alerts": {
                    "enabled": True,
                    "email": {
                        "enabled": False,
                        "smtp_server": "smtp.gmail.com",
                        "smtp_port": 587,
                        "username": "",
                        "password": "",
                        "to_email": ""
                    },
                    "webhook": {
                        "enabled": False,
                        "url": ""
                    }
                }
            }

    def setup_logging(self):
        """Setup logging for monitor"""
        log_dir = self.project_dir / "logs"
        log_dir.mkdir(exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / "monitor_24_7.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def check_trader_process(self) -> Dict:
        """Check if trader process is running and healthy"""
        result = {
            "running": False,
            "pid": None,
            "cpu_percent": 0,
            "memory_mb": 0,
            "uptime": None,
            "issues": []
        }

        pid_file = self.project_dir / self.config["trader"]["pid_file"]

        try:
            if pid_file.exists():
                with open(pid_file, 'r') as f:
                    pid = int(f.read().strip())

                if psutil.pid_exists(pid):
                    process = psutil.Process(pid)
                    result.update({
                        "running": True,
                        "pid": pid,
                        "cpu_percent": process.cpu_percent(),
                        "memory_mb": process.memory_info().rss / 1024 / 1024,
                        "uptime": datetime.now() - datetime.fromtimestamp(process.create_time())
                    })

                    # Check resource usage
                    if result["memory_mb"] > self.config["monitoring"]["max_memory_mb"]:
                        result["issues"].append(f"High memory usage: {result['memory_mb']:.1f}MB")

                    if result["cpu_percent"] > self.config["monitoring"]["max_cpu_percent"]:
                        result["issues"].append(f"High CPU usage: {result['cpu_percent']:.1f}%")
                else:
                    result["issues"].append("PID file exists but process not running")
                    pid_file.unlink()  # Remove stale PID file
            else:
                result["issues"].append("PID file not found")

        except Exception as e:
            result["issues"].append(f"Error checking process: {str(e)}")

        return result

    def check_log_activity(self) -> Dict:
        """Check log file for recent activity"""
        result = {
            "active": False,
            "last_entry": None,
            "file_size": 0,
            "issues": []
        }

        log_file = self.project_dir / self.config["trader"]["log_file"]

        try:
            if log_file.exists():
                stat = log_file.stat()
                result["file_size"] = stat.st_size

                # Check modification time
                last_modified = datetime.fromtimestamp(stat.st_mtime)
                time_since_modified = datetime.now() - last_modified

                threshold = timedelta(seconds=self.config["monitoring"]["log_activity_threshold"])

                if time_since_modified < threshold:
                    result["active"] = True
                    result["last_entry"] = last_modified
                else:
                    result["issues"].append(f"No log activity for {time_since_modified}")

                # Check if file is growing
                if hasattr(self, 'last_log_size'):
                    if result["file_size"] == self.last_log_size:
                        result["issues"].append("Log file not growing")

                self.last_log_size = result["file_size"]

            else:
                result["issues"].append("Log file not found")

        except Exception as e:
            result["issues"].append(f"Error checking log: {str(e)}")

        return result

    def check_system_resources(self) -> Dict:
        """Check system resource usage"""
        result = {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent,
            "network_active": True,
            "issues": []
        }

        # Check thresholds
        if result["cpu_percent"] > 90:
            result["issues"].append(f"High system CPU usage: {result['cpu_percent']:.1f}%")

        if result["memory_percent"] > 90:
            result["issues"].append(f"High system memory usage: {result['memory_percent']:.1f}%")

        if result["disk_percent"] > 85:
            result["issues"].append(f"High disk usage: {result['disk_percent']:.1f}%")

        # Test network connectivity (ping Binance API)
        try:
            result_ping = subprocess.run(
                ["ping", "-c", "1", "api.binance.com"],
                capture_output=True,
                timeout=10
            )
            if result_ping.returncode != 0:
                result["network_active"] = False
                result["issues"].append("Cannot reach Binance API")
        except Exception:
            result["network_active"] = False
            result["issues"].append("Network connectivity test failed")

        return result

    def restart_trader(self) -> bool:
        """Restart the trader process"""
        try:
            self.logger.info("Attempting to restart trader...")

            # Stop existing process
            self.stop_trader()

            # Wait a moment
            time.sleep(5)

            # Start new process
            start_script = self.project_dir / "scripts" / "start_24_7.sh"
            if start_script.exists():
                result = subprocess.run([str(start_script), "start"], capture_output=True)
                if result.returncode == 0:
                    self.logger.info("Trader restarted successfully")
                    return True
                else:
                    self.logger.error(f"Failed to restart trader: {result.stderr.decode()}")
            else:
                self.logger.error("Start script not found")

        except Exception as e:
            self.logger.error(f"Error restarting trader: {str(e)}")

        return False

    def stop_trader(self):
        """Stop the trader process"""
        pid_file = self.project_dir / self.config["trader"]["pid_file"]

        try:
            if pid_file.exists():
                with open(pid_file, 'r') as f:
                    pid = int(f.read().strip())

                if psutil.pid_exists(pid):
                    process = psutil.Process(pid)
                    process.terminate()

                    # Wait for graceful shutdown
                    try:
                        process.wait(timeout=30)
                    except psutil.TimeoutExpired:
                        process.kill()

                    self.logger.info(f"Stopped trader process (PID: {pid})")

                pid_file.unlink()

        except Exception as e:
            self.logger.error(f"Error stopping trader: {str(e)}")

    def send_alert(self, subject: str, message: str):
        """Send alert notification"""
        if not self.config["alerts"]["enabled"]:
            return

        # Rate limiting - don't spam alerts
        alert_key = f"{subject}_{datetime.now().hour}"
        if alert_key in self.alerts_sent:
            return

        self.alerts_sent[alert_key] = datetime.now()

        # Clean old alerts (older than 24 hours)
        cutoff = datetime.now() - timedelta(hours=24)
        self.alerts_sent = {
            k: v for k, v in self.alerts_sent.items()
            if v > cutoff
        }

        # Email alert
        if self.config["alerts"]["email"]["enabled"]:
            self._send_email_alert(subject, message)

        # Webhook alert
        if self.config["alerts"]["webhook"]["enabled"]:
            self._send_webhook_alert(subject, message)

        # Log alert
        self.logger.warning(f"ALERT: {subject} - {message}")

    def _send_email_alert(self, subject: str, message: str):
        """Send email alert"""
        try:
            email_config = self.config["alerts"]["email"]

            msg = f"Subject: [Trading Alert] {subject}\n\n{message}"

            with smtplib.SMTP(email_config["smtp_server"], email_config["smtp_port"]) as server:
                server.starttls()
                server.login(email_config["username"], email_config["password"])
                server.sendmail(
                    email_config["username"],
                    email_config["to_email"],
                    msg
                )

        except Exception as e:
            self.logger.error(f"Failed to send email alert: {str(e)}")

    def _send_webhook_alert(self, subject: str, message: str):
        """Send webhook alert (e.g., Discord, Slack)"""
        try:
            import requests

            webhook_config = self.config["alerts"]["webhook"]
            payload = {
                "text": f"🚨 **{subject}**\n{message}",
                "timestamp": datetime.now().isoformat()
            }

            response = requests.post(webhook_config["url"], json=payload, timeout=10)
            response.raise_for_status()

        except Exception as e:
            self.logger.error(f"Failed to send webhook alert: {str(e)}")

    def run_health_check(self) -> Dict:
        """Run comprehensive health check"""
        health_report = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "healthy",
            "process": self.check_trader_process(),
            "logs": self.check_log_activity(),
            "system": self.check_system_resources(),
            "issues": []
        }

        # Collect all issues
        all_issues = []
        all_issues.extend(health_report["process"]["issues"])
        all_issues.extend(health_report["logs"]["issues"])
        all_issues.extend(health_report["system"]["issues"])

        health_report["issues"] = all_issues

        # Determine overall status
        if not health_report["process"]["running"]:
            health_report["overall_status"] = "critical"
        elif all_issues:
            health_report["overall_status"] = "warning"

        # Store in history
        self.health_history.append(health_report)
        if len(self.health_history) > 1000:  # Keep last 1000 checks
            self.health_history = self.health_history[-1000:]

        return health_report

    def handle_critical_issues(self, health_report: Dict):
        """Handle critical issues automatically"""
        if not health_report["process"]["running"]:
            self.send_alert(
                "Trader Process Down",
                f"Trading process is not running. Attempting automatic restart."
            )

            if self.restart_trader():
                self.send_alert(
                    "Trader Restarted",
                    "Trading process has been successfully restarted."
                )
            else:
                self.send_alert(
                    "Restart Failed",
                    "Failed to restart trading process. Manual intervention required."
                )

        # Handle other critical issues
        critical_issues = [
            issue for issue in health_report["issues"]
            if any(keyword in issue.lower() for keyword in ["critical", "cannot reach", "network"])
        ]

        if critical_issues:
            self.send_alert(
                "Critical System Issues",
                f"Critical issues detected:\n" + "\n".join(critical_issues)
            )

    def run_monitoring_loop(self):
        """Run continuous monitoring loop"""
        self.logger.info("Starting 24/7 trading monitor...")

        check_interval = self.config["monitoring"]["check_interval"]
        health_interval = self.config["monitoring"]["health_check_interval"]
        last_health_check = datetime.now()

        try:
            while True:
                current_time = datetime.now()

                # Run quick status check
                process_status = self.check_trader_process()
                if not process_status["running"]:
                    self.logger.error("Trader not running - attempting restart")
                    self.handle_critical_issues({"process": process_status, "issues": process_status["issues"]})

                # Run full health check periodically
                if (current_time - last_health_check).seconds >= health_interval:
                    health_report = self.run_health_check()

                    if health_report["overall_status"] == "critical":
                        self.handle_critical_issues(health_report)
                    elif health_report["issues"]:
                        self.logger.warning(f"Health issues detected: {health_report['issues']}")
                    else:
                        self.logger.info("Health check passed - all systems normal")

                    last_health_check = current_time

                # Sleep until next check
                time.sleep(check_interval)

        except KeyboardInterrupt:
            self.logger.info("Monitoring stopped by user")
        except Exception as e:
            self.logger.error(f"Monitor error: {str(e)}")
            self.send_alert("Monitor Error", f"Monitoring system error: {str(e)}")


def main():
    """Main entry point"""
    monitor = TradingMonitor()

    if len(sys.argv) > 1 and sys.argv[1] == "check":
        # Single health check
        health_report = monitor.run_health_check()
        print(json.dumps(health_report, indent=2))
        sys.exit(0 if health_report["overall_status"] == "healthy" else 1)
    else:
        # Continuous monitoring
        monitor.run_monitoring_loop()


if __name__ == "__main__":
    main()