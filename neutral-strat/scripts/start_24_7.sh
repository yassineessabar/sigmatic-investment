#!/bin/bash

# 24/7 Paper Trading Startup Script
# Runs the neutral strategy continuously with monitoring and auto-restart

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_DIR/logs"
CONFIG_FILE="config/two_week_config.yaml"
PYTHON_SCRIPT="scripts/live_trader.py"
PID_FILE="$LOG_DIR/trader.pid"
LOCK_FILE="$LOG_DIR/trader.lock"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1"
}

success() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] SUCCESS:${NC} $1"
}

warn() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARNING:${NC} $1"
}

# Create necessary directories
mkdir -p "$LOG_DIR"
mkdir -p "$PROJECT_DIR/reports"
mkdir -p "$PROJECT_DIR/exports"

# Check if already running
check_running() {
    if [[ -f "$PID_FILE" ]]; then
        local pid=$(cat "$PID_FILE")
        if ps -p "$pid" > /dev/null 2>&1; then
            return 0  # Running
        else
            rm -f "$PID_FILE"  # Stale PID file
            return 1  # Not running
        fi
    fi
    return 1  # Not running
}

# Stop existing instance
stop_trader() {
    if [[ -f "$PID_FILE" ]]; then
        local pid=$(cat "$PID_FILE")
        if ps -p "$pid" > /dev/null 2>&1; then
            log "Stopping existing trader (PID: $pid)..."
            kill -TERM "$pid"

            # Wait for graceful shutdown
            local count=0
            while ps -p "$pid" > /dev/null 2>&1 && [[ $count -lt 30 ]]; do
                sleep 1
                ((count++))
            done

            if ps -p "$pid" > /dev/null 2>&1; then
                warn "Graceful shutdown failed, forcing kill..."
                kill -KILL "$pid"
            fi

            rm -f "$PID_FILE"
            success "Trader stopped"
        else
            warn "PID file exists but process not running, cleaning up..."
            rm -f "$PID_FILE"
        fi
    else
        log "No trader running"
    fi
}

# Start trader
start_trader() {
    if check_running; then
        error "Trader is already running!"
        return 1
    fi

    log "Starting 24/7 paper trader..."

    # Check if config file exists
    if [[ ! -f "$PROJECT_DIR/$CONFIG_FILE" ]]; then
        error "Config file not found: $CONFIG_FILE"
        return 1
    fi

    # Check if Python script exists
    if [[ ! -f "$PROJECT_DIR/$PYTHON_SCRIPT" ]]; then
        error "Python script not found: $PYTHON_SCRIPT"
        return 1
    fi

    # Set up environment
    cd "$PROJECT_DIR"

    # Activate virtual environment if it exists
    if [[ -f "venv/bin/activate" ]]; then
        source venv/bin/activate
        log "Activated virtual environment"
    fi

    # Start the trader in background with logging
    nohup python3 "$PYTHON_SCRIPT" "$CONFIG_FILE" > "$LOG_DIR/trader_24_7.log" 2>&1 &
    local pid=$!

    # Save PID
    echo "$pid" > "$PID_FILE"

    # Wait a moment and check if it's still running
    sleep 2
    if ps -p "$pid" > /dev/null 2>&1; then
        success "Trader started successfully (PID: $pid)"
        log "Log file: $LOG_DIR/trader_24_7.log"
        log "Monitor with: tail -f $LOG_DIR/trader_24_7.log"
        return 0
    else
        error "Failed to start trader"
        rm -f "$PID_FILE"
        return 1
    fi
}

# Status check
check_status() {
    if check_running; then
        local pid=$(cat "$PID_FILE")
        success "Trader is running (PID: $pid)"

        # Show recent log entries
        if [[ -f "$LOG_DIR/trader_24_7.log" ]]; then
            log "Recent log entries:"
            tail -5 "$LOG_DIR/trader_24_7.log"
        fi

        # Show process info
        log "Process info:"
        ps -p "$pid" -o pid,ppid,cpu,mem,etime,cmd

        return 0
    else
        warn "Trader is not running"
        return 1
    fi
}

# Restart trader
restart_trader() {
    log "Restarting trader..."
    stop_trader
    sleep 2
    start_trader
}

# Monitor and auto-restart function
monitor_trader() {
    log "Starting 24/7 monitoring with auto-restart..."

    while true; do
        if ! check_running; then
            error "Trader not running, attempting restart..."
            start_trader

            if [[ $? -ne 0 ]]; then
                error "Failed to restart trader, waiting 60 seconds..."
                sleep 60
                continue
            fi
        fi

        # Check if log file is growing (activity check)
        if [[] -f "$LOG_DIR/trader_24_7.log" ]]; then
            local current_size=$(stat -c%s "$LOG_DIR/trader_24_7.log" 2>/dev/null || echo "0")
            sleep 300  # Wait 5 minutes
            local new_size=$(stat -c%s "$LOG_DIR/trader_24_7.log" 2>/dev/null || echo "0")

            if [[ "$current_size" == "$new_size" ]]; then
                warn "No activity detected in log file, checking if trader is stuck..."

                # Additional health check could go here
                # For now, just log the warning
                log "Trader appears inactive but process is running"
            fi
        fi

        # Health check interval
        sleep 60  # Check every minute
    done
}

# Health check
health_check() {
    log "Running health check..."

    local issues=0

    # Check if trader is running
    if ! check_running; then
        error "✗ Trader process not running"
        ((issues++))
    else
        success "✓ Trader process running"
    fi

    # Check log file
    if [[ -f "$LOG_DIR/trader_24_7.log" ]]; then
        local log_size=$(stat -f%z "$LOG_DIR/trader_24_7.log" 2>/dev/null || echo "0")
        if [[ "$log_size" -gt 0 ]]; then
            success "✓ Log file exists and has content ($log_size bytes)"
        else
            error "✗ Log file exists but is empty"
            ((issues++))
        fi
    else
        error "✗ Log file not found"
        ((issues++))
    fi

    # Check recent activity (last 5 minutes)
    if [[ -f "$LOG_DIR/trader_24_7.log" ]]; then
        local five_min_ago=$(date -v-5M '+%Y-%m-%d %H:%M')
        local recent_activity=$(grep "$five_min_ago" "$LOG_DIR/trader_24_7.log" | wc -l)
        if [[ "$recent_activity" -gt 0 ]]; then
            success "✓ Recent activity detected"
        else
            warn "⚠ No recent activity in last 5 minutes"
        fi
    fi

    # Check disk space
    local disk_usage=$(df "$PROJECT_DIR" | tail -1 | awk '{print $5}' | sed 's/%//')
    if [[ "$disk_usage" -lt 90 ]]; then
        success "✓ Disk space OK ($disk_usage% used)"
    else
        error "✗ Disk space critical ($disk_usage% used)"
        ((issues++))
    fi

    # Summary
    if [[ "$issues" -eq 0 ]]; then
        success "All health checks passed!"
        return 0
    else
        error "Found $issues issues"
        return 1
    fi
}

# Show logs
show_logs() {
    local lines=${1:-50}

    if [[ -f "$LOG_DIR/trader_24_7.log" ]]; then
        log "Showing last $lines lines of trader log:"
        echo "----------------------------------------"
        tail -"$lines" "$LOG_DIR/trader_24_7.log"
        echo "----------------------------------------"
    else
        warn "Log file not found: $LOG_DIR/trader_24_7.log"
    fi
}

# Follow logs
follow_logs() {
    if [[ -f "$LOG_DIR/trader_24_7.log" ]]; then
        log "Following trader log (Ctrl+C to stop):"
        tail -f "$LOG_DIR/trader_24_7.log"
    else
        warn "Log file not found: $LOG_DIR/trader_24_7.log"
    fi
}

# Usage
usage() {
    echo "24/7 Paper Trading Manager"
    echo ""
    echo "Usage: $0 {start|stop|restart|status|monitor|health|logs|follow|help}"
    echo ""
    echo "Commands:"
    echo "  start    - Start the paper trader"
    echo "  stop     - Stop the paper trader"
    echo "  restart  - Restart the paper trader"
    echo "  status   - Check trader status"
    echo "  monitor  - Start monitoring with auto-restart"
    echo "  health   - Run health check"
    echo "  logs     - Show recent log entries"
    echo "  follow   - Follow log output"
    echo "  help     - Show this help"
    echo ""
    echo "Examples:"
    echo "  $0 start          # Start trader"
    echo "  $0 monitor        # Start with monitoring"
    echo "  $0 logs 100       # Show last 100 log lines"
}

# Main script
main() {
    case "${1:-help}" in
        start)
            start_trader
            ;;
        stop)
            stop_trader
            ;;
        restart)
            restart_trader
            ;;
        status)
            check_status
            ;;
        monitor)
            monitor_trader
            ;;
        health)
            health_check
            ;;
        logs)
            show_logs "${2:-50}"
            ;;
        follow)
            follow_logs
            ;;
        help|--help|-h)
            usage
            ;;
        *)
            error "Unknown command: $1"
            usage
            exit 1
            ;;
    esac
}

# Run main function
main "$@"