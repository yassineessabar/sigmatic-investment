# Strategy Audit Trail System

## Overview
Complete audit trail system for tracking strategy optimizations with version control and performance metrics. Each strategy version is automatically named with its Sharpe ratio and stored with full metadata.

## Features
- **Automatic Version Creation**: Creates versions with Sharpe-based naming (e.g., `v20251129_222316_sharpe_4p10`)
- **Performance Tracking**: Tracks best Sharpe ratio, Calmar ratio, and lowest drawdown
- **File Versioning**: Saves strategy files, configs, and results with integrity checking
- **Version Comparison**: Compare performance between any two versions
- **Version Restoration**: Restore any previous version to current files

## Quick Start

### 1. Run Backtest with Audit Trail
```bash
# Basic backtest with automatic versioning
python3 scripts/run_relative_momentum_backtest.py \
    --start-date 2024-01-01 \
    --end-date 2024-11-29 \
    --description "Current optimized strategy baseline" \
    --optimization-notes "Multi-timeframe confirmation with volatility positioning"

# Skip audit trail creation
python3 scripts/run_relative_momentum_backtest.py \
    --start-date 2024-01-01 \
    --end-date 2024-11-29 \
    --no-audit
```

### 2. Manage Versions
```bash
# List all versions (sorted by creation date)
python3 scripts/manage_audit_trail.py list

# Show best performing strategies
python3 scripts/manage_audit_trail.py best

# Generate detailed performance report
python3 scripts/manage_audit_trail.py report
```

### 3. Compare Versions
```bash
# Compare two strategy versions
python3 scripts/manage_audit_trail.py compare \
    --version1 v20251129_222429_sharpe_3p17 \
    --version2 v20251129_222316_sharpe_4p10
```

### 4. Restore Previous Version
```bash
# Restore a specific version to current files
python3 scripts/manage_audit_trail.py restore \
    --version v20251129_222316_sharpe_4p10 \
    --strategy-file src/strategies/relative_momentum.py \
    --config-file config/unified_trading_config.yaml
```

## File Structure
```
audit_trail/
├── versions/                    # Strategy versions
│   └── v20251129_222316_sharpe_4p10/
│       ├── relative_momentum_sharpe_4p10.py
│       ├── config_v20251129_222316_sharpe_4p10.yaml
│       ├── results_v20251129_222316_sharpe_4p10.json
│       └── metadata_v20251129_222316_sharpe_4p10.json
├── results/                     # Backtest results CSV files
├── configs/                     # Configuration snapshots
├── metadata/                    # Version metadata
├── audit_log.jsonl             # Action log
└── performance_history.json    # Best performance tracking
```

## Version Naming Convention
- Format: `v{YYYYMMDD_HHMMSS}_sharpe_{sharpe_ratio}`
- Example: `v20251129_222316_sharpe_4p10` = Sharpe 4.10 created on 2025-11-29 at 22:23:16
- Decimal points replaced with 'p', negative signs with 'n'

## Performance Tracking
The system automatically tracks:
- **Best Sharpe Ratio**: Highest risk-adjusted return
- **Best Calmar Ratio**: Best return/max drawdown ratio
- **Lowest Drawdown**: Least risky version

## Typical Optimization Workflow

1. **Create Baseline Version**:
   ```bash
   python3 scripts/run_relative_momentum_backtest.py \
       --description "Baseline strategy" \
       --optimization-notes "Starting point for optimization"
   ```

2. **Make Strategy Changes** (edit `src/strategies/relative_momentum.py`)

3. **Test New Version**:
   ```bash
   python3 scripts/run_relative_momentum_backtest.py \
       --description "Enhanced momentum signals" \
       --optimization-notes "Added RSI filter and volatility scaling"
   ```

4. **Compare Performance**:
   ```bash
   python3 scripts/manage_audit_trail.py compare \
       --version1 {baseline_version} \
       --version2 {new_version}
   ```

5. **Keep Best Version or Restore Previous**:
   ```bash
   # If new version is worse, restore previous
   python3 scripts/manage_audit_trail.py restore --version {best_version}
   ```

## Example Output

### Version List
```
📚 STRATEGY VERSION HISTORY
================================================================================

1. v20251129_222316_sharpe_4p10
   Created: 20251129_222316
   Sharpe: 4.102 | Calmar: 8.764 | DD: -8.47%
   Description: Current optimized strategy baseline
   Notes: Multi-timeframe confirmation with volatility positioning and RSI filtering

2. v20251129_222429_sharpe_3p17
   Created: 20251129_222429
   Sharpe: 3.173 | Calmar: 6.803 | DD: -9.78%
   Description: Extended 2023 validation run
   Notes: Testing on different year for validation of strategy robustness
```

### Version Comparison
```
⚖️  COMPARING VERSIONS: v20251129_222429_sharpe_3p17 vs v20251129_222316_sharpe_4p10
================================================================================
Metric               Version 1          Version 2          Change          Improvement %
-------------------------------------------------------------------------------------
sharpe_ratio                      3.173              4.102        +0.929        +29.3%
calmar_ratio                      6.803              8.764        +1.961        +28.8%
max_drawdown                     -9.78%             -8.47%        +1.31%        +13.4%
annual_return                    66.54%             74.25%        +7.70%        +11.6%
volatility                       20.97%             18.10%        -2.87%        -13.7%
-------------------------------------------------------------------------------------
✅ Version 2 is better overall (Sharpe AND Drawdown improved)
```

This system ensures you never lose a good strategy version and can systematically track your optimization progress!