#!/usr/bin/env python3

"""
Audit Trail Management Script
Manage strategy versions, compare performance, and restore previous versions
"""

import sys
import os
import argparse
from pathlib import Path

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.audit_trail import StrategyAuditTrail


def list_versions(audit_trail):
    """List all strategy versions"""
    print("\n📚 STRATEGY VERSION HISTORY")
    print("=" * 80)

    versions = audit_trail.get_version_history()

    if not versions:
        print("No strategy versions found.")
        return

    for i, version in enumerate(versions):
        perf = version['performance']
        print(f"\n{i+1}. {version['version_id']}")
        print(f"   Created: {version['timestamp']}")
        print(f"   Sharpe: {perf['sharpe_ratio']:.3f} | Calmar: {perf['calmar_ratio']:.3f} | DD: {perf['max_drawdown']:.2%}")
        print(f"   Description: {version['description']}")
        if version['optimization_notes']:
            print(f"   Notes: {version['optimization_notes']}")


def show_best_strategies(audit_trail):
    """Show best performing strategies"""
    print("\n🏆 BEST PERFORMING STRATEGIES")
    print("=" * 50)

    best = audit_trail.get_best_strategies()

    if best['best_sharpe']['version']:
        bs = best['best_sharpe']
        print(f"Best Sharpe Ratio: {bs['value']:.3f} ({bs['version']})")

    if best['best_calmar']['version']:
        bc = best['best_calmar']
        print(f"Best Calmar Ratio: {bc['value']:.3f} ({bc['version']})")

    if best['lowest_drawdown']['version']:
        bd = best['lowest_drawdown']
        print(f"Lowest Drawdown: {bd['value']:.2%} ({bd['version']})")


def compare_versions(audit_trail, version1, version2):
    """Compare two strategy versions"""
    print(f"\n⚖️  COMPARING VERSIONS: {version1} vs {version2}")
    print("=" * 80)

    try:
        comparison = audit_trail.compare_versions(version1, version2)

        print(f"{'Metric':<20} {version1:<20} {version2:<20} {'Change':<15} {'Improvement %'}")
        print("-" * 85)

        perf_comp = comparison['performance_comparison']
        for metric, data in perf_comp.items():
            v1_val = data['v1']
            v2_val = data['v2']
            diff = data['difference']
            imp_pct = data['improvement_pct']

            if 'drawdown' in metric:
                print(f"{metric:<20} {v1_val:>18.2%} {v2_val:>18.2%} {diff:>+13.2%} {imp_pct:>+12.1f}%")
            elif 'ratio' in metric:
                print(f"{metric:<20} {v1_val:>18.3f} {v2_val:>18.3f} {diff:>+13.3f} {imp_pct:>+12.1f}%")
            else:
                print(f"{metric:<20} {v1_val:>18.2%} {v2_val:>18.2%} {diff:>+13.2%} {imp_pct:>+12.1f}%")

        print("\n" + "-" * 85)
        imp = comparison['improvement']
        if imp['overall']:
            print("✅ Version 2 is better overall (Sharpe AND Drawdown improved)")
        elif imp['sharpe_better']:
            print("⚠️ Version 2 has better Sharpe but similar/worse drawdown")
        elif imp['drawdown_better']:
            print("⚠️ Version 2 has better drawdown but similar/worse Sharpe")
        else:
            print("❌ Version 2 shows minimal improvement")

    except Exception as e:
        print(f"Error comparing versions: {e}")


def restore_version(audit_trail, version_id, strategy_file, config_file):
    """Restore a specific version"""
    print(f"\n🔄 RESTORING VERSION: {version_id}")
    print("=" * 50)

    try:
        audit_trail.restore_version(version_id, strategy_file, config_file)
        print(f"✅ Successfully restored version {version_id}")
        print(f"   Strategy file: {strategy_file}")
        print(f"   Config file: {config_file}")
    except Exception as e:
        print(f"❌ Error restoring version: {e}")


def generate_report(audit_trail):
    """Generate performance report"""
    report = audit_trail.generate_performance_report()
    print(report)


def main():
    parser = argparse.ArgumentParser(description='Manage strategy audit trail')
    parser.add_argument('action', choices=['list', 'best', 'compare', 'restore', 'report'],
                       help='Action to perform')
    parser.add_argument('--version1', type=str, help='First version ID for comparison')
    parser.add_argument('--version2', type=str, help='Second version ID for comparison')
    parser.add_argument('--version', type=str, help='Version ID to restore')
    parser.add_argument('--strategy-file', type=str,
                       default='src/strategies/relative_momentum.py',
                       help='Strategy file path for restoration')
    parser.add_argument('--config-file', type=str,
                       default='config/unified_trading_config.yaml',
                       help='Config file path for restoration')
    parser.add_argument('--audit-path', type=str,
                       default='audit_trail',
                       help='Path to audit trail directory')

    args = parser.parse_args()

    # Initialize audit trail
    audit_trail = StrategyAuditTrail(args.audit_path)

    if args.action == 'list':
        list_versions(audit_trail)

    elif args.action == 'best':
        show_best_strategies(audit_trail)

    elif args.action == 'compare':
        if not args.version1 or not args.version2:
            print("Error: Both --version1 and --version2 are required for comparison")
            sys.exit(1)
        compare_versions(audit_trail, args.version1, args.version2)

    elif args.action == 'restore':
        if not args.version:
            print("Error: --version is required for restoration")
            sys.exit(1)
        restore_version(audit_trail, args.version, args.strategy_file, args.config_file)

    elif args.action == 'report':
        generate_report(audit_trail)


if __name__ == "__main__":
    main()