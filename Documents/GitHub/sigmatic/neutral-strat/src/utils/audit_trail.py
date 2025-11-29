#!/usr/bin/env python3

"""
Strategy Audit Trail System
Tracks all strategy optimizations with version control and performance metrics
"""

import os
import json
import datetime
from pathlib import Path
import shutil
import hashlib
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class StrategyAuditTrail:
    """
    Comprehensive audit trail system for strategy optimization tracking
    """

    def __init__(self, base_path: str = "audit_trail"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)

        # Create directory structure
        self.versions_path = self.base_path / "versions"
        self.results_path = self.base_path / "results"
        self.configs_path = self.base_path / "configs"
        self.metadata_path = self.base_path / "metadata"

        for path in [self.versions_path, self.results_path, self.configs_path, self.metadata_path]:
            path.mkdir(exist_ok=True)

        self.audit_log_file = self.base_path / "audit_log.jsonl"
        self.performance_history = self.base_path / "performance_history.json"

        # Initialize performance history
        if not self.performance_history.exists():
            self._save_json(self.performance_history, {
                "strategies": [],
                "best_sharpe": {"value": -999, "version": None},
                "best_calmar": {"value": -999, "version": None},
                "lowest_drawdown": {"value": -1, "version": None}
            })

    def create_strategy_version(self,
                               strategy_file: str,
                               config_file: str,
                               results: Dict[str, Any],
                               description: str = "",
                               optimization_notes: str = "") -> str:
        """
        Create a new strategy version with full audit trail

        Args:
            strategy_file: Path to strategy implementation file
            config_file: Path to configuration file
            results: Backtest results dictionary
            description: Human description of changes
            optimization_notes: Technical optimization notes

        Returns:
            version_id: Unique version identifier
        """

        # Extract performance metrics
        portfolio_results = self._extract_portfolio_metrics(results)
        sharpe_ratio = portfolio_results.get('sharpe', 0)
        calmar_ratio = portfolio_results.get('calmar', 0)
        max_drawdown = portfolio_results.get('max_dd', 0)

        # Generate version ID based on timestamp and performance
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        sharpe_str = f"{sharpe_ratio:.2f}".replace('.', 'p').replace('-', 'n')
        version_id = f"v{timestamp}_sharpe_{sharpe_str}"

        # Create version directory
        version_dir = self.versions_path / version_id
        version_dir.mkdir(exist_ok=True)

        # Copy strategy file with versioned name
        strategy_src = Path(strategy_file)
        strategy_filename = f"{strategy_src.stem}_sharpe_{sharpe_str}{strategy_src.suffix}"
        strategy_dest = version_dir / strategy_filename
        shutil.copy2(strategy_src, strategy_dest)

        # Copy config file
        config_src = Path(config_file)
        config_dest = version_dir / f"config_{version_id}.yaml"
        shutil.copy2(config_src, config_dest)

        # Save results
        results_file = version_dir / f"results_{version_id}.json"
        self._save_json(results_file, {
            "version_id": version_id,
            "timestamp": timestamp,
            "performance": portfolio_results,
            "full_results": results,
            "files": {
                "strategy": str(strategy_dest),
                "config": str(config_dest)
            }
        })

        # Create metadata
        metadata = {
            "version_id": version_id,
            "timestamp": timestamp,
            "description": description,
            "optimization_notes": optimization_notes,
            "performance": {
                "sharpe_ratio": sharpe_ratio,
                "calmar_ratio": calmar_ratio,
                "max_drawdown": max_drawdown,
                "annual_return": portfolio_results.get('ann_return', 0),
                "volatility": portfolio_results.get('volatility', 0)
            },
            "files": {
                "strategy": strategy_filename,
                "config": f"config_{version_id}.yaml",
                "results": f"results_{version_id}.json"
            },
            "file_hashes": {
                "strategy": self._get_file_hash(strategy_dest),
                "config": self._get_file_hash(config_dest)
            }
        }

        metadata_file = version_dir / f"metadata_{version_id}.json"
        self._save_json(metadata_file, metadata)

        # Update audit log
        self._append_to_audit_log({
            "action": "create_version",
            "version_id": version_id,
            "timestamp": timestamp,
            "performance": metadata["performance"],
            "description": description
        })

        # Update performance history
        self._update_performance_history(version_id, metadata["performance"])

        # Save results to main results folder with version
        main_results_file = self.results_path / f"backtest_results_{version_id}.csv"
        if 'results_table' in results:
            results['results_table'].to_csv(main_results_file, index=False)

        logger.info(f"Created strategy version {version_id} with Sharpe {sharpe_ratio:.2f}")
        return version_id

    def compare_versions(self, version1: str, version2: str) -> Dict[str, Any]:
        """Compare two strategy versions"""

        v1_metadata = self._load_version_metadata(version1)
        v2_metadata = self._load_version_metadata(version2)

        if not v1_metadata or not v2_metadata:
            raise ValueError("One or both versions not found")

        comparison = {
            "version1": version1,
            "version2": version2,
            "performance_comparison": {},
            "improvement": {}
        }

        # Compare performance metrics
        v1_perf = v1_metadata["performance"]
        v2_perf = v2_metadata["performance"]

        for metric in ["sharpe_ratio", "calmar_ratio", "max_drawdown", "annual_return", "volatility"]:
            v1_val = v1_perf.get(metric, 0)
            v2_val = v2_perf.get(metric, 0)

            comparison["performance_comparison"][metric] = {
                "v1": v1_val,
                "v2": v2_val,
                "difference": v2_val - v1_val,
                "improvement_pct": ((v2_val - v1_val) / abs(v1_val) * 100) if v1_val != 0 else 0
            }

        # Overall assessment
        sharpe_improved = v2_perf["sharpe_ratio"] > v1_perf["sharpe_ratio"]
        dd_improved = v2_perf["max_drawdown"] > v1_perf["max_drawdown"]  # Less negative is better

        comparison["improvement"]["overall"] = sharpe_improved and dd_improved
        comparison["improvement"]["sharpe_better"] = sharpe_improved
        comparison["improvement"]["drawdown_better"] = dd_improved

        return comparison

    def get_version_history(self) -> List[Dict[str, Any]]:
        """Get complete version history sorted by timestamp"""

        versions = []
        for version_dir in self.versions_path.iterdir():
            if version_dir.is_dir():
                metadata_file = version_dir / f"metadata_{version_dir.name}.json"
                if metadata_file.exists():
                    metadata = self._load_json(metadata_file)
                    versions.append(metadata)

        # Sort by timestamp
        versions.sort(key=lambda x: x["timestamp"], reverse=True)
        return versions

    def get_best_strategies(self) -> Dict[str, Any]:
        """Get best performing strategies by different metrics"""

        performance_history = self._load_json(self.performance_history)
        return {
            "best_sharpe": performance_history.get("best_sharpe"),
            "best_calmar": performance_history.get("best_calmar"),
            "lowest_drawdown": performance_history.get("lowest_drawdown")
        }

    def restore_version(self, version_id: str, target_strategy_file: str, target_config_file: str):
        """Restore a specific version to current files"""

        version_dir = self.versions_path / version_id
        if not version_dir.exists():
            raise ValueError(f"Version {version_id} not found")

        metadata = self._load_version_metadata(version_id)
        if not metadata:
            raise ValueError(f"Metadata for version {version_id} not found")

        # Restore strategy file
        strategy_src = version_dir / metadata["files"]["strategy"]
        config_src = version_dir / metadata["files"]["config"]

        if strategy_src.exists():
            shutil.copy2(strategy_src, target_strategy_file)

        if config_src.exists():
            shutil.copy2(config_src, target_config_file)

        # Log restoration
        self._append_to_audit_log({
            "action": "restore_version",
            "version_id": version_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "target_files": {
                "strategy": target_strategy_file,
                "config": target_config_file
            }
        })

        logger.info(f"Restored version {version_id} to current files")

    def generate_performance_report(self) -> str:
        """Generate detailed performance report"""

        versions = self.get_version_history()
        best_strategies = self.get_best_strategies()

        report = []
        report.append("STRATEGY OPTIMIZATION AUDIT TRAIL")
        report.append("=" * 50)
        report.append(f"Total Versions: {len(versions)}")
        report.append("")

        # Best performers
        report.append("BEST PERFORMING STRATEGIES:")
        report.append("-" * 30)

        if best_strategies["best_sharpe"]["version"]:
            bs = best_strategies["best_sharpe"]
            report.append(f"Best Sharpe Ratio: {bs['value']:.3f} (Version: {bs['version']})")

        if best_strategies["best_calmar"]["version"]:
            bc = best_strategies["best_calmar"]
            report.append(f"Best Calmar Ratio: {bc['value']:.3f} (Version: {bc['version']})")

        if best_strategies["lowest_drawdown"]["version"]:
            bd = best_strategies["lowest_drawdown"]
            report.append(f"Lowest Drawdown: {bd['value']:.2%} (Version: {bd['version']})")

        report.append("")

        # Recent versions
        report.append("RECENT VERSIONS:")
        report.append("-" * 20)

        for i, version in enumerate(versions[:10]):  # Last 10 versions
            perf = version["performance"]
            report.append(f"{i+1}. {version['version_id']}")
            report.append(f"   Sharpe: {perf['sharpe_ratio']:.3f} | Calmar: {perf['calmar_ratio']:.3f} | DD: {perf['max_drawdown']:.2%}")
            report.append(f"   Description: {version['description'][:60]}...")
            report.append("")

        return "\n".join(report)

    def _extract_portfolio_metrics(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Extract key portfolio metrics from results"""

        # Try to find Equal-Weight Portfolio results first
        if 'results_table' in results and hasattr(results['results_table'], 'iterrows'):
            for _, row in results['results_table'].iterrows():
                if 'Equal-Weight Portfolio' in str(row.get('Pair', '')):
                    return {
                        'sharpe': float(row.get('Sharpe', 0)),
                        'calmar': float(row.get('Calmar', 0)),
                        'max_dd': float(row.get('Max DD', 0)),
                        'ann_return': float(row.get('Ann.Return', 0)),
                        'volatility': float(row.get('Vol', 0)),
                        'win_rate': float(row.get('Win Rate', 0)),
                        'final_perf': float(row.get('Final_Perf', 0))
                    }

        # Fallback to first available results
        if isinstance(results, dict):
            return {
                'sharpe': float(results.get('sharpe', 0)),
                'calmar': float(results.get('calmar', 0)),
                'max_dd': float(results.get('max_dd', 0)),
                'ann_return': float(results.get('ann_return', 0)),
                'volatility': float(results.get('ann_vol', 0)),
                'win_rate': 0,
                'final_perf': float(results.get('final_performance', 1))
            }

        return {
            'sharpe': 0, 'calmar': 0, 'max_dd': 0,
            'ann_return': 0, 'volatility': 0, 'win_rate': 0, 'final_perf': 1
        }

    def _update_performance_history(self, version_id: str, performance: Dict[str, float]):
        """Update best performance tracking"""

        history = self._load_json(self.performance_history)

        # Check if this is a new best
        if performance["sharpe_ratio"] > history["best_sharpe"]["value"]:
            history["best_sharpe"] = {"value": performance["sharpe_ratio"], "version": version_id}

        if performance["calmar_ratio"] > history["best_calmar"]["value"]:
            history["best_calmar"] = {"value": performance["calmar_ratio"], "version": version_id}

        if performance["max_drawdown"] > history["lowest_drawdown"]["value"]:
            history["lowest_drawdown"] = {"value": performance["max_drawdown"], "version": version_id}

        # Add to strategies list
        history["strategies"].append({
            "version_id": version_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "performance": performance
        })

        self._save_json(self.performance_history, history)

    def _load_version_metadata(self, version_id: str) -> Dict[str, Any]:
        """Load metadata for a specific version"""

        version_dir = self.versions_path / version_id
        metadata_file = version_dir / f"metadata_{version_id}.json"

        if metadata_file.exists():
            return self._load_json(metadata_file)
        return None

    def _append_to_audit_log(self, entry: Dict[str, Any]):
        """Append entry to audit log"""

        with open(self.audit_log_file, 'a') as f:
            f.write(json.dumps(entry) + '\n')

    def _save_json(self, file_path: Path, data: Dict[str, Any]):
        """Save JSON data to file"""

        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    def _load_json(self, file_path: Path) -> Dict[str, Any]:
        """Load JSON data from file"""

        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _get_file_hash(self, file_path: Path) -> str:
        """Generate hash for file integrity checking"""

        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()