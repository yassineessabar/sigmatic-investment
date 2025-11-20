import os
import yaml
import json
from typing import Dict, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ConfigManager:
    def __init__(self, base_path: Optional[str] = None):
        if base_path is None:
            self.base_path = Path(__file__).parent.parent.parent
        else:
            self.base_path = Path(base_path)

    def load_config(self, config_path: str) -> Dict[str, Any]:
        full_path = self.base_path / config_path

        if not full_path.exists():
            raise FileNotFoundError(f"Config file not found: {full_path}")

        try:
            if full_path.suffix.lower() in ['.yaml', '.yml']:
                return self._load_yaml(full_path)
            elif full_path.suffix.lower() == '.json':
                return self._load_json(full_path)
            else:
                raise ValueError(f"Unsupported config file format: {full_path.suffix}")

        except Exception as e:
            logger.error(f"Error loading config from {full_path}: {e}")
            raise

    def _load_yaml(self, file_path: Path) -> Dict[str, Any]:
        with open(file_path, 'r') as f:
            content = f.read()

        content = self._substitute_env_vars(content)

        return yaml.safe_load(content)

    def _load_json(self, file_path: Path) -> Dict[str, Any]:
        with open(file_path, 'r') as f:
            content = f.read()

        content = self._substitute_env_vars(content)

        return json.loads(content)

    def _substitute_env_vars(self, content: str) -> str:
        import re

        def replace_env_var(match):
            var_name = match.group(1)
            default_value = match.group(2) if match.group(2) else ""
            return os.getenv(var_name, default_value)

        content = re.sub(r'\$\{([^}:]+)(?::([^}]*))?\}', replace_env_var, content)

        return content

    def save_config(self, config: Dict[str, Any], config_path: str):
        full_path = self.base_path / config_path
        full_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            if full_path.suffix.lower() in ['.yaml', '.yml']:
                self._save_yaml(config, full_path)
            elif full_path.suffix.lower() == '.json':
                self._save_json(config, full_path)
            else:
                raise ValueError(f"Unsupported config file format: {full_path.suffix}")

            logger.info(f"Config saved to {full_path}")

        except Exception as e:
            logger.error(f"Error saving config to {full_path}: {e}")
            raise

    def _save_yaml(self, config: Dict[str, Any], file_path: Path):
        with open(file_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    def _save_json(self, config: Dict[str, Any], file_path: Path):
        with open(file_path, 'w') as f:
            json.dump(config, f, indent=2, sort_keys=False)

    def update_config(self, config_path: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        config = self.load_config(config_path)
        config = self._deep_update(config, updates)
        self.save_config(config, config_path)
        return config

    def _deep_update(self, base_dict: Dict, update_dict: Dict) -> Dict:
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                base_dict[key] = self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
        return base_dict

    def get_config_value(self, config_path: str, key_path: str, default: Any = None) -> Any:
        config = self.load_config(config_path)
        keys = key_path.split('.')

        current = config
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default

        return current

    def set_config_value(self, config_path: str, key_path: str, value: Any):
        config = self.load_config(config_path)
        keys = key_path.split('.')

        current = config
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        current[keys[-1]] = value
        self.save_config(config, config_path)

    def validate_config(self, config: Dict[str, Any]) -> bool:
        required_sections = ['pairs', 'risk', 'execution']

        for section in required_sections:
            if section not in config:
                logger.error(f"Missing required config section: {section}")
                return False

        if 'pairs' in config:
            for i, pair in enumerate(config['pairs']):
                required_pair_fields = ['base', 'hedge', 'lookback', 'entry_z', 'exit_z']
                for field in required_pair_fields:
                    if field not in pair:
                        logger.error(f"Missing required field '{field}' in pair {i}")
                        return False

        if 'execution' in config and 'mode' in config['execution']:
            if config['execution']['mode'] not in ['paper', 'live']:
                logger.error("Execution mode must be either 'paper' or 'live'")
                return False

        return True


def create_default_config() -> Dict[str, Any]:
    return {
        'pairs': [
            {
                'base': 'BTCUSDT',
                'hedge': 'ETHUSDT',
                'lookback': 100,
                'entry_z': 2.0,
                'exit_z': 0.5,
                'max_notional': 1000
            }
        ],
        'risk': {
            'max_daily_dd': 0.05,
            'max_total_dd': 0.10,
            'leverage_limit': 2.0,
            'max_position_size': 0.1,
            'enabled': True
        },
        'execution': {
            'mode': 'paper',
            'interval': '1h',
            'slippage': 0.001,
            'fees': 0.001
        },
        'binance': {
            'api_key': '${BINANCE_API_KEY}',
            'api_secret': '${BINANCE_API_SECRET}',
            'testnet': True
        },
        'backtest': {
            'initial_capital': 100000,
            'start_date': '2023-01-01',
            'end_date': '2023-12-31'
        },
        'logging': {
            'level': 'INFO',
            'file': 'logs/trading.log'
        }
    }