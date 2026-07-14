"""Simple config loader — supports JSON and YAML (if PyYAML installed)."""
from __future__ import annotations

import json
import os
from typing import Any, Dict


def load_config(path: str) -> Dict[str, Any]:
    """Load a .json or .yaml config file and return a plain dict."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    with open(path) as f:
        if ext == ".json":
            return json.load(f)
        elif ext in (".yaml", ".yml"):
            try:
                import yaml  # type: ignore
                return yaml.safe_load(f)
            except ImportError:
                raise ImportError(
                    "PyYAML is required for YAML configs: pip install pyyaml"
                )
        else:
            raise ValueError(f"Unsupported config format: {ext}")
