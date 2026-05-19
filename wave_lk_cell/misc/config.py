from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg


def save_results(results: dict[str, float], output_dir: str | Path, name: str = "test_results") -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    json_path = out / f"{name}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    csv_path = out / f"{name}.csv"
    with open(csv_path, "w") as f:
        f.write("metric,value\n")
        for k, v in results.items():
            f.write(f"{k},{v}\n")

    print(f"\nResults saved to:\n  {json_path}\n  {csv_path}")
