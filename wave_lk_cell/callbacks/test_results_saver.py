from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import lightning.pytorch as pl


class TestResultsSaver(pl.Callback):
    def __init__(self, output_dir: str = "test_results", experiment_name: str = "WaveLKCell") -> None:
        super().__init__()
        self.output_dir = Path(output_dir)
        self.experiment_name = experiment_name

    def on_test_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        logged = trainer.callback_metrics

        if not logged:
            return

        flat: dict[str, Any] = {}
        for k, v in logged.items():
            if hasattr(v, "item"):
                flat[k] = round(float(v.item()), 6)
            else:
                flat[k] = v

        self.output_dir.mkdir(parents=True, exist_ok=True)

        json_path = self.output_dir / f"{self.experiment_name}_test_results.json"
        with open(json_path, "w") as f:
            json.dump(flat, f, indent=2)

        csv_path = self.output_dir / f"{self.experiment_name}_test_results.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            for k, v in flat.items():
                writer.writerow([k, v])

        print(f"\nTest results saved to:\n  {json_path}\n  {csv_path}")
