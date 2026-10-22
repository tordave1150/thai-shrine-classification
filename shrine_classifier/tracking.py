from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


class RunLogger:
    """Append local history first; optionally mirror it to W&B."""
    def __init__(self, history_path: Path, *, mode: str, project: str, group: str,
                 name: str, config: dict[str, Any], entity: str | None = None, active_run=None) -> None:
        self.history_path, self.mode = history_path, mode
        self.project, self.group, self.name, self.config, self.entity = project, group, name, config, entity
        self.run = active_run
        self.owns_run = False
        self.tracking_error: str | None = None
        self._fallback_no = 0
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        if active_run is None and mode != "disabled":
            try:
                import wandb
                self.run = wandb.init(project=project, group=group, name=name, config=config,
                                      entity=entity, mode=mode, reinit=True)
                self.owns_run = True
            except Exception as exc:
                self.tracking_error = f"{type(exc).__name__}: {exc}"
                try:
                    if mode == "online":
                        import wandb
                        self.run = wandb.init(project=project, group=group, name=name, config=config,
                                              entity=entity, mode="offline", reinit=True)
                        self.owns_run = True
                        self.mode = "offline"
                    else:
                        self.mode = "disabled"
                except Exception as fallback_exc:
                    self.tracking_error += f"; offline fallback: {type(fallback_exc).__name__}: {fallback_exc}"
                    self.mode = "disabled"

    def log(self, values: dict[str, Any]) -> None:
        base_fields = ["epoch", "train_loss", "train_acc", "train_macro_f1", "val_loss", "val_acc",
                       "val_macro_f1", "learning_rate", "best_epoch", "best_val_acc", "best_val_macro_f1",
                       "test_loss", "test_acc", "test_macro_precision", "test_macro_recall", "test_macro_f1"]
        class_fields = [f"{metric}_{cls}" for metric in ("precision", "recall", "f1", "support")
                        for cls in ("san_phraphrom", "san_phraphum", "san_tayay", "tee_ju_eia")]
        fields = base_fields + class_fields
        exists = self.history_path.exists()
        with self.history_path.open("a", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            if not exists:
                writer.writeheader()
            writer.writerow({k: values.get(k, "") for k in fields})
        if self.run is not None:
            try:
                self.run.log(values, step=int(values["epoch"]) if values.get("epoch") is not None else None)
            except Exception as exc:
                self.tracking_error = f"{type(exc).__name__}: {exc}"
                old_run, self.run = self.run, None
                try:
                    if self.owns_run:
                        old_run.finish(exit_code=1)
                except Exception as finish_exc:
                    self.tracking_error += f"; failed to finish online run: {type(finish_exc).__name__}: {finish_exc}"
                self.owns_run = False
                if self.mode == "online":
                    try:
                        import wandb
                        self._fallback_no += 1
                        self.run = wandb.init(project=self.project, group=self.group,
                            name=f"{self.name}_offline{self._fallback_no}", config=self.config,
                            entity=self.entity, mode="offline", reinit=True)
                        self.owns_run = True
                        self.mode = "offline"
                    except Exception as fallback_exc:
                        self.tracking_error += f"; offline fallback failed: {type(fallback_exc).__name__}: {fallback_exc}"
                        self.mode = "disabled"

    def finish(self, *, exit_code: int = 0) -> None:
        if self.owns_run and self.run is not None:
            try:
                self.run.finish(exit_code=exit_code)
            except Exception as exc:
                message=f"{type(exc).__name__}: {exc}"
                self.tracking_error=f"{self.tracking_error}; finish: {message}" if self.tracking_error else message
                self.mode="disabled"
            finally:
                self.owns_run=False
