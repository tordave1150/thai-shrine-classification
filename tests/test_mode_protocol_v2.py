import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pipeline_modes as modes


class ModeProtocolTests(unittest.TestCase):
    def test_search_configs_repeat_from_fixed_seed(self):
        first = modes._sample_configs(12, 42)
        self.assertEqual(first, modes._sample_configs(12, 42))
        self.assertEqual(len(first), 12)
        self.assertTrue(all(1e-5 <= r["learning_rate"] <= 1e-3 for r in first))
        self.assertTrue(all(1e-6 <= r["weight_decay"] <= 1e-2 for r in first))

    def test_mobilenet_partial_mode_opens_whole_tail_stage_and_projection(self):
        from shrine_classifier.models import build_model
        from shrine_classifier_ext.finetuning_modes import _get_partial_backbone_modules

        model, _ = build_model("mobilenet_v3_large", pretrained=False)
        stage = _get_partial_backbone_modules(model, "mobilenet_v3_large")
        names = [name for name, module in model.named_modules() if any(module is item for item in stage)]
        self.assertEqual(names, ["features.13", "features.14", "features.15", "features.16"])

    def test_lock_selects_validation_and_keeps_architecture_mode_pairs(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            (out / "protocol.json").write_text(json.dumps({"protocol_fingerprint": "fixed-protocol", "source_fingerprint": "source", "assignment_fingerprint": "split"}), encoding="utf-8")
            (out / "tuning").mkdir()
            archs, selected_modes = ["resnet50", "densenet121"], ["full", "frozen"]
            rows = []
            for ai, arch in enumerate(archs):
                for mi, mode in enumerate(selected_modes):
                    for trial in (1, 2):
                        score = .75 + .01 * ai + .02 * mi + .001 * trial
                        rows.append({"identity": f"{arch}::{mode}::trial{trial:03d}", "architecture": arch,
                            "mode": mode, "trial": trial, "status": "completed",
                            "config": {"learning_rate": trial * 1e-4,
                                "weight_decay": 1e-4, "optimizer": "adamw", "batch_size": 16},
                            "best_val_macro_f1": score, "best_val_acc": score, "best_val_loss": 1-score,
                            "test_macro_f1": .01 if mode == "full" else .99,
                            "config_fingerprint": f"fp-{arch}-{mode}-{trial}", "checkpoint": f"{arch}-{mode}-{trial}.pt"})
            (out / "run_registry.json").write_text(json.dumps({"protocol_fingerprint": "fixed-protocol", "trials": rows, "final_runs": []}), encoding="utf-8")
            (out / "splits.csv").write_text("image_id,split\n", encoding="utf-8")
            args = SimpleNamespace(architectures=archs, modes=selected_modes, trials_per_pair=2,
                                   seeds=[42,123,456,789,1024], epochs=30)
            locked = modes._lock(args, out)
            self.assertEqual(locked["reference_pair"], {"architecture": "densenet121", "mode": "frozen"})
            self.assertFalse(locked["test_metrics_read"])
            winners = json.loads((out / "tuning" / "best_params_by_arch_mode.json").read_text(encoding="utf-8"))
            self.assertEqual(set(winners["pairs"]), {f"{a}::{m}" for a in archs for m in selected_modes})
            self.assertEqual(winners["pairs"]["resnet50::full"]["winning_trial"], 2)
            self.assertEqual(winners["pairs"]["resnet50::full"]["learning_rate"], 2e-4)
            self.assertEqual(modes._lock(args, out), locked)

    def test_tuning_plots_read_nested_logged_config(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            rows = [{"architecture": "resnet50", "mode": mode, "status": "completed",
                     "best_val_macro_f1": .8,
                     "config": {"learning_rate": 1e-4, "weight_decay": 1e-5}}
                    for mode in modes.MODES]
            modes._tuning_plot(out, rows)
            for filename in ("tuning_validation_scatter.png", "weight_decay_validation_scatter.png"):
                self.assertGreater((out / "tuning" / filename).stat().st_size, 0)

    def test_test_rows_join_by_seed_and_fixed_family_size_survives_undefined(self):
        ref, other = {"architecture": "resnet50", "mode": "full"}, {"architecture": "resnet50", "mode": "frozen"}
        rows = []
        for seed in [456, 42, 1024, 123, 789]:
            rows.extend([{"architecture": "resnet50", "mode": "full", "seed": seed, "status": "completed", "test_macro_f1": .8},
                         {"architecture": "resnet50", "mode": "frozen", "seed": seed, "status": "completed", "test_macro_f1": .8}])
        result = modes._test_summary(rows, ref, other, "test_macro_f1", "primary")
        self.assertEqual(result["seed_ids"], "42,123,456,789,1024")
        self.assertEqual(result["status"], "undefined_zero_variance")
        self.assertIsNone(result["p_value"])
        record = {"p_value": .01}
        modes._holm([record], 28)
        self.assertEqual(record["holm_family_size"], 28)
        self.assertEqual(record["p_holm"], .28)

    def test_cli_dry_run_reports_revised_five_trial_grid_without_starting(self):
        with tempfile.TemporaryDirectory() as temp:
            destination = Path(temp) / "new"
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = modes.main(["--stage", "all", "--dry-run", "--out-dir", str(destination)])
            plan = json.loads(output.getvalue())
            self.assertEqual(code, 0)
            self.assertEqual(plan["tuning_runs"], 75)
            self.assertEqual(plan["final_runs"], 75)
            self.assertEqual(plan["max_epochs"], 4500)
            self.assertFalse(plan["measured_run_started"])
            self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()
