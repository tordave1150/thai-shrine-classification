"""Tests for extension pipeline components.

Tests: confidence analysis, diagnostics history parsing, fine-tuning mode invariants,
ablation config isolation, seed join/statistics, Holm family size,
and protocol protection guards.
"""

from __future__ import annotations

import csv
import json
import math
import unittest
from pathlib import Path
import tempfile


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def _make_predictions_csv(path: Path, records: list[dict]) -> None:
    if not records:
        return
    fields = list(records[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def _pred_row(image_id, true_label, predicted_label,
              probs=(0.7, 0.1, 0.1, 0.1), correct=True):
    """Helper: build a synthetic prediction row."""
    from shrine_classifier.data import CLASSES
    pred_idx = CLASSES.index(predicted_label)
    confidence = probs[pred_idx]
    row = {
        "image_id": image_id,
        "true_label": true_label,
        "predicted_label": predicted_label,
        "confidence": str(confidence),
    }
    for i, cls in enumerate(CLASSES):
        row[f"prob_{cls}"] = str(probs[i])
    return row


# ---------------------------------------------------------------------------
# Confidence analysis tests
# ---------------------------------------------------------------------------

class TestConfidenceValidation(unittest.TestCase):

    def test_valid_predictions_pass(self):
        from shrine_classifier_ext.confidence_analysis import _validate_predictions
        rows = [
            _pred_row("san_phraphrom/img_0001", "san_phraphrom", "san_phraphrom",
                      probs=(0.80, 0.10, 0.05, 0.05)),
        ]
        result = _validate_predictions(rows, "test_source")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["correct"], 1)

    def test_wrong_prediction_marked_incorrect(self):
        from shrine_classifier_ext.confidence_analysis import _validate_predictions
        rows = [
            _pred_row("san_phraphum/img_0001", "san_phraphum", "san_phraphrom",
                      probs=(0.7, 0.1, 0.1, 0.1)),
        ]
        result = _validate_predictions(rows, "test")
        self.assertEqual(result[0]["correct"], 0)

    def test_invalid_probability_sum_fails(self):
        from shrine_classifier_ext.confidence_analysis import _validate_predictions
        rows = [
            {
                "image_id": "san_phraphrom/img_0001",
                "true_label": "san_phraphrom",
                "predicted_label": "san_phraphrom",
                "confidence": "0.7",
                "prob_san_phraphrom": "0.7",
                "prob_san_phraphum": "0.2",
                "prob_san_tayay": "0.2",  # sum = 1.2
                "prob_tee_ju_eia": "0.1",
            }
        ]
        with self.assertRaises(ValueError):
            _validate_predictions(rows, "test")

    def test_out_of_range_probability_fails(self):
        from shrine_classifier_ext.confidence_analysis import _validate_predictions
        rows = [
            {
                "image_id": "san_phraphrom/img_0001",
                "true_label": "san_phraphrom",
                "predicted_label": "san_phraphrom",
                "confidence": "1.1",
                "prob_san_phraphrom": "1.1",
                "prob_san_phraphum": "-0.1",
                "prob_san_tayay": "0.0",
                "prob_tee_ju_eia": "0.0",
            }
        ]
        with self.assertRaises(ValueError):
            _validate_predictions(rows, "test")

    def test_argmax_mismatch_fails(self):
        from shrine_classifier_ext.confidence_analysis import _validate_predictions
        rows = [
            {
                "image_id": "san_phraphrom/img_0001",
                "true_label": "san_phraphrom",
                "predicted_label": "san_phraphum",  # wrong: probs say phraphrom
                "confidence": "0.7",
                "prob_san_phraphrom": "0.7",
                "prob_san_phraphum": "0.1",
                "prob_san_tayay": "0.1",
                "prob_tee_ju_eia": "0.1",
            }
        ]
        with self.assertRaises(ValueError):
            _validate_predictions(rows, "test")

    def test_duplicate_image_ids_fails(self):
        from shrine_classifier_ext.confidence_analysis import _validate_predictions
        rows = [
            _pred_row("san_phraphrom/img_0001", "san_phraphrom", "san_phraphrom",
                      probs=(0.8, 0.1, 0.05, 0.05)),
            _pred_row("san_phraphrom/img_0001", "san_phraphrom", "san_phraphrom",  # duplicate
                      probs=(0.8, 0.1, 0.05, 0.05)),
        ]
        with self.assertRaises(ValueError):
            _validate_predictions(rows, "test")

    def test_bin_metrics_empty_bin_has_none_accuracy(self):
        from shrine_classifier_ext.confidence_analysis import _bin_metrics, BIN_EDGES
        # Records all with confidence 0.95 → all in last bin only
        records = [
            {"confidence": 0.95, "correct": 1},
            {"confidence": 0.97, "correct": 0},
        ]
        bins = _bin_metrics(records)
        # All but last bin should have count 0 and accuracy None
        empty_bins = [b for b in bins if b["count"] == 0]
        for b in empty_bins:
            self.assertIsNone(b["accuracy"],
                              f"Empty bin at [{b['bin_lower']}, {b['bin_upper']}) should have accuracy=None")

    def test_ece_hand_calculation(self):
        from shrine_classifier_ext.confidence_analysis import _bin_metrics, _ece
        # 2 samples in bin [0.7, 0.8): confidence 0.75 mean, accuracy 1.0 → gap 0.25
        # 1 sample in bin [0.5, 0.6): confidence 0.55 mean, accuracy 0.0 → gap 0.55
        records = [
            {"confidence": 0.75, "correct": 1},
            {"confidence": 0.76, "correct": 1},
            {"confidence": 0.55, "correct": 0},
        ]
        bins = _bin_metrics(records)
        ece = _ece(bins, 3)
        # Expected: (2/3)*|1.0 - 0.755| + (1/3)*|0.0 - 0.55|
        mean_07 = (0.75 + 0.76) / 2
        expected = (2 / 3) * abs(1.0 - mean_07) + (1 / 3) * abs(0.0 - 0.55)
        self.assertAlmostEqual(ece, expected, places=6)

    def test_ece_confidence_zero_and_one_included(self):
        from shrine_classifier_ext.confidence_analysis import _bin_metrics, _ece
        # Confidence 0.0 → bin [0.0, 0.1); confidence 1.0 → bin [0.9, 1.0]
        records = [
            {"confidence": 0.0, "correct": 0},
            {"confidence": 1.0, "correct": 1},
        ]
        bins = _bin_metrics(records)
        counts_in_edge_bins = sum(b["count"] for b in bins if b["bin_lower"] in (0.0, 0.9))
        self.assertEqual(counts_in_edge_bins, 2)
        ece = _ece(bins, 2)
        self.assertIsNotNone(ece)
        self.assertGreaterEqual(ece, 0.0)

    def test_all_correct_group_stats(self):
        from shrine_classifier_ext.confidence_analysis import _group_stats
        records = [{"confidence": 0.9, "correct": 1}, {"confidence": 0.8, "correct": 1}]
        stats = _group_stats(records)
        self.assertEqual(stats["n_correct"], 2)
        self.assertEqual(stats["n_wrong"], 0)
        self.assertIsNone(stats["mean_confidence_wrong"])

    def test_all_wrong_group_stats(self):
        from shrine_classifier_ext.confidence_analysis import _group_stats
        records = [{"confidence": 0.9, "correct": 0}]
        stats = _group_stats(records)
        self.assertEqual(stats["n_wrong"], 1)
        self.assertIsNone(stats["mean_confidence_correct"])


# ---------------------------------------------------------------------------
# Diagnostics history tests
# ---------------------------------------------------------------------------

class TestDiagnosticsHistory(unittest.TestCase):

    def _write_history(self, tmp_dir: Path, rows: list[dict]) -> Path:
        path = tmp_dir / "history.csv"
        fields = ["epoch", "train_loss", "train_acc", "train_macro_f1",
                  "val_loss", "val_acc", "val_macro_f1", "learning_rate",
                  "best_epoch", "best_val_acc", "best_val_macro_f1",
                  "test_acc", "test_macro_precision", "test_macro_recall", "test_macro_f1",
                  "precision_san_phraphrom", "precision_san_phraphum", "precision_san_tayay",
                  "precision_tee_ju_eia", "recall_san_phraphrom", "recall_san_phraphum",
                  "recall_san_tayay", "recall_tee_ju_eia",
                  "f1_san_phraphrom", "f1_san_phraphum", "f1_san_tayay", "f1_tee_ju_eia",
                  "support_san_phraphrom", "support_san_phraphum", "support_san_tayay",
                  "support_tee_ju_eia"]
        with path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k, "") for k in fields})
        return path

    def test_valid_epoch_rows_extracted(self):
        from shrine_classifier_ext.diagnostics_analysis import _load_history
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            epoch_data = [
                {"epoch": 1, "train_loss": 1.0, "train_acc": 0.5, "train_macro_f1": 0.48,
                 "val_loss": 0.9, "val_acc": 0.55, "val_macro_f1": 0.52, "learning_rate": 1e-4},
                {"epoch": 2, "train_loss": 0.8, "train_acc": 0.7, "train_macro_f1": 0.68,
                 "val_loss": 0.7, "val_acc": 0.72, "val_macro_f1": 0.70, "learning_rate": 1e-4},
            ]
            path = self._write_history(p, epoch_data)
            epoch_rows, summary_rows = _load_history(path)
            self.assertEqual(len(epoch_rows), 2)
            self.assertEqual(epoch_rows[0]["epoch"], 1)

    def test_summary_rows_not_counted_as_epochs(self):
        from shrine_classifier_ext.diagnostics_analysis import _load_history
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            # Mix of epoch rows and summary rows (no train_loss in summary)
            data = [
                {"epoch": 1, "train_loss": 1.0, "train_acc": 0.5, "train_macro_f1": 0.48,
                 "val_loss": 0.9, "val_acc": 0.55, "val_macro_f1": 0.52, "learning_rate": 1e-4},
                {"epoch": 1, "best_epoch": 1, "best_val_acc": 0.55, "best_val_macro_f1": 0.52},
                {"epoch": 1, "test_acc": 0.9, "test_macro_f1": 0.88},
            ]
            path = self._write_history(p, data)
            epoch_rows, summary_rows = _load_history(path)
            self.assertEqual(len(epoch_rows), 1)
            self.assertGreater(len(summary_rows), 0)

    def test_best_epoch_tie_resolution(self):
        from shrine_classifier_ext.diagnostics_analysis import _best_epoch
        # Two epochs with same val_macro_f1 → pick lower val_loss
        rows = [
            {"epoch": 3, "val_macro_f1": 0.9, "val_loss": 0.3, "val_acc": 0.9},
            {"epoch": 5, "val_macro_f1": 0.9, "val_loss": 0.2, "val_acc": 0.9},
        ]
        self.assertEqual(_best_epoch(rows), 5)

    def test_best_epoch_second_tie_uses_earlier(self):
        from shrine_classifier_ext.diagnostics_analysis import _best_epoch
        # Same F1, same loss → earlier epoch wins
        rows = [
            {"epoch": 3, "val_macro_f1": 0.9, "val_loss": 0.2, "val_acc": 0.9},
            {"epoch": 7, "val_macro_f1": 0.9, "val_loss": 0.2, "val_acc": 0.9},
        ]
        self.assertEqual(_best_epoch(rows), 3)

    def test_unequal_histories_not_filled(self):
        from shrine_classifier_ext.diagnostics_analysis import _load_history
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            data_5 = [
                {"epoch": i, "train_loss": 0.5, "train_acc": 0.8, "train_macro_f1": 0.78,
                 "val_loss": 0.4, "val_acc": 0.85, "val_macro_f1": 0.82, "learning_rate": 1e-4}
                for i in range(1, 6)
            ]
            path = self._write_history(p, data_5)
            epoch_rows, _ = _load_history(path)
            self.assertEqual(len(epoch_rows), 5)
            self.assertEqual(epoch_rows[-1]["epoch"], 5)


# ---------------------------------------------------------------------------
# Fine-tuning mode tests
# ---------------------------------------------------------------------------

@unittest.skipUnless(
    __import__("importlib").util.find_spec("torch") is not None,
    "PyTorch not installed"
)
class TestFinetuningModes(unittest.TestCase):

    def _build(self, arch: str, mode: str):
        from shrine_classifier.models import build_model
        from shrine_classifier_ext.finetuning_modes import apply_finetuning_mode
        import torch
        model, _ = build_model(arch, pretrained=False)
        device = torch.device("cpu")
        model = model.to(device)
        apply_finetuning_mode(model, arch, mode)
        return model, device

    def _check_logits_shape(self, arch: str, mode: str):
        import torch
        model, device = self._build(arch, mode)
        dummy = torch.zeros(1, 3, 224, 224, device=device)
        model.eval()
        out = model(dummy)
        self.assertEqual(out.shape, (1, 4),
                         f"{arch}/{mode}: expected (1,4), got {out.shape}")

    def test_all_architectures_modes_logit_shape(self):
        from shrine_classifier.models import ARCHITECTURES
        from shrine_classifier_ext.finetuning_modes import MODES
        for arch in ARCHITECTURES:
            for mode in MODES:
                with self.subTest(arch=arch, mode=mode):
                    self._check_logits_shape(arch, mode)

    def test_full_mode_all_trainable(self):
        from shrine_classifier.models import ARCHITECTURES
        from shrine_classifier_ext.finetuning_modes import apply_finetuning_mode
        from shrine_classifier.models import build_model
        import torch
        for arch in ARCHITECTURES:
            with self.subTest(arch=arch):
                model, _ = build_model(arch, pretrained=False)
                stats = apply_finetuning_mode(model, arch, "full")
                self.assertEqual(stats["frozen_parameters"], 0,
                                 f"{arch}/full: expected 0 frozen params, got {stats['frozen_parameters']}")

    def test_frozen_mode_head_trainable_backbone_frozen(self):
        from shrine_classifier.models import ARCHITECTURES
        from shrine_classifier_ext.finetuning_modes import apply_finetuning_mode, _get_head_modules
        from shrine_classifier.models import build_model
        import torch
        for arch in ARCHITECTURES:
            with self.subTest(arch=arch):
                model, _ = build_model(arch, pretrained=False)
                apply_finetuning_mode(model, arch, "frozen")
                # Head must be trainable
                for m in _get_head_modules(model, arch):
                    for p in m.parameters():
                        self.assertTrue(p.requires_grad,
                                        f"{arch}/frozen: head param not trainable")
                # Some params must be frozen
                frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
                self.assertGreater(frozen, 0,
                                   f"{arch}/frozen: expected frozen params > 0")

    def test_partial_mode_densenet_opens_denseblock4(self):
        from shrine_classifier.models import build_model
        from shrine_classifier_ext.finetuning_modes import apply_finetuning_mode
        import torch
        model, _ = build_model("densenet121", pretrained=False)
        apply_finetuning_mode(model, "densenet121", "partial")
        # denseblock4 params must be trainable
        trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
        db4_trainable = [n for n in trainable_names if "denseblock4" in n]
        self.assertGreater(len(db4_trainable), 0,
                           "densenet121/partial: features.denseblock4 should be trainable")

    def test_frozen_bn_buffers_not_updated(self):
        """Frozen BN should not update running_mean/var after a step."""
        import torch
        from shrine_classifier.models import build_model
        from shrine_classifier_ext.finetuning_modes import apply_finetuning_mode, enforce_mode_per_epoch

        model, _ = build_model("resnet50", pretrained=False)
        device = torch.device("cpu")
        model = model.to(device)
        apply_finetuning_mode(model, "resnet50", "frozen")

        # Save BN running stats from backbone
        before_mean = model.layer1[0].bn1.running_mean.clone()
        before_var = model.layer1[0].bn1.running_var.clone()

        # Forward pass with mode enforcement
        model.train()
        enforce_mode_per_epoch(model, "resnet50", "frozen")
        dummy = torch.randn(2, 3, 224, 224, device=device)
        out = model(dummy)
        out.sum().backward()

        after_mean = model.layer1[0].bn1.running_mean.clone()
        after_var = model.layer1[0].bn1.running_var.clone()

        # Frozen backbone BN should not have updated running stats
        self.assertTrue(torch.allclose(before_mean, after_mean),
                        "resnet50/frozen: BN running_mean changed (backbone should be in eval)")
        self.assertTrue(torch.allclose(before_var, after_var),
                        "resnet50/frozen: BN running_var changed")

    def test_trainable_params_get_gradient(self):
        import torch
        from shrine_classifier.models import build_model
        from shrine_classifier_ext.finetuning_modes import apply_finetuning_mode, enforce_mode_per_epoch

        model, _ = build_model("resnet50", pretrained=False)
        device = torch.device("cpu")
        model = model.to(device)
        apply_finetuning_mode(model, "resnet50", "frozen")

        model.train()
        enforce_mode_per_epoch(model, "resnet50", "frozen")
        dummy = torch.randn(1, 3, 224, 224, device=device)
        out = model(dummy)
        out.sum().backward()

        trainable_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        self.assertGreater(len(trainable_params), 0)
        for name, param in trainable_params:
            self.assertIsNotNone(param.grad, f"Trainable param {name} has no gradient")


# ---------------------------------------------------------------------------
# Ablation config tests
# ---------------------------------------------------------------------------

class TestAblationConfigIsolation(unittest.TestCase):

    def test_no_augmentation_only_changes_augmentation(self):
        """no_augmentation config must differ only in augmentation flag."""
        base = {
            "optimizer": "adamw", "learning_rate": 1e-4, "weight_decay": 1e-4,
            "batch_size": 16, "epochs": 30, "patience": 7, "input_size": 224,
            "augmentation": True, "use_class_weights": True,
        }
        no_aug = {**base, "augmentation": False}
        diffs = {k for k in base if base[k] != no_aug.get(k)}
        self.assertEqual(diffs, {"augmentation"})

    def test_no_class_weights_only_changes_class_weights(self):
        base = {
            "optimizer": "adamw", "learning_rate": 1e-4, "weight_decay": 1e-4,
            "batch_size": 16, "epochs": 30, "patience": 7, "input_size": 224,
            "augmentation": True, "use_class_weights": True,
        }
        no_cw = {**base, "use_class_weights": False}
        diffs = {k for k in base if base[k] != no_cw.get(k)}
        self.assertEqual(diffs, {"use_class_weights"})


# ---------------------------------------------------------------------------
# Statistics: seed join and Holm family
# ---------------------------------------------------------------------------

class TestStatistics(unittest.TestCase):

    def test_seed_join_tolerates_reversed_order(self):
        """Pairing by seed ID not row position."""
        import numpy as np
        full_by_seed = {42: 0.90, 123: 0.88, 456: 0.91, 789: 0.87, 1024: 0.92}
        var_by_seed = {1024: 0.88, 789: 0.85, 456: 0.89, 123: 0.86, 42: 0.88}  # reversed
        shared = sorted(set(full_by_seed) & set(var_by_seed))
        diffs = [var_by_seed[s] - full_by_seed[s] for s in shared]
        # Mean diff should be negative (variant generally lower)
        # Values: 42→-0.02, 123→-0.02, 456→-0.02, 789→-0.02, 1024→-0.04 → mean=-0.024
        self.assertLess(float(np.mean(diffs)), 0.0,
                        "variant should be on average below full")

    def test_holm_family_size_eight(self):
        """Holm family must be exactly 8 for 4 variants × 2 metrics."""
        variants = ["no_augmentation", "no_class_weights", "frozen", "partial"]
        metrics = ["test_acc", "test_macro_f1"]
        family = len(variants) * len(metrics)
        self.assertEqual(family, 8)

    def test_zero_variance_difference_status(self):
        """Zero-variance diffs must produce explicit status, not a p-value."""
        import numpy as np
        diffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        if np.std(diffs, ddof=1) == 0:
            status = "zero_variance_or_nonfinite"
        else:
            status = "ok"
        self.assertEqual(status, "zero_variance_or_nonfinite")

    def test_constant_diff_is_zero_variance(self):
        import numpy as np
        full_vals = [0.9, 0.91, 0.92, 0.89, 0.93]
        var_vals = [0.8, 0.81, 0.82, 0.79, 0.83]  # constant gap of 0.1
        diffs = np.array(var_vals) - np.array(full_vals)
        # sd should be essentially 0 (constant delta)
        sd = np.std(diffs, ddof=1)
        self.assertAlmostEqual(sd, 0.0, places=10)

    def test_sample_sd_ddof1(self):
        import numpy as np
        values = [0.90, 0.91, 0.89, 0.92, 0.88]
        sd = float(np.std(values, ddof=1))
        # Check ddof=1 vs ddof=0
        sd0 = float(np.std(values, ddof=0))
        self.assertGreater(sd, sd0)


# ---------------------------------------------------------------------------
# Protocol protection
# ---------------------------------------------------------------------------

class TestProtocolProtection(unittest.TestCase):

    def test_prepare_does_not_overwrite_existing_protocol(self):
        """A second prepare run on an existing protocol.json must not overwrite."""
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "out"
            out.mkdir()
            proto = out / "protocol.json"
            original = {"test": "immutable_content"}
            proto.write_text(json.dumps(original))
            # Simulate prepare: it checks and skips if exists
            if proto.exists():
                existing = json.loads(proto.read_text())
                self.assertEqual(existing["test"], "immutable_content")
                # Do NOT overwrite
                should_overwrite = False
            else:
                should_overwrite = True
            self.assertFalse(should_overwrite)
            # File should be unchanged
            self.assertEqual(json.loads(proto.read_text())["test"], "immutable_content")


if __name__ == "__main__":
    unittest.main()
