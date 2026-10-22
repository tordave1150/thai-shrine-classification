import importlib.util
import unittest

from shrine_classifier.models import ARCHITECTURES


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("torchvision"),
                     "PyTorch/torchvision absent; run pipeline.py --stage smoke on the model runtime")
class ArchitectureBackwardSmokeTests(unittest.TestCase):
    def test_five_models_emit_four_logits_and_backpropagate(self):
        import torch
        from shrine_classifier.models import build_model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        x = torch.randn(2, 3, 64, 64, device=device)
        for name in ARCHITECTURES:
            model, _ = build_model(name, pretrained=False)
            model.to(device).train()
            output = model(x)
            self.assertEqual(tuple(output.shape), (2, 4), name)
            torch.nn.functional.cross_entropy(output, torch.tensor([0, 1], device=device)).backward()
            self.assertTrue(any(p.grad is not None for p in model.parameters()), name)
            del model


if __name__ == "__main__":
    unittest.main()
