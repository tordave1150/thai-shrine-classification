"""Ensure cleanup cannot be undone by accidentally invoking the retired CLI."""
import unittest
from unittest.mock import patch
import pipeline_extensions


class RetiredProtocolTests(unittest.TestCase):
    def test_retired_stages_fail_before_dispatch(self):
        for stage in ("prepare", "ablation", "finetune", "report"):
            with self.subTest(stage=stage), patch.object(pipeline_extensions, "stage_" + stage) as dispatch:
                with self.assertRaises(SystemExit) as error:
                    pipeline_extensions.main(["--stage", stage])
                self.assertIn("retired fixed-Full-recipe", str(error.exception))
                self.assertIn("not implemented yet", str(error.exception))
                dispatch.assert_not_called()

    def test_dry_run_does_not_restore_retired_stages(self):
        for stage in ("prepare", "ablation", "finetune", "report"):
            with self.subTest(stage=stage), patch.object(pipeline_extensions, "stage_" + stage) as dispatch:
                with self.assertRaises(SystemExit):
                    pipeline_extensions.main(["--stage", stage, "--dry-run"])
                dispatch.assert_not_called()

    def test_historical_analysis_still_dispatches(self):
        with patch.object(pipeline_extensions, "stage_analyze") as dispatch, patch("builtins.print"):
            pipeline_extensions.main(["--stage", "analyze"])
            dispatch.assert_called_once()


if __name__ == "__main__":
    unittest.main()
