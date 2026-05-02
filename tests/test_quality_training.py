import importlib.util
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from app.processing.quality import _normalize_label


def _load_training_script():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "train_but_ppg_quality_rf.py"
    spec = importlib.util.spec_from_file_location("train_but_ppg_quality_rf", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class QualityLabelTests(unittest.TestCase):
    def test_normalize_label_maps_binary_random_forest_classes(self) -> None:
        self.assertEqual(_normalize_label(0), "low")
        self.assertEqual(_normalize_label(1), "high")
        self.assertEqual(_normalize_label(np.int64(1)), "high")

    def test_normalize_label_preserves_named_levels(self) -> None:
        self.assertEqual(_normalize_label("low"), "low")
        self.assertEqual(_normalize_label("medium"), "medium")
        self.assertEqual(_normalize_label("high"), "high")


class ButPpgTrainingParserTests(unittest.TestCase):
    def test_build_feature_dataset_reads_rgb_records_and_reports_legacy_skips(self) -> None:
        training = _load_training_script()
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = Path(temp_dir)
            (dataset / "quality-hr-ann.csv").write_text(
                "ID,Quality,HR\n"
                "100001,0,72\n"
                "100002,1,74\n"
                "100003,1,74\n",
                encoding="utf-8",
            )
            self._write_rgb_record(dataset, "100001", bpm=72)
            self._write_rgb_record(dataset, "100002", bpm=74)
            self._write_legacy_record(dataset, "100003")

            feature_dataset = training.build_feature_dataset(dataset)

        self.assertEqual(feature_dataset.features.shape, (2, len(training.FEATURE_NAMES)))
        self.assertEqual(feature_dataset.labels.tolist(), ["low", "high"])
        self.assertEqual(feature_dataset.groups.tolist(), ["100", "100"])
        self.assertEqual(feature_dataset.skipped, {"legacy_transposed": 1})

    @staticmethod
    def _write_rgb_record(dataset: Path, record_id: str, *, bpm: float) -> None:
        record_dir = dataset / record_id
        record_dir.mkdir()
        (record_dir / f"{record_id}_PPG.hea").write_text(
            f"{record_id}_PPG 3 30 300\n"
            f"{record_id}_PPG.dat 16 1(0)/a.u. 0 0 0 0 0 PPG_R\n"
            f"{record_id}_PPG.dat 16 1(0)/a.u. 0 0 0 0 0 PPG_G\n"
            f"{record_id}_PPG.dat 16 1(0)/a.u. 0 0 0 0 0 PPG_B\n",
            encoding="utf-8",
        )
        t = np.arange(300) / 30.0
        pulse = np.sin(2 * math.pi * bpm / 60.0 * t)
        red = 10_000 + 700 * pulse
        green = 12_000 + 1_000 * pulse
        blue = 8_000 + 500 * pulse
        samples = np.column_stack([red, green, blue]).astype("<i2")
        samples.tofile(record_dir / f"{record_id}_PPG.dat")

    @staticmethod
    def _write_legacy_record(dataset: Path, record_id: str) -> None:
        record_dir = dataset / record_id
        record_dir.mkdir()
        (record_dir / f"{record_id}_PPG.hea").write_text(
            f"{record_id}_PPG 300 30 1\n"
            f"{record_id}_PPG.dat 16 1(0)/a.u. 0 0 0 0 0\n",
            encoding="utf-8",
        )
        np.zeros(300, dtype="<i2").tofile(record_dir / f"{record_id}_PPG.dat")


if __name__ == "__main__":
    unittest.main()
