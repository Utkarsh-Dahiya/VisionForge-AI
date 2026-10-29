import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
import cv2  # noqa: E402
import helpers  # noqa: E402


class HelperTests(unittest.TestCase):
    def test_decode_invalid_bytes(self):
        self.assertIsNone(helpers.decode_image(b"not an image"))
        self.assertIsNone(helpers.decode_image(b""))

    def test_decode_roundtrip(self):
        img = np.zeros((20, 30, 3), np.uint8)
        data = cv2.imencode(".png", img)[1].tobytes()
        out = helpers.decode_image(data)
        self.assertEqual(out.shape, (20, 30, 3))

    def test_hash_differs(self):
        self.assertNotEqual(helpers.file_hash(b"a"), helpers.file_hash(b"b"))

    def test_temp_file_removed_on_success_and_error(self):
        img = np.zeros((8, 8, 3), np.uint8)
        with helpers.temporary_image_file(img) as p:
            self.assertTrue(p.exists())
        self.assertFalse(p.exists())
        with self.assertRaises(ValueError):
            with helpers.temporary_image_file(img) as p2:
                raise ValueError
        self.assertFalse(p2.exists())

    def test_temp_files_are_unique(self):
        img = np.zeros((8, 8, 3), np.uint8)
        with helpers.temporary_image_file(img) as a, helpers.temporary_image_file(img) as b:
            self.assertNotEqual(a, b)

    def test_detections_frame_and_empty(self):
        self.assertTrue(helpers.detections_to_frame([]).empty)
        self.assertTrue(helpers.detections_to_frame(None).empty)
        f = helpers.detections_to_frame(
            [{"label": "person", "confidence": 0.9}, {"label": "dog", "confidence": 0.5}])
        self.assertEqual(list(f.columns), ["ID", "Object", "Confidence"])
        self.assertEqual(len(f), 2)
        self.assertIn("person,0.9", f.to_csv(index=False))

    def test_distribution(self):
        f = helpers.detections_to_frame([{"label": "a", "confidence": .1}] * 2
                                        + [{"label": "b", "confidence": .2}])
        d = helpers.class_distribution(f)
        self.assertEqual(d.iloc[0].tolist(), ["a", 2])
        self.assertTrue(helpers.class_distribution(helpers.detections_to_frame([])).empty)

    def test_summary_and_history(self):
        result = {"detections": [{"label": "cat", "confidence": 0.8}], "object_count": 1,
                  "inference_time_ms": 12.5, "image_width": 10, "image_height": 5}
        rec = helpers.summarize_analysis("x.png", result, 0.25)
        self.assertEqual(rec["objects"], 1)
        self.assertAlmostEqual(rec["mean_confidence"], 0.8)
        empty = helpers.summarize_analysis("y.png", {"detections": [], "object_count": 0}, 0.5)
        self.assertIsNone(empty["mean_confidence"])
        h = helpers.history_frame([rec, empty])
        self.assertEqual(len(h), 1)

    def test_find_weights_missing(self):
        self.assertIsNone(helpers.find_weights(Path("/nonexistent"), "nope.pt"))


if __name__ == "__main__":
    unittest.main()
