import csv
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from shrine_classifier.data import CLASSES, NEUTRAL, groups_from_audit, letterbox, open_rgb


class DataContractTests(unittest.TestCase):
    def test_letterbox_preserves_aspect_and_uses_neutral_padding(self):
        image = Image.new("RGB", (640, 320), (180, 20, 30))
        canvas, bbox = letterbox(image, 224)
        self.assertEqual(canvas.size, (224, 224))
        self.assertEqual(bbox, (0, 56, 224, 168))
        self.assertEqual(canvas.getpixel((0, 0)), NEUTRAL)
        self.assertEqual(canvas.getpixel((100, 100)), (180, 20, 30))

    def test_transparency_is_composited_on_neutral_color(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "alpha.png"
            image = Image.new("RGBA", (5, 5), (255, 0, 0, 0))
            image.putpixel((2, 2), (200, 10, 10, 255))
            image.save(path)
            converted = open_rgb(path)
            self.assertEqual(converted.mode, "RGB")
            self.assertEqual(converted.getpixel((0, 0)), NEUTRAL)
            self.assertEqual(converted.getpixel((2, 2)), (200, 10, 10))

    def test_confirmed_duplicate_groups_are_transitive_and_label_safe(self):
        rows = [
            {"image_id": "san_phraphum/a", "label": "san_phraphum", "pixel_sha256": "same", "source_url": "u1"},
            {"image_id": "san_phraphum/b", "label": "san_phraphum", "pixel_sha256": "b", "source_url": "u1"},
            {"image_id": "san_phraphum/c", "label": "san_phraphum", "pixel_sha256": "c", "source_url": "u2"},
        ]
        with tempfile.TemporaryDirectory() as temp:
            review = Path(temp) / "review.csv"
            with review.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=["id_a", "id_b", "review_status"])
                writer.writeheader(); writer.writerow({"id_a": rows[1]["image_id"], "id_b": rows[2]["image_id"], "review_status": "confirmed_same_source"})
            groups = groups_from_audit(rows, review)
            self.assertEqual(len(set(groups.values())), 1)

    def test_class_order_is_explicit(self):
        self.assertEqual(CLASSES, ["san_phraphrom", "san_phraphum", "san_tayay", "tee_ju_eia"])


if __name__ == "__main__":
    unittest.main()
