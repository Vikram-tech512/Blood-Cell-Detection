import os
import unittest
import cv2
import numpy as np

from core.morphology import analyze_morphology
from core.explainability import generate_gradcam_overlay


class TestCytometryEngine(unittest.TestCase):

    def setUp(self):
        """Create a synthetic 240x240 RGB microscopic cell image with a purple nucleus."""
        self.img = np.ones((240, 240, 3), dtype=np.uint8) * 230
        # Cytoplasm
        cv2.circle(self.img, (120, 120), 80, (220, 180, 210), -1)
        # Nucleus lobes
        cv2.circle(self.img, (100, 120), 30, (90, 30, 120), -1)
        cv2.circle(self.img, (140, 120), 30, (90, 30, 120), -1)

    def test_morphology_analysis_synthetic(self):
        results = analyze_morphology(self.img)
        self.assertIsInstance(results, dict)
        self.assertIn("nc_ratio", results)
        self.assertIn("lobe_count", results)
        self.assertIn("granularity_index", results)
        self.assertIn("cell_diameter_um", results)
        self.assertGreater(results["nc_ratio"], 0)
        self.assertGreater(results["cell_diameter_um"], 0)

    def test_morphology_analysis_real_sample(self):
        sample_path = "static/cell-library/eosinophil.jpg"
        if os.path.exists(sample_path):
            results = analyze_morphology(sample_path)
            self.assertGreater(results["nc_ratio"], 0)
            self.assertGreaterEqual(results["lobe_count"], 1)
            self.assertIn("qualitative", results)
            self.assertIn("reference_ranges", results)

    def test_gradcam_overlay_synthetic(self):
        overlay, heatmap, colored_heatmap = generate_gradcam_overlay(None, self.img)
        self.assertEqual(overlay.shape, self.img.shape)
        self.assertEqual(heatmap.shape, (self.img.shape[0], self.img.shape[1]))
        self.assertEqual(colored_heatmap.shape, self.img.shape)
        self.assertLessEqual(np.max(heatmap), 1.0)
        self.assertGreaterEqual(np.min(heatmap), 0.0)


if __name__ == "__main__":
    unittest.main()
