import os
import unittest
import cv2
import numpy as np

from core.morphology import analyze_morphology
from core.explainability import generate_gradcam_overlay
from core.accuracy import assess_image_quality, calibrate_with_cytometry
from core.copilot import detect_anomalies, get_clinical_guidance, copilot_query_engine


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

    def test_image_quality_assessment(self):
        quality = assess_image_quality(self.img)
        self.assertIn("iqi_score", quality)
        self.assertIn("sharpness_val", quality)
        self.assertIn("status", quality)
        self.assertTrue(quality["passed"])

    def test_cytometry_calibration(self):
        raw_probs = {"eosinophil": 0.25, "lymphocyte": 0.25, "monocyte": 0.25, "neutrophil": 0.25}
        morphology = {"nc_ratio": 2.2, "lobe_count": 1, "granularity_index": 10.0, "cell_diameter_um": 11.0}
        calibrated = calibrate_with_cytometry(raw_probs, morphology, ["eosinophil", "lymphocyte", "monocyte", "neutrophil"])
        self.assertGreater(calibrated["lymphocyte"], calibrated["neutrophil"])

    def test_anomaly_detection(self):
        # Blast simulation: high N:C ratio and large diameter
        morphology = {"nc_ratio": 2.8, "lobe_count": 1, "granularity_index": 15.0, "cell_diameter_um": 18.0}
        anomalies = detect_anomalies("lymphocyte", 0.90, morphology)
        self.assertTrue(any(a["severity"] == "critical" for a in anomalies))

    def test_clinical_guidance(self):
        guidance = get_clinical_guidance("neutrophil")
        self.assertIn("triage", guidance)
        self.assertIn("icd10", guidance)
        self.assertIn("confirmatory_tests", guidance)

    def test_copilot_query_engine(self):
        answer = copilot_query_engine("What is the differential diagnosis?", {"cell_type": "eosinophil", "confidence": 92.0})
        self.assertIn("Eosinophil", answer)


if __name__ == "__main__":
    unittest.main()
