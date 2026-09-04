import io
import os
import unittest
from app import app


class TestBloodCellApp(unittest.TestCase):

    def setUp(self):
        app.config["TESTING"] = True
        self.client = app.test_client()

    def test_overview_page(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Blood Cell AI", response.data)

    def test_analysis_page_get(self):
        response = self.client.get("/analysis")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(b"Specimen" in response.data or b"Analysis" in response.data)

    def test_batch_page_get(self):
        response = self.client.get("/batch")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Batch", response.data)

    def test_library_page_get(self):
        response = self.client.get("/library")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(b"Cell Library" in response.data or b"Eosinophil" in response.data)

    def test_model_page_get(self):
        response = self.client.get("/model")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(b"Model" in response.data or b"Architecture" in response.data)

    def test_api_docs_page_get(self):
        response = self.client.get("/api/docs")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"REST API", response.data)

    def test_api_health(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("status", data)
        self.assertIn("classes", data)
        self.assertEqual(len(data["classes"]), 4)

    def test_api_samples(self):
        response = self.client.get("/api/samples")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(len(data["samples"]), 4)

    def test_api_predict_sample(self):
        sample_file = "static/cell-library/neutrophil.jpg"
        if os.path.exists(sample_file):
            with open(sample_file, "rb") as fh:
                img_bytes = fh.read()
            response = self.client.post(
                "/api/predict",
                data={"file": (io.BytesIO(img_bytes), "test_neutrophil.jpg")},
                content_type="multipart/form-data",
            )
            self.assertEqual(response.status_code, 200)
            json_data = response.get_json()
            self.assertTrue(json_data["success"])
            record = json_data["data"]
            self.assertIn("prediction", record)
            self.assertIn("confidence", record)
            self.assertIn("morphology", record)


if __name__ == "__main__":
    unittest.main()
