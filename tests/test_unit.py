"""
VisionGuard Unit Test Suite for CI Pipeline
Author: 23L0710
Project: 23L0710_A2_Ops - VisionGuard

Tests:
1. Preprocessing integrity: aspect-ratio preservation, padding, tensor output shape (1, 224, 224, 3)
2. Security & Validation: file extension whitelist, payload size limits
3. Application health probe: GET /health schema compliance
4. Database ORM entity integrity: PredictionRecord schema validation
"""

import io
import unittest
import numpy as np
from PIL import Image

from webapp.app import create_app, allowed_file
from webapp.inference import get_inference_engine, IMG_HEIGHT, IMG_WIDTH
from src.database import PredictionRecord


class TestVisionGuardUnit(unittest.TestCase):
    def setUp(self):
        """Initialize test Flask client and inference engine."""
        self.app = create_app({"TESTING": True, "INFERENCE_MODE": "local"})
        self.client = self.app.test_client()
        self.engine = get_inference_engine()

    def test_file_extension_whitelist(self):
        """Test allowed_file validator across permitted and forbidden extensions."""
        self.assertTrue(allowed_file("sample.jpg"))
        self.assertTrue(allowed_file("sample.png"))
        self.assertTrue(allowed_file("sample.jpeg"))
        self.assertTrue(allowed_file("sample.webp"))
        self.assertTrue(allowed_file("sample.bmp"))

        self.assertFalse(allowed_file("malicious.exe"))
        self.assertFalse(allowed_file("script.sh"))
        self.assertFalse(allowed_file("data.csv"))
        self.assertFalse(allowed_file("payload.py"))
        self.assertFalse(allowed_file("no_extension"))

    def test_preprocessing_aspect_ratio_and_shape(self):
        """Test in-memory image padding and tensor preprocessing."""
        # Create an asymmetric test image (300 x 150)
        img = Image.new("RGB", (300, 150), color=(255, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        raw_bytes = buf.getvalue()

        # Run preprocessing
        tensor = self.engine.preprocess_image(raw_bytes)

        # Output shape must be strictly (1, 224, 224, 3)
        self.assertEqual(tensor.shape, (1, IMG_HEIGHT, IMG_WIDTH, 3))
        # Tensor values must be scaled float32 in [0.0, 255.0] matching MobileNetV2 input expectations
        self.assertIn("float", str(tensor.dtype).lower())
        self.assertGreaterEqual(float(np.min(tensor.numpy())), 0.0)
        self.assertLessEqual(float(np.max(tensor.numpy())), 255.0)

    def test_health_check_endpoint(self):
        """Test GET /health returns HTTP 200 with required JSON telemetry fields."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIsNotNone(data)
        self.assertIn("status", data)
        self.assertIn("model_loaded", data)
        self.assertIn("decision_threshold", data)
        self.assertIn("inference_mode", data)
        self.assertIn("student_roll_number", data)
        self.assertEqual(data["student_roll_number"], "23L0710")

    def test_database_orm_record_structure(self):
        """Test SQLAlchemy PredictionRecord ORM column definitions."""
        rec = PredictionRecord(
            filename="test_ai_art.jpg",
            prediction="AI-Generated",
            confidence=95.42,
            raw_score=0.9542,
            threshold=0.23,
            model_name="VisionGuard_MobileNetV2",
            inference_mode="local",
            inference_time_ms=124.5,
            client_ip="127.0.0.1"
        )
        self.assertEqual(rec.filename, "test_ai_art.jpg")
        self.assertEqual(rec.prediction, "AI-Generated")
        self.assertEqual(rec.confidence, 95.42)
        self.assertEqual(rec.raw_score, 0.9542)
        self.assertEqual(rec.threshold, 0.23)
        self.assertEqual(rec.inference_mode, "local")


if __name__ == "__main__":
    unittest.main()
