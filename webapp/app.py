"""
VisionGuard Web Application (Flask)
Author: 23L0710
Project: 23L0710_A2_Ops

Features:
- Enforces strict input validation:
  1. File type verification against ALLOWED_EXTENSIONS
  2. Max file size enforcement (5MB via MAX_CONTENT_LENGTH)
  3. Image decodability validation via Pillow (rejects corrupt/spoofed files)
- Zero-disk storage policy: All processing occurs entirely in-memory (io.BytesIO + Base64 Data URI)
- Filename sanitization using werkzeug.utils.secure_filename
- Production configuration driven by environment variables (.env)
- Routes: '/', '/predict', '/history', '/health'
- Custom friendly error pages for 400, 404, 413, 500
"""

import os
import sys
import io
import time
import base64
from datetime import datetime
from typing import Dict, Any, List

# Sandboxed runtime directories
os.environ["KERAS_HOME"] = "/tmp/.keras"
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, abort
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge, BadRequest
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

from webapp.inference import predict_image, get_inference_engine, MODEL_PATH, DEFAULT_THRESHOLD, INFERENCE_MODE

# Allowed image extensions
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp"}
MAX_CONTENT_LENGTH_MB = 5
MAX_CONTENT_LENGTH_BYTES = MAX_CONTENT_LENGTH_MB * 1024 * 1024


def allowed_file(filename: str) -> bool:
    """Checks if the filename has an allowed extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def create_app(test_config=None) -> Flask:
    """Application factory for VisionGuard Flask web app."""
    app = Flask(__name__, template_folder="templates", static_folder="static")

    # Configuration driven by environment variables (Task 4.4)
    app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "visionguard-secret-key-23l0710")
    app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH_BYTES
    app.config["MODEL_PATH"] = os.environ.get("MODEL_PATH", MODEL_PATH)
    app.config["DECISION_THRESHOLD"] = float(os.environ.get("DECISION_THRESHOLD", DEFAULT_THRESHOLD))
    app.config["INFERENCE_MODE"] = os.environ.get("INFERENCE_MODE", INFERENCE_MODE)

    if test_config:
        app.config.update(test_config)

    # In-memory audit trail storage (ready to be backed by DB in Task 5)
    app.audit_trail: List[Dict[str, Any]] = []

    # Pre-warm model inference engine on app creation
    try:
        get_inference_engine(
            model_path=app.config["MODEL_PATH"],
            threshold=app.config["DECISION_THRESHOLD"]
        )
    except Exception as e:
        app.logger.warning(f"Engine deferred startup: {e}")

    # ==========================================
    # ROUTES (Task 4.2 & 4.3)
    # ==========================================

    @app.route("/", methods=["GET"])
    def index():
        """Home page with image upload interface."""
        return render_template("index.html")

    @app.route("/predict", methods=["POST"])
    def predict():
        """
        Receives image upload, executes validation, performs inference,
        logs to audit trail, and renders result page.
        """
        # 1. Check if file part is present
        if "image" not in request.files:
            return render_template(
                "error.html",
                error_code=400,
                error_title="Missing Image Upload",
                error_message="No image file part was found in the submission. Please select an image to inspect."
            ), 400

        file = request.files["image"]

        # 2. Check if a file was selected
        if file.filename == "" or not file:
            return render_template(
                "error.html",
                error_code=400,
                error_title="No File Selected",
                error_message="Please select an image file before clicking Analyze Image."
            ), 400

        # 3. Sanitize filename (Werkzeug security)
        filename = secure_filename(file.filename)
        if not filename:
            filename = "uploaded_image.jpg"

        # 4. VALIDATION CASE 1: Reject Wrong File Types
        if not allowed_file(filename):
            ext = filename.rsplit(".", 1)[1].lower() if "." in filename else "unknown"
            return render_template(
                "error.html",
                error_code=400,
                error_title="Unsupported File Format",
                error_message=(
                    f"The file format '.{ext}' is not supported. "
                    f"VisionGuard only accepts image files in PNG, JPG, JPEG, WEBP, or BMP format."
                )
            ), 400

        # 5. Read stream entirely in-memory (ZERO DISK WRITES)
        try:
            raw_bytes = file.read()
            if not raw_bytes or len(raw_bytes) == 0:
                raise ValueError("Uploaded file buffer is completely empty.")
        except Exception as e:
            return render_template(
                "error.html",
                error_code=400,
                error_title="Empty File Stream",
                error_message="The uploaded file contains zero bytes or could not be read."
            ), 400

        # 6. VALIDATION CASE 3: Reject Undecodable / Corrupt Images
        try:
            with Image.open(io.BytesIO(raw_bytes)) as pil_img:
                pil_img.verify()
        except Exception as e:
            return render_template(
                "error.html",
                error_code=400,
                error_title="Corrupt or Undecodable Image",
                error_message=(
                    "The file has a valid image extension, but its binary data is corrupted, "
                    "truncated, or not a genuine decodable image."
                )
            ), 400

        # 7. Convert in-memory bytes to Base64 Data URI for display (Zero disk storage)
        b64_str = base64.b64encode(raw_bytes).decode("utf-8")
        # Detect mime type from file extension
        mime_type = "image/jpeg"
        if filename.lower().endswith(".png"):
            mime_type = "image/png"
        elif filename.lower().endswith(".webp"):
            mime_type = "image/webp"
        data_uri = f"data:{mime_type};base64,{b64_str}"

        # 8. Execute Model Inference
        try:
            result = predict_image(io.BytesIO(raw_bytes), threshold=app.config["DECISION_THRESHOLD"])
        except Exception as e:
            app.logger.error(f"Inference error: {e}")
            return render_template(
                "error.html",
                error_code=500,
                error_title="Inference Pipeline Error",
                error_message=f"An error occurred while evaluating the image: {str(e)}"
            ), 500

        # 9. Record to Audit Trail (Task 4.2 / Task 5 DB hook)
        record = {
            "id": len(app.audit_trail) + 1,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "filename": filename,
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "raw_score": result["raw_score"],
            "inference_time_ms": result["inference_time_ms"],
            "model_name": result["model_name"]
        }
        app.audit_trail.insert(0, record)  # Newest first

        # If Task 5 database model is available, insert into DB as well
        try:
            from src.database import log_prediction_to_db
            log_prediction_to_db(record)
        except Exception:
            pass  # Task 5 database integration connects here

        # 10. Render Result Page
        return render_template(
            "result.html",
            result=result,
            filename=filename,
            image_data_uri=data_uri
        )

    @app.route("/history", methods=["GET"])
    def history():
        """Audit trail page showing past classifications."""
        # Check if database records exist
        records = app.audit_trail
        try:
            from src.database import get_prediction_history
            db_records = get_prediction_history()
            if db_records:
                records = db_records
        except Exception:
            pass
        return render_template("history.html", history_records=records)

    @app.route("/health", methods=["GET"])
    def health():
        """Health-check API endpoint returning system status (Task 4.4)."""
        engine = get_inference_engine()
        is_loaded = engine.model is not None or engine.mode == "remote"

        health_data = {
            "status": "healthy" if is_loaded else "degraded",
            "model_loaded": is_loaded,
            "model_name": engine.model.name if engine.model else "VisionGuard_MobileNetV2_Champion",
            "model_path": app.config["MODEL_PATH"],
            "inference_mode": app.config["INFERENCE_MODE"],
            "decision_threshold": app.config["DECISION_THRESHOLD"],
            "max_upload_size_mb": MAX_CONTENT_LENGTH_MB,
            "student_roll_number": "23L0710",
            "project_name": "23L0710_A2_Ops",
            "timestamp": datetime.now().isoformat()
        }
        return jsonify(health_data), 200

    # ==========================================
    # ERROR HANDLERS (Task 4.2 Validation)
    # ==========================================

    @app.errorhandler(413)
    @app.errorhandler(RequestEntityTooLarge)
    def handle_payload_too_large(error):
        """VALIDATION CASE 2: File Size Exceeds Limit (>5MB)"""
        return render_template(
            "error.html",
            error_code=413,
            error_title="File Size Limit Exceeded",
            error_message=(
                f"The uploaded file exceeds the maximum allowed size limit of {MAX_CONTENT_LENGTH_MB}.0 MB. "
                "Please compress or select a smaller image."
            )
        ), 413

    @app.errorhandler(400)
    @app.errorhandler(BadRequest)
    def handle_bad_request(error):
        return render_template(
            "error.html",
            error_code=400,
            error_title="Invalid Request",
            error_message=str(error.description) if hasattr(error, "description") else "The request was invalid or incomplete."
        ), 400

    @app.errorhandler(404)
    def handle_not_found(error):
        return render_template(
            "error.html",
            error_code=404,
            error_title="Page Not Found",
            error_message="The requested page or endpoint does not exist on VisionGuard."
        ), 404

    @app.errorhandler(500)
    def handle_server_error(error):
        return render_template(
            "error.html",
            error_code=500,
            error_title="Internal Server Error",
            error_message="An unexpected server error occurred. Please try again shortly."
        ), 500

    return app


# Production application instance
app = create_app()

if __name__ == "__main__":
    # Task 4.4: Run in Flask server with debug mode OFF
    port = int(os.environ.get("PORT", 5000))
    host = os.environ.get("HOST", "0.0.0.0")
    print(f"[*] Starting VisionGuard Flask server on http://{host}:{port} with debug=False")
    app.run(host=host, port=port, debug=False)

