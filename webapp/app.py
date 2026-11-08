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

    # Initialize Database Trail (Task 5)
    try:
        from src.database import init_db
        init_db()
    except Exception as e:
        app.logger.warning(f"Database init warning: {e}")

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

    def _process_prediction(wants_json: bool):
        def format_error(code: int, title: str, message: str):
            if wants_json:
                return jsonify({
                    "status": "error",
                    "error_code": code,
                    "error_title": title,
                    "error_message": message
                }), code
            return render_template(
                "error.html",
                error_code=code,
                error_title=title,
                error_message=message
            ), code

        # 1. Check if file part is present (supports both 'image' and 'file')
        file = request.files.get("image") or request.files.get("file")
        if not file:
            return format_error(
                400,
                "Missing Image Upload",
                "No image file part was found in the submission. Send multipart form-data with key 'image' or 'file'."
            )

        # 2. Check if a file was selected
        if file.filename == "" or not file:
            return format_error(
                400,
                "No File Selected",
                "Please select an image file before clicking Analyze Image."
            )

        # 3. Sanitize filename (Werkzeug security)
        filename = secure_filename(file.filename)
        if not filename:
            filename = "uploaded_image.jpg"

        # 4. VALIDATION CASE 1: Reject Wrong File Types
        if not allowed_file(filename):
            ext = filename.rsplit(".", 1)[1].lower() if "." in filename else "unknown"
            return format_error(
                400,
                "Unsupported File Format",
                f"The file format '.{ext}' is not supported. VisionGuard only accepts image files in PNG, JPG, JPEG, WEBP, or BMP format."
            )

        # 5. Read stream entirely in-memory (ZERO DISK WRITES)
        try:
            raw_bytes = file.read()
            if not raw_bytes or len(raw_bytes) == 0:
                raise ValueError("Uploaded file buffer is completely empty.")
        except Exception as e:
            return format_error(
                400,
                "Empty File Stream",
                "The uploaded file contains zero bytes or could not be read."
            )

        # 6. VALIDATION CASE 3: Reject Undecodable / Corrupt Images
        try:
            with Image.open(io.BytesIO(raw_bytes)) as pil_img:
                pil_img.verify()
        except Exception as e:
            return format_error(
                400,
                "Corrupt or Undecodable Image",
                "The file has a valid image extension, but its binary data is corrupted, truncated, or not a genuine decodable image."
            )

        # 7. Execute Model Inference
        try:
            result = predict_image(io.BytesIO(raw_bytes), threshold=app.config["DECISION_THRESHOLD"])
        except Exception as e:
            app.logger.error(f"Inference error: {e}")
            return format_error(
                500,
                "Inference Pipeline Error",
                f"An error occurred while evaluating the image: {str(e)}"
            )

        # 8. Record to Audit Trail (Task 5 Database Trail)
        client_ip = request.headers.get("X-Forwarded-For", request.remote_addr) or "127.0.0.1"
        record = {
            "id": len(app.audit_trail) + 1,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "filename": filename,
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "raw_score": result["raw_score"],
            "threshold": app.config["DECISION_THRESHOLD"],
            "inference_mode": result.get("inference_mode", "local"),
            "inference_time_ms": result["inference_time_ms"],
            "model_name": result["model_name"],
            "client_ip": client_ip
        }
        app.audit_trail.insert(0, record)  # Newest first

        # Log permanently into Task 5 Database
        try:
            from src.database import log_prediction_to_db
            db_id = log_prediction_to_db(record)
            if db_id:
                record["id"] = db_id
        except Exception as e:
            app.logger.error(f"Database logging error: {e}")

        # 9. Return JSON for API requests
        if wants_json:
            return jsonify({
                "status": "success",
                "filename": filename,
                "prediction": result["prediction"],
                "confidence": result["confidence"],
                "raw_score": result["raw_score"],
                "threshold": result["threshold"],
                "model_name": result["model_name"],
                "model_version": "VisionGuard_MobileNetV2_Champion",
                "inference_mode": result.get("inference_mode", "local"),
                "inference_time_ms": result["inference_time_ms"],
                "student_roll_number": "23L0710",
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }), 200

        # 10. Convert in-memory bytes to Base64 Data URI for HTML display (Zero disk storage)
        b64_str = base64.b64encode(raw_bytes).decode("utf-8")
        mime_type = "image/jpeg"
        if filename.lower().endswith(".png"):
            mime_type = "image/png"
        elif filename.lower().endswith(".webp"):
            mime_type = "image/webp"
        data_uri = f"data:{mime_type};base64,{b64_str}"

        # 11. Render Result Page
        return render_template(
            "result.html",
            result=result,
            filename=filename,
            image_data_uri=data_uri
        )

    @app.route("/predict", methods=["POST"])
    def predict():
        """Web UI endpoint: processes upload and renders HTML result.html."""
        wants_json = (
            request.headers.get("Accept") == "application/json"
            or request.args.get("format") == "json"
        )
        return _process_prediction(wants_json=wants_json)

    @app.route("/api/predict", methods=["POST"])
    def api_predict():
        """Dedicated API endpoint: always returns JSON response for Task 7 / Postman / external clients."""
        return _process_prediction(wants_json=True)

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
        wants_json = (
            request.headers.get("Accept") == "application/json"
            or request.path.startswith("/api/")
            or request.args.get("format") == "json"
        )
        msg = f"The uploaded file exceeds the maximum allowed size limit of {MAX_CONTENT_LENGTH_MB}.0 MB. Please compress or select a smaller image."
        if wants_json:
            return jsonify({
                "status": "error",
                "error_code": 413,
                "error_title": "File Size Limit Exceeded",
                "error_message": msg
            }), 413
        return render_template(
            "error.html",
            error_code=413,
            error_title="File Size Limit Exceeded",
            error_message=msg
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
    import socket

    def is_port_bindable(port_num: int, host_ip: str = "0.0.0.0") -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((host_ip, port_num))
                return True
            except OSError:
                return False

    host = os.environ.get("HOST", "0.0.0.0")
    requested_port = int(os.environ.get("PORT", 5000))
    port = requested_port

    if not is_port_bindable(port, host):
        # Port is occupied (e.g. AirPlay Receiver on port 5000 or Docker on port 5001)
        for candidate in [5002, 5003, 5050, 8080]:
            if is_port_bindable(candidate, host):
                print(
                    f"[!] Notice: Port {requested_port} is occupied by macOS AirPlay Receiver / background services.\n"
                    f"[+] Auto-switching local Flask server to available port: http://127.0.0.1:{candidate}"
                )
                port = candidate
                break

    print(f"[*] Starting VisionGuard Flask server on http://127.0.0.1:{port} (host: {host}) with debug=False")
    app.run(host=host, port=port, debug=False)

