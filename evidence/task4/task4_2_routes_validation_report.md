# Task 4.2: Routes & Input Validation Technical Report
**Student Roll Number:** 23L0710  
**Project:** 23L0710_A2_Ops - VisionGuard  
**Routes Implemented:** `GET /`, `POST /predict`, `GET /history`, `GET /health`  
**Max Content Length:** 5.0 MB (`5 * 1024 * 1024` bytes)  
**Allowed Extensions:** `.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`  

---

## 1. Input Validation Architecture & Security Posture

The VisionGuard web service enforces defensive programming to protect server integrity and user security:

```
                            +-----------------------------------+
                            |       Incoming POST /predict      |
                            +-----------------+-----------------+
                                              |
                                              v
                            +-----------------------------------+
                            | Check 1: File Presence & Filename |
                            | werkzeug secure_filename()        |
                            +-----------------+-----------------+
                                              |
                                              v
                       +---------------------------------------------+
                       | Check 2: Extension Whitelist Verification   |
                       | ALLOWED_EXTENSIONS = {png, jpg, jpeg, webp} |
                       +----------------------+----------------------+
                                              |
                          [Pass]              | [Fail]
                            +                 +-------------------> HTTP 400 Bad Request
                            |                                       "Unsupported File Format"
                            v
                       +---------------------------------------------+
                       | Check 3: Flask MAX_CONTENT_LENGTH (5MB)     |
                       | Enforced at HTTP body streaming boundary    |
                       +----------------------+----------------------+
                                              |
                          [Pass]              | [Fail]
                            +                 +-------------------> HTTP 413 Payload Too Large
                            |                                       "File Size Limit Exceeded"
                            v
                       +---------------------------------------------+
                       | Check 4: Deep Image Decode via Pillow verify|
                       | In-Memory Stream Decode (Zero Disk Writes)  |
                       +----------------------+----------------------+
                                              |
                          [Pass]              | [Fail]
                            +                 +-------------------> HTTP 400 Bad Request
                            |                                       "Corrupt or Undecodable Image"
                            v
                       +---------------------------------------------+
                       | Base64 Data URI Conversion & Inference Pass|
                       +---------------------------------------------+
```

---

## 2. The Three Mandatory Failure Cases & Visual Proofs

### Failure Case 1: Wrong File Extension (Rejecting Non-Image Types)
- **Scenario:** A user or script attempts to upload a non-image file (e.g. `malicious_script.txt`, executable, or script).
- **Security Defense:** The filename is sanitized via `secure_filename()` to remove directory traversal characters (`../`). The extension is extracted and matched against `ALLOWED_EXTENSIONS`.
- **HTTP Response:** `400 Bad Request`
- **Rendered Message:** *"The file format '.txt' is not supported. VisionGuard only accepts image files in PNG, JPG, JPEG, WEBP, or BMP format."*
- **Evidence Screenshot:** `evidence/task4/failure_case1_wrong_extension.png`

### Failure Case 2: File Size Limit Exceeded (> 5MB)
- **Scenario:** An adversary or user uploads an uncompressed RAW image or massive payload exceeding server limits ($>5	ext{ MB}$).
- **Security Defense:** Flask's `MAX_CONTENT_LENGTH` aborts stream ingestion at the WSGI boundary, preventing Denial of Service (DoS) attacks via memory exhaustion. Handled gracefully by `@app.errorhandler(413)`.
- **HTTP Response:** `413 Payload Too Large`
- **Rendered Message:** *"The uploaded file exceeds the maximum allowed size limit of 5.0 MB. Please compress or select a smaller image."*
- **Evidence Screenshot:** `evidence/task4/failure_case2_size_limit.png`

### Failure Case 3: Files that Look Like Images but Cannot be Decoded (Corrupt/Spoofed)
- **Scenario:** A file has a valid extension (e.g. `broken_image.jpg`), but its byte contents are corrupted, truncated, or contain arbitrary non-image text/binary data.
- **Security Defense:** Before feeding tensors to TensorFlow, the in-memory byte buffer is opened via Pillow's `Image.open(io.BytesIO(raw_bytes))` and validated using `pil_img.verify()`. If parsing fails, an immediate friendly error is raised.
- **HTTP Response:** `400 Bad Request`
- **Rendered Message:** *"The file has a valid image extension, but its binary data is corrupted, truncated, or not a genuine decodable image."*
- **Evidence Screenshot:** `evidence/task4/failure_case3_corrupt_image.png`

---

## 3. Zero Disk Leak Verification (In-Memory Processing)
As strictly required by the assignment guidelines (*"Do not keep the uploaded image on disk"*):
1. **Direct Stream Ingestion:** In `webapp/app.py`, the uploaded file stream is consumed directly via `file.read()` into an in-memory `io.BytesIO` buffer.
2. **Ephemeral Inference:** Preprocessing, decoding, and tensor construction operate purely in RAM.
3. **Data URI Image Preview:** Rather than saving the image to a `static/uploads/` directory on disk, the in-memory bytes are encoded into a base64 Data URI string (`data:image/jpeg;base64,...`), which is passed directly to the Jinja2 template. Zero files are ever created or left on disk.
