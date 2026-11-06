# Task 4.3: HTML & CSS Frontend Interface Technical Report
**Student Roll Number:** 23L0710  
**Project:** 23L0710_A2_Ops - VisionGuard  
**Templates Created:** `base.html`, `index.html`, `result.html`, `history.html`, `error.html`  
**Stylesheet:** `webapp/static/css/style.css`  

---

## 1. Interface Design & User Experience (UX)

The user interface follows a modern, accessible, and responsive card-based aesthetic:
1. **Navigation Header:** Displays platform branding (`🛡️ VisionGuard - 23L0710_A2_Ops`), navigation links (`Classifier`, `Audit Trail`), and a live green pulse status pill linking to the `/health` diagnostic endpoint.
2. **Interactive Upload Zone:** Implements both drag-and-drop file ingestion and standard file browser selection, with client-side JavaScript providing immediate feedback on selected file size and name.
3. **Verdict Card & Confidence Visualizer:**
   - Color-coded classification badge: Emerald Green for **🌿 REAL PHOTOGRAPHY** vs Rose Red for **🤖 AI-GENERATED**.
   - Smooth animated confidence bar showing the model's calibrated confidence percentage.
   - Comprehensive metadata table displaying exact raw probability, decision threshold, inference latency in milliseconds, and model architecture.
4. **Mandatory Honest Caveat Banner:**
   Prominently placed yellow-amber advisory banner stating:
   > *"This prediction is produced by a statistical neural network (MobileNetV2) calibrated to a decision threshold of $\tau^* = 0.23$. Deep learning classifiers operate on subtle texture, frequency, and spectral patterns observed during training. High confidence indicates strong statistical resemblance, not absolute forensic proof. Results should serve as assistive evidence in conjunction with contextual human verification."*
5. **Audit Trail Page (`/history`):** A clean tabular overview displaying all recent image classifications with timestamp, filename, prediction, confidence, and latency.

---

## 2. Frontend Evidence Captures
- Home Upload Form: `evidence/task4/frontend_home_upload.png`
- Prediction Result View: `evidence/task4/frontend_prediction_result.png`
- Audit Trail History View: `evidence/task4/frontend_audit_history.png`
