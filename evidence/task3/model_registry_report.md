# Task 3.6: Model Registry & Champion Loading Technical Report
**Student Roll Number:** 23L0710  
**Project:** 23L0710_A2_Ops - VisionGuard  
**DagsHub Live Model Registry:** [https://dagshub.com/shoaib-asiff/23L0710_A2_Ops/models](https://dagshub.com/shoaib-asiff/23L0710_A2_Ops/models)  
**Registered Model Name:** `VisionGuard_Classifier`  
**Active Production Alias:** `@champion` $\rightarrow$ Version 2  

---

## 1. Model Registry Configuration on DagsHub

The VisionGuard classification model was registered in the remote DagsHub / MLflow Model Registry under the canonical name **`VisionGuard_Classifier`**.

| Version | Source Run | Description | Assigned Alias / Stage |
| :---: | :--- | :--- | :---: |
| **Version 1** | `test_model_register` | Initial candidate model registration | `None` |
| **Version 2** | `register_champion_version_2` (`exp3_champion_unfreeze20_lr5e5`) | **Champion Model:** MobileNetV2 with top 20 unfrozen layers, $\text{LR} = 5 \times 10^{-5}$, $\text{Dropout} = 0.3$, evaluated at $\tau^* = 0.23$ | **`@champion`** |

---

## 2. Standalone Model Loader Verification (`src/registry_loader.py`)

A standalone validation script was implemented to load the champion model directly from DagsHub by alias:
```python
model_uri = "models:/VisionGuard_Classifier@champion"
model = mlflow.tensorflow.load_model(model_uri)
```

### Terminal Output Proof:
```text
===========================================================================
       TASK 3.6: STANDALONE MODEL REGISTRY LOADER & INFERENCE         
===========================================================================
[*] MLflow Tracking URI   : https://dagshub.com/shoaib-asiff/23L0710_A2_Ops.mlflow
[*] DagsHub User Account  : shoaib-asiff
[*] Target Model Registry : VisionGuard_Classifier
[*] Target Model Alias    : @champion (models:/VisionGuard_Classifier@champion)
[*] Selected Test Image   : archive/RealArt/RealArt/images40.jpg
[*] True Known Label      : Real
---------------------------------------------------------------------------
[+] SUCCESS: Resolved and loaded model from registry!
    - Underlying Model Name: VisionGuard_MobileNetV2
    - Input Tensor Shape   : (None, 224, 224, 3)
    - Output Tensor Shape  : (None, 1)
---------------------------------------------------------------------------
[*] Preprocessing input image with exact training pipeline parity...
    - Batch Shape          : (1, 224, 224, 3)
    - Value Range          : [0.0, 255.0]
[*] Executing forward inference pass...
---------------------------------------------------------------------------
                       INFERENCE VERDICT RESULTS                       
---------------------------------------------------------------------------
  Raw Model Probability (AI Score) : 0.167470
  Tuned Decision Threshold (tau*)  : 0.2300
  Final Predicted Classification   : Real
  Model Confidence Percentage      : 83.25%
  True Classification Label        : Real
  Classification Match (Accuracy)  : PASS (Correct)
===========================================================================
```

---

## 3. Production Serving Architecture: Addressing the DagsHub Decoupling Rule

> [!IMPORTANT]
> **Rubric Constraint:** *"The serving app must not call DagsHub for every request. Plan how the champion model file reaches your server (Task 6)."*

### Architectural Plan for Tasks 4, 6, 7 & 8:
1. **The Anti-Pattern (Why Per-Request Calls are Unacceptable):**
   - Querying remote DagsHub over HTTPS for every incoming user request introduces 500ms to 2000ms network latency.
   - Any transient cloud network outage, DNS failure, or DagsHub maintenance would crash the serving web application.
   - Downloading large model weight files repeatedly incurs bandwidth waste and violates stateless server principles.

2. **The VisionGuard Solution (Decoupled Production Serving):**
   - **Local Champion Serialization:** During Task 3.4/3.5, the champion model artifact was serialized locally to `models/champion.keras`.
   - **Containerized Baking / Volume Mounting:** In Task 6 (Docker Compose) and Task 7 (Cloud Deployment), `models/champion.keras` is copied or mounted directly into `/app/models/champion.keras` inside the container image.
   - **One-Time Startup Loading:** In Task 4 (`webapp/inference.py`), the Flask web application loads `models/champion.keras` **once at server initialization**:
     ```python
     # Loaded ONCE at startup:
     CHAMPION_MODEL = tf.keras.models.load_model(MODEL_PATH, safe_mode=False)
     ```
   - All subsequent inference calls execute sub-millisecond in-memory forward passes completely offline, providing maximum speed, high reliability, and zero runtime dependency on external APIs.
