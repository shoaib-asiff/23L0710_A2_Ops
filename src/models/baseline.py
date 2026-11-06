"""
Task 2.2: Baseline CNN Architecture from Scratch
Author: 23L0710
Project: 23L0710_A2_Ops - VisionGuard

Implements:
- Modular custom CNN with Conv2D, BatchNormalization, MaxPooling2D, Dropout, Dense
- Rescaling layer to normalize pixel values [0, 255] -> [0, 1]
- Binary classification head (sigmoid output)
- Compilation with BinaryCrossentropy, Adam optimizer, and evaluation metrics
- Generates model summary, layer justification table, and architectural report
"""

import os
import sys

# Configure sandboxed runtime paths
os.environ["KERAS_HOME"] = "/tmp/.keras"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow.keras import layers, models, metrics, optimizers, losses

OUTPUT_DIR = "evidence/task2"
SUMMARY_TXT = os.path.join(OUTPUT_DIR, "baseline_summary.txt")
ARCH_MD = os.path.join(OUTPUT_DIR, "baseline_architecture.md")

def build_baseline_cnn(input_shape=(224, 224, 3), learning_rate=1e-3):
    """
    Constructs a lightweight baseline CNN designed from scratch.
    
    Architecture:
    Input (224x224x3)
      -> Rescaling [0, 1]
      -> Block 1: Conv2D(32, 3x3) -> BatchNorm -> MaxPool(2x2)
      -> Block 2: Conv2D(64, 3x3) -> BatchNorm -> MaxPool(2x2)
      -> Block 3: Conv2D(128, 3x3) -> BatchNorm -> MaxPool(2x2)
      -> Block 4: Conv2D(128, 3x3) -> BatchNorm -> MaxPool(2x2)
      -> GlobalAveragePooling2D
      -> Dense(128, ReLU) -> BatchNorm -> Dropout(0.4)
      -> Dense(1, Sigmoid)
    """
    model = models.Sequential(name="VisionGuard_Baseline_CNN")
    
    # 1. Input & Normalization
    model.add(layers.Input(shape=input_shape, name="input_tensor"))
    model.add(layers.Rescaling(1.0 / 255.0, name="pixel_rescaling"))
    
    # 2. Convolutional Block 1: Low-level edges & gradients (32 filters)
    model.add(layers.Conv2D(32, kernel_size=(3, 3), padding="same", activation="relu", name="conv1"))
    model.add(layers.BatchNormalization(name="bn1"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), name="pool1"))
    
    # 3. Convolutional Block 2: Texture & motif detection (64 filters)
    model.add(layers.Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu", name="conv2"))
    model.add(layers.BatchNormalization(name="bn2"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), name="pool2"))
    
    # 4. Convolutional Block 3: Structural patterns & shapes (128 filters)
    model.add(layers.Conv2D(128, kernel_size=(3, 3), padding="same", activation="relu", name="conv3"))
    model.add(layers.BatchNormalization(name="bn3"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), name="pool3"))
    
    # 5. Convolutional Block 4: High-level semantic synthesis (128 filters)
    model.add(layers.Conv2D(128, kernel_size=(3, 3), padding="same", activation="relu", name="conv4"))
    model.add(layers.BatchNormalization(name="bn4"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), name="pool4"))
    
    # 6. Global Pooling: Eliminates parameter explosion and enforces spatial invariance
    model.add(layers.GlobalAveragePooling2D(name="global_avg_pool"))
    
    # 7. Dense Classification Head
    model.add(layers.Dense(128, activation="relu", name="dense_features"))
    model.add(layers.BatchNormalization(name="bn_dense"))
    model.add(layers.Dropout(0.4, name="dropout_regularization"))
    
    # 8. Output Layer: Single probability score for AI-Generated vs Real
    model.add(layers.Dense(1, activation="sigmoid", name="output_probability"))
    
    # Compilation
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss=losses.BinaryCrossentropy(),
        metrics=[
            metrics.BinaryAccuracy(name="accuracy"),
            metrics.Precision(name="precision"),
            metrics.Recall(name="recall"),
            metrics.AUC(name="auc")
        ]
    )
    
    return model

def export_model_reports(model):
    """Saves model summary and generates a formatted Markdown architecture table."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Save standard Keras summary text
    stringlist = []
    model.summary(print_fn=lambda x: stringlist.append(x))
    summary_text = "\n".join(stringlist)
    
    with open(SUMMARY_TXT, "w") as f:
        f.write(summary_text)
    print(f"[+] Saved model summary text: {SUMMARY_TXT}")
    
    # 2. Extract layer details for structured table
    layer_data = []
    for layer in model.layers:
        layer_data.append({
            "Layer Name": layer.name,
            "Layer Type": layer.__class__.__name__,
            "Output Shape": str(layer.output_shape),
            "Param Count": f"{layer.count_params():,}",
            "Trainable": layer.trainable
        })
        
    import pandas as pd
    df_layers = pd.DataFrame(layer_data)
    
    total_params = model.count_params()
    trainable_params = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
    non_trainable_params = total_params - trainable_params
    
    # 3. Build Markdown Report with Technical Justifications
    report_md = f"""# Task 2.2: Baseline CNN Architecture & Design Rationale
**Student Roll Number:** 23L0710  
**Project:** 23L0710_A2_Ops - VisionGuard  
**Model Name:** `VisionGuard_Baseline_CNN`  

---

## 1. Layer-by-Layer Architectural Specifications

{df_layers.to_markdown(index=False)}

### Parameter Summary:
- **Total Parameters:** {total_params:,} ({total_params * 4 / (1024**2):.2f} MB float32 weights)
- **Trainable Parameters:** {trainable_params:,}
- **Non-Trainable Parameters (BatchNorm running stats):** {non_trainable_params:,}

---

## 2. Layer Design Rationales & Engineering Explanations

| Layer Name | Type | Purpose & Technical Rationale |
| :--- | :--- | :--- |
| `pixel_rescaling` | `Rescaling` | Converts raw pixel tensors [0.0, 255.0] to normalized [0.0, 1.0]. Prevents numerical instability and exploding gradients during backpropagation. |
| `conv1` (32 filters) | `Conv2D (3x3)` | Extracts low-level visual primitives, such as fine digital noise, gradient edges, and high-frequency pixel transitions characteristic of synthetic generators. |
| `bn1`, `bn2`, `bn3`, `bn4` | `BatchNormalization` | Normalizes intermediate activations across each mini-batch (zero mean, unit variance). Mitigates internal covariate shift, accelerates training convergence, and provides a regularizing effect. |
| `pool1`, `pool2`, `pool3`, `pool4` | `MaxPooling2D (2x2)` | Halves spatial dimensions at each stage (224 -> 112 -> 56 -> 28 -> 14), expanding the receptive field while providing translational invariance and reducing computational load. |
| `conv2` (64 filters) | `Conv2D (3x3)` | Captures intermediate visual patterns, localized textures, and synthetic skin/hair smoothing artifacts. |
| `conv3` (128 filters) | `Conv2D (3x3)` | Captures complex geometric shapes, lighting inconsistencies, and compositional structures. |
| `conv4` (128 filters) | `Conv2D (3x3)` | Synthesizes high-level semantic features across wide receptive fields to distinguish holistic generative patterns from photographic realism. |
| `global_avg_pool` | `GlobalAveragePooling2D` | Computes the spatial mean across each of the 128 feature maps (14x14 -> 1). Compared to Flatten(), this reduces parameter count from 25,088 to 128, eliminating spatial overfitting and enforcing global translation invariance. |
| `dense_features` (128 units) | `Dense (ReLU)` | Fully connected non-linear feature integration layer mapping spatial representations into class-discriminative decision boundaries. |
| `dropout_regularization` | `Dropout (0.4)` | Randomly deactivates 40% of feature activations during training passes, preventing co-adaptation of neurons and mitigating overfitting on small datasets. |
| `output_probability` | `Dense (Sigmoid)` | Outputs a single scalar probability in range [0.0, 1.0], representing the calibrated likelihood that the input image is AI-Generated (Label = 1). |

---

## 3. Justification of Loss, Optimizer, and Metrics

### 1. Loss Function: `BinaryCrossentropy`
- **Formula:** `L(y, y_hat) = - [ y * log(y_hat) + (1 - y) * log(1 - y_hat) ]`
- **Justification:** VisionGuard is a mutually exclusive two-class problem (Real vs. AI-Generated). Binary Crossentropy mathematically corresponds to the negative log-likelihood of the Bernoulli distribution, directly penalizing confident incorrect classifications with steep logarithmic loss slopes.

### 2. Optimizer: `Adam` (Learning Rate = 1e-3)
- **Justification:** Adaptive Moment Estimation (Adam) combines the advantages of AdaGrad (handling sparse gradients) and RMSProp (exponentially decaying average of squared gradients). It maintains per-parameter adaptive learning rates, providing fast convergence and robust optimization on convolutional feature maps without requiring manual momentum tuning.

### 3. Evaluation Metrics:
- **Binary Accuracy:** Measures overall correctness across the dataset.
- **Precision:** `TP / (TP + FP)`. Critical for measuring the proportion of flagged images that are genuinely AI-generated (guarding against false accusations on real artists).
- **Recall:** `TP / (TP + FN)`. Measures the detection rate of AI-generated content (guarding against synthetic media slipping through undetected).
- **ROC-AUC:** Measures the Area Under the Receiver Operating Characteristic curve across all possible decision thresholds, assessing the classifier's discriminative power independent of any specific threshold.
"""
    with open(ARCH_MD, "w") as f:
        f.write(report_md)
    print(f"[+] Saved architectural report & justification table: {ARCH_MD}")

def main():
    print("=" * 70)
    print("        TASK 2.2: BASELINE CNN ARCHITECTURE & MODEL AUDIT             ")
    print("=" * 70)
    
    model = build_baseline_cnn()
    
    print("\nModel Architecture Summary:")
    model.summary()
    
    export_model_reports(model)
    
    print("\n" + "=" * 70)
    print("                 TASK 2.2 COMPLETED SUCCESSFULLY                      ")
    print("=" * 70)

if __name__ == "__main__":
    main()
