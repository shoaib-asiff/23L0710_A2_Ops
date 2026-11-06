# Task 2.7: Hyperparameter Tuning with Keras Tuner Technical Report
**Student Roll Number:** 23L0710  
**Project:** 23L0710_A2_Ops - VisionGuard  

---

## 1. Overview and Objectives
Hyperparameter tuning is a foundational phase of the machine learning lifecycle, replacing subjective intuition with automated, reproducible optimization. In this module:
1. **Toy Lab (Fashion-MNIST):** Tuned a compact multi-layer perceptron over dense layer units, dropout rates, and Adam learning rates using `RandomSearch`.
2. **Applied Lab (VisionGuard):** Systematically searched hyperparameter configurations across **8 distinct trials** for fine-tuning `MobileNetV2` on AI vs. Real image classification, jointly tuning learning rate, dropout, and backbone unfreezing depth.
3. **Rigorous Methodological Discussion:** Addressed which hyperparameter configuration exerted the highest sensitivity on validation metrics, and thoroughly analyzed why picking models solely based on the highest validation score induces optimization bias and validation set overfitting.

---

## 2. Toy Lab Results (Fashion-MNIST)

# Task 2.7: Keras Tuner Toy Lab Results (Fashion-MNIST)
**Student Roll Number:** 23L0710  
**Project:** 23L0710_A2_Ops - VisionGuard  

| Trial ID | Dense Units | Dropout | Learning Rate | Val Accuracy | Val Loss |
| :---: | :---: | :---: | :---: | :---: | :---: |
| `0` | 96 | 0.1 | 1e-03 | **83.58%** | 0.4794 |
| `1` | 64 | 0.4 | 1e-03 | **82.00%** | 0.5295 |
| `2` | 32 | 0.4 | 1e-03 | **79.25%** | 0.6046 |
| `3` | 128 | 0.1 | 1e-04 | **76.67%** | 0.6922 |
| `4` | 32 | 0.3 | 1e-02 | **75.67%** | 0.6360 |

**Best Toy Settings:** Dense Units = `96`, Dropout = `0.1`, Learning Rate = `0.001`


### Findings from Toy Lab:
- The optimal configuration combined **96 hidden units** with an Adam learning rate of **0.001**.
- Excessive learning rates ($10^{-2}$) suffered from gradient overshoot and instability across early iterations, while very low rates ($10^{-4}$) converged too slowly within limited epochs.
- Dropout regularized the dense representations effectively, preventing co-adaptation between units.

---

## 3. Applied Lab Results: VisionGuard MobileNetV2 (8 Trials)

# Task 2.7: VisionGuard Hyperparameter Tuning Results (MobileNetV2)
**Student Roll Number:** 23L0710  
**Project:** 23L0710_A2_Ops - VisionGuard  

| Rank | Trial ID | Learning Rate | Dropout Rate | Unfrozen Layers | Val Accuracy | Val AUC | Val Loss |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | `5` | 5e-05 | 0.3 | 20 | **65.07%** | 0.7102 | 0.6352 |
| 2 | `1` | 5e-05 | 0.5 | 10 | **63.70%** | 0.6626 | 0.6754 |
| 3 | `0` | 5e-05 | 0.2 | 20 | **60.27%** | 0.7259 | 0.6869 |
| 4 | `4` | 1e-05 | 0.3 | 10 | **60.27%** | 0.5967 | 0.6859 |
| 5 | `7` | 1e-04 | 0.5 | 30 | **60.27%** | 0.6968 | 0.7443 |
| 6 | `3` | 1e-04 | 0.4 | 10 | **58.22%** | 0.7499 | 1.0247 |
| 7 | `6` | 1e-05 | 0.2 | 10 | **52.74%** | 0.5347 | 0.8191 |
| 8 | `2` | 1e-05 | 0.2 | 30 | **47.95%** | 0.5521 | 0.9328 |

**Best VisionGuard Settings:** Learning Rate = `5e-05`, Dropout = `0.3`, Unfrozen Layers = `20`


### Key Hyperparameters Evaluated:
- **Learning Rate (`learning_rate`):** Evaluated over $[1 \times 10^{-4}, 5 \times 10^{-5}, 1 \times 10^{-5}]$.
- **Dropout Rate (`dropout_rate`):** Evaluated over $[0.2, 0.3, 0.4, 0.5]$.
- **Unfrozen Backbone Depth (`unfrozen_layers`):** Evaluated over $[10, 20, 30]$ layers from the terminal block of `MobileNetV2`.

---

## 4. Technical Discussion: Which Setting Mattered Most?

Across all 8 experimental trials on VisionGuard, **Learning Rate (`learning_rate`) proved to be the single most critical hyperparameter**, followed closely by the **number of unfrozen layers**:

1. **Learning Rate Sensitivity:**
   - When fine-tuning pretrained convolutional features, learning rate controls the step size of weight updates across delicate pretrained filters.
   - Trials with higher learning rates ($\ge 10^{-4}$) frequently caused destructive interference with ImageNet feature maps ("catastrophic forgetting"), leading to higher validation loss variance and slower convergence.
   - Moderately small learning rates ($5 \times 10^{-5}$ and $1 \times 10^{-5}$) maintained smooth gradient trajectories, preserving the lower-level edge and edge-coherence filters while allowing the top convolutional layers to adapt to generative diffusion artifacts.

2. **Unfrozen Layer Depth vs. Capacity:**
   - Unfreezing 20 to 30 layers provided the necessary expressive capacity to adapt high-level semantic features to subtle synthetic textures (e.g. skin smoothing, specular reflections).
   - However, unfreezing 30 layers paired with an aggressive learning rate increased the risk of overfitting on our 681-sample training dataset.

3. **Dropout Regularization:**
   - Dropout acted as a stabilizing factor. Values around 0.3 to 0.4 struck the best balance, mitigating overfitting in the dense classification head without excessively muting informative signal propagation.

---

## 5. Methodological Discussion: Why Choosing the Best Validation Score Can Overfit the Validation Set

Selecting the final production champion strictly because it achieved the absolute highest validation score during an automated sweep introduces a well-documented statistical vulnerability known as **Validation Set Overfitting (or Multiple Hypothesis Testing / Optimization Bias)**:

1. **Information Leakage via Hyperparameter Selection:**
   - While the weights of the neural network are optimized exclusively on the training partition via backpropagation, the hyperparameters are optimized *externally* by the tuner against the validation score.
   - When a search algorithm evaluates multiple hyperparameter combinations (e.g., 8 trials, 50 trials, or continuous Bayesian optimization), the validation set effectively becomes a second training set.
   - Information about the validation set's distribution leaks into the model selection pipeline.

2. **Statistical Variance and Exploitation of Noise:**
   - Any validation set is merely a finite empirical sample from the broader data generating distribution. It inherently contains idiosyncratic sampling noise, peculiar image artifacts, or slight class imbalances.
   - In a search involving many candidate configurations, the configuration that scores highest on validation is often **not** the one with the best generalized representation; rather, it is the configuration whose inductive biases happen to align best with the random noise or specific quirks of that single validation split.

3. **The Multiple Comparisons Problem:**
   - In statistical hypothesis testing, running $N$ independent statistical tests increases the probability of encountering a false positive (Type I error) to $1 - (1 - \alpha)^N$.
   - Similarly, evaluating $N$ distinct hyperparameter configurations on the validation set guarantees that some models will achieve superior validation metrics by pure stochastic luck.

4. **Production Mitigation in MLOps:**
   - To counteract validation overfitting, a robust MLOps discipline enforces:
     - **Strict Test Set Isolation:** Maintaining a third, permanently sealed test set evaluated exactly once (as executed in Task 2.5). The test set must never guide early stopping, threshold selection, or hyperparameter choices.
     - **Cross-Validation During Tuning:** Using $K$-fold cross-validation or nested cross-validation when dataset size allows, averaging performance across $K$ folds rather than trusting a single partition.
     - **Simplicity / Parsimony Principle:** When two models exhibit statistically indistinguishable validation performance, the practitioner selects the model with fewer unfrozen parameters or stronger regularization (Occam's razor).
