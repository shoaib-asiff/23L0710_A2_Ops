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
