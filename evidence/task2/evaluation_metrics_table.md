| Model                  | Threshold               | Accuracy   | Precision   | Recall   | F1-Score   |   ROC-AUC |
|:-----------------------|:------------------------|:-----------|:------------|:---------|:-----------|----------:|
| Baseline CNN (Scratch) | tau = 0.50 (Standard)   | 58.22%     | 57.58%      | 93.83%   | 71.36%     |    0.6479 |
| Baseline CNN (Scratch) | tau* = 0.23 (Optimized) | 57.53%     | 56.64%      | 100.00%  | 72.32%     |    0.6479 |
| MobileNetV2 (Transfer) | tau = 0.50 (Standard)   | 73.29%     | 80.88%      | 67.90%   | 73.83%     |    0.8275 |
| MobileNetV2 (Transfer) | tau* = 0.23 (Optimized) | 70.55%     | 66.38%      | 95.06%   | 78.17%     |    0.8275 |