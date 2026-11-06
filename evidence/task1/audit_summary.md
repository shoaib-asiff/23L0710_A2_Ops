# Task 1.2: Dataset Audit Report
**Project:** 23L0710_A2_Ops - VisionGuard  
**Dataset Path:** `./archive`  
**Total Images Audited:** 973  

---

## 1. Class Distribution and Image Integrity Summary

| Class        |   Total Files |   Valid Images |   Corrupt Files |   Duplicates |   Width (Min) |   Width (Median) |   Width (Max) |   Height (Min) |   Height (Median) |   Height (Max) |   Size KB (Min) |   Size KB (Median) |   Size KB (Max) |
|:-------------|--------------:|---------------:|----------------:|-------------:|--------------:|-----------------:|--------------:|---------------:|------------------:|---------------:|----------------:|-------------------:|----------------:|
| AI-Generated |           538 |            538 |               0 |            2 |           225 |           1028.5 |          6000 |            135 |               800 |           4001 |           11.46 |             195.12 |         5494.32 |
| Real         |           435 |            435 |               0 |            0 |           183 |            940   |          8495 |            158 |               720 |           7091 |           11.05 |             141.58 |         9956.4  |
| Overall      |           973 |            973 |               0 |            2 |           183 |           1024   |          8495 |            135 |               768 |           7091 |           11.05 |             165.44 |         9956.4  |

---

## 2. Image Formats Distribution per Class
| class        |   JPEG |   PNG |
|:-------------|-------:|------:|
| AI-Generated |    428 |   110 |
| Real         |    402 |    33 |

---

## 3. Color Modes Distribution per Class
| class        |   L |   P |   RGB |   RGBA |
|:-------------|----:|----:|------:|-------:|
| AI-Generated |   1 |  14 |   488 |     35 |
| Real         |   6 |   9 |   406 |     14 |

---

## 4. Duplicate Files Analysis
- **Total Duplicate Image Instances:** 2
- **Distinct Content Hashes with Duplicates:** 1

### Duplicate Groups Detail:
| filename    | class        | sha256                                                           | filepath                                |
|:------------|:-------------|:-----------------------------------------------------------------|:----------------------------------------|
| image34.jpg | AI-Generated | f3268a4fc0477b2f3d9728f118afe32bf5196c3b476def53e34229672f77dc97 | archive/AiArtData/AiArtData/image34.jpg |
| image33.jpg | AI-Generated | f3268a4fc0477b2f3d9728f118afe32bf5196c3b476def53e34229672f77dc97 | archive/AiArtData/AiArtData/image33.jpg |
---

## 5. Corrupt / Unreadable Files Analysis
- **Total Corrupt Files Detected:** 0

- All 973 files successfully decoded with PIL (no corruption detected).
