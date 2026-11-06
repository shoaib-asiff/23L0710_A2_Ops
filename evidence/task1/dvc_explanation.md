# Task 1.4: DVC Data Version Control Technical Report
**Student Roll Number:** 23L0710  
**Project:** 23L0710_A2_Ops - VisionGuard  

---

## 1. Overview of Task 1.4 Implementations

Subtask 1.4 was completed in two distinct stages:
1. **Toy Demonstration (`toy_dvc_demo/`):** An isolated Git and DVC environment tracking a small CSV dataset across two consecutive versions (`v1.0` and `v2.0`). Demonstrated tag-based version restoration (`git checkout v1.0 && dvc checkout`), remote storage configuration, pushing data, deleting local files, and pulling data back (`dvc pull`).
2. **Applied Dataset Tracking:** Initialized Git and DVC in the root VisionGuard repository (`23L0710_A2_Ops`), tracked the entire raw dataset folder (`archive/`) containing 978 files (~503.8 MB), pushed data to local DVC remote storage (`/tmp/visionguard_dvc_storage`), and committed the lightweight `archive.dvc` pointer file to Git.

---

## 2. Anatomy of the DVC Pointer File (`archive.dvc`)

When `dvc add archive` is executed, DVC creates a small, human-readable YAML pointer file named `archive.dvc`:

```yaml
outs:
- md5: 21fa43c3eb533c1613e51cd0085035d5.dir
  size: 503801822
  nfiles: 978
  hash: md5
  path: archive
```

### Breakdown of Pointer File Fields:
1. **`md5` (`21fa43c3eb533c1613e51cd0085035d5.dir`):**
   - For a directory, DVC computes an MD5 content hash of a directory manifest JSON file.
   - The `.dir` suffix signifies that this hash points to an internal directory index stored in `.dvc/cache/21/fa43c3eb533c1613e51cd0085035d5.dir`.
   - This manifest contains an array of relative paths and individual MD5 hashes for all 978 files inside the directory.
   - Any addition, deletion, or modification of even a single byte in any image alters this master hash, providing strict cryptographic content integrity.
2. **`size` (`503801822` bytes):**
   - The exact aggregate byte count of all files tracked inside `archive/` (~503.8 MB).
3. **`nfiles` (`978`):**
   - The total number of files indexed within the directory.
4. **`path` (`archive`):**
   - The relative workspace path where DVC expects the directory to reside when checked out.

---

## 3. Why Large Datasets Do Not Belong in Git

Committing raw binary data, high-resolution images, and model weights directly to Git is considered an anti-pattern in software and MLOps engineering for several reasons:

1. **Git is Optimized for Text Diffs, Not Binary Data:**
   - Git operates by computing line-by-line differences (deltas) between commits.
   - Binary files (such as JPEGs, PNGs, and NumPy arrays) cannot be diffed meaningfully. Even a 1-pixel change or compression adjustment in an image causes Git to store the entire replacement file in `.git/objects/`.
2. **Permanent Repository Bloat:**
   - Every time a dataset is modified, re-split, or cleaned, committing it directly to Git causes the `.git` database to grow monotonically.
   - Even if files are later deleted with `git rm`, the full binary history remains permanently embedded in Git packfiles, forcing every future developer to download gigabytes of historical data upon `git clone`.
3. **Platform and Bandwidth Constraints:**
   - Remote platforms like GitHub enforce strict limits: individual files over 100 MB are blocked, and repositories exceeding 1–5 GB incur severe warnings or performance degradation.
4. **Decoupling Code from Data Life Cycles:**
   - Source code evolves rapidly (multiple commits per day), whereas datasets evolve in discrete batches (cleaning, relabeling, new collection).
   - DVC decouples the storage medium (storing large binaries on remote object stores such as S3, Google Cloud Storage, or DagsHub) while maintaining lightweight text pointer files (`.dvc`) inside Git.
   - This guarantees reproducible data versioning without bloating the code repository.

---

## 4. Verification Evidence & Command Outputs

### Toy Demonstration Execution Summary:
- **Version 1:** 5 records tracked $\to$ Git commit `b4b3a96` $\to$ Tag `v1.0`
- **Version 2:** 10 records tracked $\to$ Git commit `111edf5` $\to$ Tag `v2.0`
- **Checkout to `v1.0`:** File restored to exactly 6 lines (header + 5 data rows).
- **Checkout to `main` (`v2.0`):** File restored to exactly 11 lines (header + 10 data rows).
- **Remote Push & Pull:** Pushed to `/tmp/dvc_toy_remote_storage`, local file deleted, and restored cleanly via `dvc pull`.
- Full command log preserved in: `evidence/task1/dvc_toy_demo_log.txt`

### Applied Dataset Tracking Summary:
- **Directory Tracked:** `archive/` (978 files, 503,801,822 bytes)
- **DVC Pointer File:** `archive.dvc` (MD5: `21fa43c3eb533c1613e51cd0085035d5.dir`)
- **Remote Storage:** `local_remote` (`/tmp/visionguard_dvc_storage`)
- **Initial Graded Commit:** `362136f` (`feat(dvc): initialize DVC and track raw dataset archive.dvc`)
- Full command log preserved in: `evidence/task1/dvc_applied_log.txt`
