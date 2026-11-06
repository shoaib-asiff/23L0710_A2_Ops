#!/usr/bin/env bash
# ==============================================================================
# Task 1.4: Applied DVC on VisionGuard Dataset
# Author: 23L0710
# Project: 23L0710_A2_Ops - VisionGuard
#
# Steps executed:
# 1. Initialize Git in root project repository (branch: main)
# 2. Initialize DVC in root repository
# 3. Configure local remote storage (/tmp/visionguard_dvc_storage)
# 4. Track raw dataset folder `archive/` with DVC
# 5. Push dataset to remote storage
# 6. Verify contents of pointer file `archive.dvc`
# 7. Commit pointer files and project baseline to Git
# ==============================================================================

set -e

DVC_BIN="$(pwd)/.venv/bin/dvc"
REMOTE_STORAGE="/tmp/visionguard_dvc_storage"

echo "======================================================================"
echo "      TASK 1.4: APPLIED DVC ON VISIONGUARD DATASET (23L0710_A2_Ops)   "
echo "======================================================================"

# Step 1: Initialize Git in root repository if not already initialized
echo -e "\n[STEP 1] Initializing Git in root repository..."
if [ ! -d ".git" ]; then
    git init -b main
    git config user.name "23L0710"
    git config user.email "23L0710@student.university.edu"
    echo "Git repository initialized on branch 'main'."
else
    echo "Git repository already initialized."
fi

# Step 2: Initialize DVC in root repository
echo -e "\n[STEP 2] Initializing DVC..."
if [ ! -d ".dvc" ]; then
    "$DVC_BIN" init
    echo "DVC repository initialized."
else
    echo "DVC already initialized in root."
fi

# Step 3: Configure remote storage
echo -e "\n[STEP 3] Configuring DVC remote storage..."
mkdir -p "$REMOTE_STORAGE"
"$DVC_BIN" remote add -d -f local_remote "$REMOTE_STORAGE"
echo "Configured default remote: 'local_remote' -> $REMOTE_STORAGE"
"$DVC_BIN" remote list

# Step 4: Track dataset folder with DVC
echo -e "\n[STEP 4] Tracking 'archive' directory with DVC..."
"$DVC_BIN" add archive
echo "'archive' directory successfully tracked by DVC."

# Step 5: Push dataset to remote storage
echo -e "\n[STEP 5] Pushing raw dataset to remote storage..."
"$DVC_BIN" push
echo "Dataset push complete. Storage location: $REMOTE_STORAGE"

# Step 6: Inspect DVC pointer file
echo -e "\n[STEP 6] Inspecting generated pointer file (archive.dvc):"
cat archive.dvc

# Step 7: Verify .gitignore updates
echo -e "\n[STEP 7] Verifying .gitignore contains /archive:"
grep -n "archive" .gitignore || true

# Step 8: Commit pointer files and project baseline to Git
echo -e "\n[STEP 8] Committing DVC pointer files and project baseline to Git..."
git add .dvc archive.dvc .gitignore requirements.txt src/ evidence/ IMPLEMENTATION_PLAN.md PROGRESS.md
git commit -m "feat(dvc): initialize DVC and track raw dataset archive.dvc" || true

echo -e "\nGit status summary:"
git status -s

echo -e "\nLatest Git commit:"
git log -1 --stat

echo -e "\n======================================================================"
echo "          APPLIED DVC TRACKING COMPLETED SUCCESSFULLY!                "
echo "======================================================================"
