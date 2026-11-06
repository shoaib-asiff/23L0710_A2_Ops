#!/usr/bin/env bash
# ==============================================================================
# Task 1.4: DVC Toy Demonstration Script
# Author: 23L0710
# Project: 23L0710_A2_Ops - VisionGuard
#
# Steps executed:
# 1. Create a clean empty directory `toy_dvc_demo`
# 2. Initialize Git and DVC
# 3. Create sample CSV (v1), track with DVC, commit v1 and tag
# 4. Modify CSV (v2), track with DVC, commit v2 and tag
# 5. Return to older version (v1) via Git and DVC checkout
# 6. Configure a remote storage location
# 7. Push dataset to remote storage
# 8. Delete local data and pull it back to prove recovery
# ==============================================================================

set -e

DVC_BIN="$(pwd)/.venv/bin/dvc"
TOY_DIR="$(pwd)/toy_dvc_demo"
REMOTE_STORAGE="/tmp/dvc_toy_remote_storage"

echo "======================================================================"
echo "         TASK 1.4: DVC TOY DEMONSTRATION (23L0710_A2_Ops)             "
echo "======================================================================"

# Step 1: Prepare clean directories
echo -e "\n[STEP 1] Setting up clean toy environment..."
rm -rf "$TOY_DIR" "$REMOTE_STORAGE"
mkdir -p "$TOY_DIR" "$REMOTE_STORAGE"
cd "$TOY_DIR"
echo "Working directory: $(pwd)"

# Step 2: Initialize Git and DVC
echo -e "\n[STEP 2] Initializing Git and DVC..."
git init -b main
git config user.name "23L0710"
git config user.email "23L0710@student.university.edu"

"$DVC_BIN" init
echo "Git and DVC successfully initialized."
git status -s

# Step 3: Create Version 1 of data.csv and track with DVC
echo -e "\n[STEP 3] Creating data.csv (Version 1) and tracking with DVC..."
cat << 'EOF' > data.csv
id,image_name,label,confidence
1,sample_001.jpg,Real,0.92
2,sample_002.jpg,AI-Generated,0.88
3,sample_003.jpg,Real,0.95
4,sample_004.jpg,AI-Generated,0.91
5,sample_005.jpg,Real,0.85
EOF

echo "--- data.csv content (Version 1 - 5 records): ---"
cat data.csv

"$DVC_BIN" add data.csv
echo -e "\nContents of pointer file (data.csv.dvc):"
cat data.csv.dvc

git add .dvc data.csv.dvc .gitignore
git commit -m "feat(toy): track data.csv v1 with DVC (5 records)"
git tag -a v1.0 -m "Dataset version 1.0 (5 samples)"
echo "Git commit and tag v1.0 created."

# Step 4: Update data.csv to Version 2 and commit
echo -e "\n[STEP 4] Updating data.csv to Version 2 (adding 5 new records)..."
cat << 'EOF' >> data.csv
6,sample_006.jpg,AI-Generated,0.97
7,sample_007.jpg,Real,0.89
8,sample_008.jpg,AI-Generated,0.94
9,sample_009.jpg,Real,0.93
10,sample_010.jpg,AI-Generated,0.99
EOF

echo "--- data.csv content (Version 2 - 10 records): ---"
cat data.csv

"$DVC_BIN" add data.csv
echo -e "\nContents of updated pointer file (data.csv.dvc):"
cat data.csv.dvc

git add data.csv.dvc
git commit -m "feat(toy): update data.csv to v2 with DVC (10 records)"
git tag -a v2.0 -m "Dataset version 2.0 (10 samples)"
echo "Git commit and tag v2.0 created."

# Step 5: Version Switching Demonstration (Return to v1.0)
echo -e "\n[STEP 5] Demonstrating version restoration: Returning to v1.0..."
echo "Checking out Git tag v1.0..."
git checkout v1.0
echo "Running dvc checkout to restore data file content..."
"$DVC_BIN" checkout

echo "--- data.csv content after checkout to v1.0: ---"
cat data.csv
echo "Line count in data.csv (Expect 6: header + 5 rows): $(wc -l < data.csv | tr -d ' ')"

echo -e "\nReturning back to latest main branch (v2.0)..."
git checkout main
"$DVC_BIN" checkout
echo "Line count in data.csv on main (Expect 11: header + 10 rows): $(wc -l < data.csv | tr -d ' ')"

# Step 6: Configure Remote Storage
echo -e "\n[STEP 6] Configuring DVC remote storage..."
"$DVC_BIN" remote add -d toy_remote "$REMOTE_STORAGE"
git add .dvc/config
git commit -m "chore(toy): configure DVC remote storage"
"$DVC_BIN" remote list

# Step 7: Push Data to Remote Storage
echo -e "\n[STEP 7] Pushing tracked data to remote storage..."
"$DVC_BIN" push
echo "Contents of remote storage directory ($REMOTE_STORAGE):"
ls -la "$REMOTE_STORAGE"
find "$REMOTE_STORAGE" -type f

# Step 8: Delete Local Data and Pull Back
echo -e "\n[STEP 8] Deleting local data and restoring via dvc pull..."
rm -f data.csv
rm -rf .dvc/cache
echo "data.csv exists locally? $([ -f data.csv ] && echo 'YES' || echo 'NO (Successfully deleted)')"

echo "Executing dvc pull from remote storage..."
"$DVC_BIN" pull

echo "data.csv exists locally after dvc pull? $([ -f data.csv ] && echo 'YES (Successfully restored!)' || echo 'NO')"
echo "--- Restored data.csv content: ---"
cat data.csv

echo -e "\n[STEP 9] Git history log of the toy repository:"
git log --oneline --decorate --graph

echo -e "\n======================================================================"
echo "          DVC TOY DEMONSTRATION COMPLETED SUCCESSFULLY!               "
echo "======================================================================"
