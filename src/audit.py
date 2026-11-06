"""
Task 1.2: Dataset Audit Script
Author: 23L0710
Project: 23L0710_A2_Ops - VisionGuard

Audits the raw dataset in ./archive:
- Traverses all files per class
- Extracts: class, file size, width, height, colour mode, format, SHA-256 hash
- Identifies unreadable/corrupt images
- Detects exact duplicate images based on content hash
- Generates summary statistics (min, median, max of dimensions and file sizes)
- Exports manifest CSV and formatted summary table
"""

import os
import hashlib
import pandas as pd
import numpy as np
from PIL import Image

DATASET_ROOT = "archive"
OUTPUT_CSV = "evidence/task1/audit_manifest.csv"
OUTPUT_SUMMARY_MD = "evidence/task1/audit_summary.md"

def compute_sha256(filepath):
    """Computes SHA-256 content hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()

def audit_dataset(root_dir):
    records = []
    corrupt_files = []
    
    print(f"[*] Starting dataset audit on: '{root_dir}'...")
    
    # Supported common image extensions
    valid_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff"}

    for root, dirs, files in os.walk(root_dir):
        for file in files:
            # Skip OS metadata files like .DS_Store
            if file.startswith(".") or file.lower() == "thumbs.db":
                continue
            
            filepath = os.path.join(root, file)
            ext = os.path.splitext(file)[1].lower()
            
            # Determine class label based on directory name
            norm_path = os.path.normpath(filepath).split(os.sep)
            if any("aiart" in part.lower() for part in norm_path):
                class_label = "AI-Generated"
            elif any("realart" in part.lower() for part in norm_path):
                class_label = "Real"
            else:
                class_label = "Unknown"
                
            file_size_bytes = os.path.getsize(filepath)
            file_size_kb = file_size_bytes / 1024.0
            
            # Compute SHA-256 hash
            content_hash = compute_sha256(filepath)
            
            # Check image readability and inspect metadata
            is_corrupt = False
            error_msg = ""
            width, height, color_mode, img_format = None, None, None, None
            
            try:
                with Image.open(filepath) as img:
                    # Force load image data to catch truncation or decoding errors
                    img.load()
                    width, height = img.size
                    color_mode = img.mode
                    img_format = img.format if img.format else ext.replace(".", "").upper()
            except Exception as e:
                is_corrupt = True
                error_msg = str(e)
                corrupt_files.append((filepath, error_msg))
                
            records.append({
                "filename": file,
                "filepath": filepath,
                "class": class_label,
                "file_size_bytes": file_size_bytes,
                "file_size_kb": round(file_size_kb, 2),
                "width": width,
                "height": height,
                "color_mode": color_mode,
                "format": img_format,
                "sha256": content_hash,
                "is_corrupt": is_corrupt,
                "error": error_msg
            })
            
    df = pd.DataFrame(records)
    return df, corrupt_files

def generate_report(df, corrupt_files):
    # Detect exact duplicates using SHA-256
    hash_counts = df["sha256"].value_counts()
    duplicate_hashes = hash_counts[hash_counts > 1]
    df["is_duplicate"] = df["sha256"].isin(duplicate_hashes.index)
    total_duplicate_entries = df["is_duplicate"].sum()
    unique_duplicate_groups = len(duplicate_hashes)
    
    # Save full audit manifest CSV
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"[+] Full audit manifest saved to: {OUTPUT_CSV}")
    
    # Filter valid images for statistical calculations
    valid_df = df[~df["is_corrupt"]].copy()
    
    # Compute summary metrics per class and overall
    summary_rows = []
    classes = list(valid_df["class"].unique()) + ["Overall"]
    
    for c in classes:
        sub = valid_df if c == "Overall" else valid_df[valid_df["class"] == c]
        sub_all = df if c == "Overall" else df[df["class"] == c]
        
        summary_rows.append({
            "Class": c,
            "Total Files": len(sub_all),
            "Valid Images": len(sub),
            "Corrupt Files": sub_all["is_corrupt"].sum(),
            "Duplicates": sub_all["is_duplicate"].sum(),
            "Width (Min)": int(sub["width"].min()) if len(sub) else 0,
            "Width (Median)": float(sub["width"].median()) if len(sub) else 0,
            "Width (Max)": int(sub["width"].max()) if len(sub) else 0,
            "Height (Min)": int(sub["height"].min()) if len(sub) else 0,
            "Height (Median)": float(sub["height"].median()) if len(sub) else 0,
            "Height (Max)": int(sub["height"].max()) if len(sub) else 0,
            "Size KB (Min)": round(sub["file_size_kb"].min(), 2) if len(sub) else 0,
            "Size KB (Median)": round(sub["file_size_kb"].median(), 2) if len(sub) else 0,
            "Size KB (Max)": round(sub["file_size_kb"].max(), 2) if len(sub) else 0,
        })
        
    summary_df = pd.DataFrame(summary_rows)
    
    # Distinct Formats and Color Modes
    format_counts = valid_df.groupby(["class", "format"]).size().unstack(fill_value=0)
    mode_counts = valid_df.groupby(["class", "color_mode"]).size().unstack(fill_value=0)
    
    # Build Markdown Summary Report
    summary_md = f"""# Task 1.2: Dataset Audit Report
**Project:** 23L0710_A2_Ops - VisionGuard  
**Dataset Path:** `./{DATASET_ROOT}`  
**Total Images Audited:** {len(df)}  

---

## 1. Class Distribution and Image Integrity Summary

{summary_df.to_markdown(index=False)}

---

## 2. Image Formats Distribution per Class
{format_counts.to_markdown()}

---

## 3. Color Modes Distribution per Class
{mode_counts.to_markdown()}

---

## 4. Duplicate Files Analysis
- **Total Duplicate Image Instances:** {total_duplicate_entries}
- **Distinct Content Hashes with Duplicates:** {unique_duplicate_groups}
"""
    if unique_duplicate_groups > 0:
        summary_md += "\n### Duplicate Groups Detail:\n"
        dup_df = df[df["is_duplicate"]].sort_values("sha256")[["filename", "class", "sha256", "filepath"]]
        summary_md += dup_df.head(20).to_markdown(index=False)
        if len(dup_df) > 20:
            summary_md += f"\n\n*(Showing first 20 duplicates out of {len(dup_df)})*\n"
    else:
        summary_md += "\n- No duplicate files found.\n"

    summary_md += f"""
---

## 5. Corrupt / Unreadable Files Analysis
- **Total Corrupt Files Detected:** {len(corrupt_files)}
"""
    if len(corrupt_files) > 0:
        summary_md += "\n### Corrupt Files Detail:\n"
        for path, err in corrupt_files:
            summary_md += f"- `{path}`: {err}\n"
    else:
        summary_md += f"\n- All {len(df)} files successfully decoded with PIL (no corruption detected).\n"

    with open(OUTPUT_SUMMARY_MD, "w") as f:
        f.write(summary_md)
        
    print(f"[+] Summary markdown report saved to: {OUTPUT_SUMMARY_MD}")
    
    # Print to console for verification
    print("\n" + "=" * 70)
    print("                    DATASET AUDIT SUMMARY TABLE                      ")
    print("=" * 70)
    print(summary_df.to_string(index=False))
    print("=" * 70)
    print(f"\nCorrupt Files Found: {len(corrupt_files)}")
    print(f"Exact Duplicate Instances Found: {total_duplicate_entries}")
    print(f"Formats Found: {dict(valid_df['format'].value_counts())}")
    print(f"Color Modes Found: {dict(valid_df['color_mode'].value_counts())}")
    print("=" * 70 + "\n")

def main():
    df, corrupt_files = audit_dataset(DATASET_ROOT)
    generate_report(df, corrupt_files)

if __name__ == "__main__":
    main()
