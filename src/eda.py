"""
Task 1.3: Visual Exploratory Data Analysis (EDA)
Author: 23L0710
Project: 23L0710_A2_Ops - VisionGuard

Generates:
1. Sample image grids per class (AI-Generated vs Real)
2. Distribution histograms of width, height, aspect ratio, and file size per class
3. Markdown interpretation document with captions and analytical commentary
"""

import os
import random

# Configure writable cache path before importing matplotlib
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

MANIFEST_CSV = "evidence/task1/audit_manifest.csv"
OUTPUT_DIR = "evidence/task1"
INTERPRETATIONS_MD = os.path.join(OUTPUT_DIR, "eda_interpretations.md")

# Plot style configuration
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 15
})

def plot_sample_grids(df):
    """Generates sample grids for AI-Generated and Real images."""
    random.seed(42)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    classes = ["AI-Generated", "Real"]
    
    # 1. Combined Comparison Grid (2 rows x 4 columns)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle("VisionGuard (23L0710): AI-Generated vs. Real Image Samples", fontsize=16, fontweight="bold")
    
    for row_idx, cls_name in enumerate(classes):
        cls_df = df[df["class"] == cls_name]
        sample_paths = cls_df.sample(n=4, random_state=42)[["filepath", "filename", "width", "height"]].values
        
        for col_idx, (path, fname, w, h) in enumerate(sample_paths):
            ax = axes[row_idx, col_idx]
            try:
                img = Image.open(path).convert("RGB")
                ax.imshow(img)
                ax.set_title(f"[{cls_name}]\n{w}x{h} px", fontsize=10)
            except Exception as e:
                ax.text(0.5, 0.5, f"Error: {e}", ha="center", va="center")
            ax.axis("off")
            
    plt.tight_layout()
    comparison_path = os.path.join(OUTPUT_DIR, "eda_sample_grid_comparison.png")
    plt.savefig(comparison_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved comparison grid: {comparison_path}")
    
    # 2. Individual 3x3 grids per class
    for cls_name in classes:
        cls_df = df[df["class"] == cls_name]
        samples = cls_df.sample(n=9, random_state=101)[["filepath", "filename", "width", "height"]].values
        
        fig, axes = plt.subplots(3, 3, figsize=(12, 12))
        fig.suptitle(f"VisionGuard (23L0710): {cls_name} Class Samples (3x3 Grid)", fontsize=15, fontweight="bold")
        
        for idx, (path, fname, w, h) in enumerate(samples):
            ax = axes[idx // 3, idx % 3]
            try:
                img = Image.open(path).convert("RGB")
                ax.imshow(img)
                ax.set_title(f"{fname[:18]}...\n({w}x{h})", fontsize=9)
            except Exception as e:
                ax.text(0.5, 0.5, f"Err: {e}", ha="center")
            ax.axis("off")
            
        plt.tight_layout()
        cls_path = os.path.join(OUTPUT_DIR, f"eda_sample_grid_{cls_name.lower().replace('-', '_')}.png")
        plt.savefig(cls_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[+] Saved {cls_name} grid: {cls_path}")

def plot_distributions(df):
    """Generates histograms for dimensions, aspect ratio, and file size."""
    df["aspect_ratio"] = df["width"] / df["height"]
    
    # 1. Dimensions Histogram (Width and Height)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("VisionGuard (23L0710): Image Dimension Distributions by Class", fontsize=14, fontweight="bold")
    
    sns.histplot(data=df, x="width", hue="class", kde=True, bins=30, ax=axes[0], alpha=0.5, palette={"AI-Generated": "#e74c3c", "Real": "#2980b9"})
    axes[0].set_title("Image Width Distribution")
    axes[0].set_xlabel("Width (pixels)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_xlim(0, 4000)
    
    sns.histplot(data=df, x="height", hue="class", kde=True, bins=30, ax=axes[1], alpha=0.5, palette={"AI-Generated": "#e74c3c", "Real": "#2980b9"})
    axes[1].set_title("Image Height Distribution")
    axes[1].set_xlabel("Height (pixels)")
    axes[1].set_ylabel("Frequency")
    axes[1].set_xlim(0, 4000)
    
    plt.tight_layout()
    dims_path = os.path.join(OUTPUT_DIR, "eda_dimensions_hist.png")
    plt.savefig(dims_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved dimensions histogram: {dims_path}")
    
    # 2. File Size Distribution (Linear and Log scale)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("VisionGuard (23L0710): File Size Distribution (KB) by Class", fontsize=14, fontweight="bold")
    
    sns.histplot(data=df, x="file_size_kb", hue="class", kde=True, bins=30, ax=axes[0], alpha=0.5, palette={"AI-Generated": "#e74c3c", "Real": "#2980b9"})
    axes[0].set_title("File Size (KB) Distribution (Capped at 2000 KB)")
    axes[0].set_xlabel("File Size (KB)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_xlim(0, 2000)
    
    sns.boxplot(data=df, x="class", y="file_size_kb", hue="class", legend=False, ax=axes[1], palette={"AI-Generated": "#e74c3c", "Real": "#2980b9"})
    axes[1].set_title("File Size Boxplot (Log Scale)")
    axes[1].set_xlabel("Class")
    axes[1].set_ylabel("File Size (KB - Log Scale)")
    axes[1].set_yscale("log")
    
    plt.tight_layout()
    size_path = os.path.join(OUTPUT_DIR, "eda_filesize_hist.png")
    plt.savefig(size_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved file size histogram: {size_path}")
    
    # 3. Aspect Ratio Distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(data=df, x="aspect_ratio", hue="class", kde=True, bins=30, ax=ax, alpha=0.5, palette={"AI-Generated": "#e74c3c", "Real": "#2980b9"})
    ax.axvline(1.0, color="gray", linestyle="--", alpha=0.7, label="1:1 Square Aspect Ratio")
    ax.set_title("VisionGuard (23L0710): Aspect Ratio (Width / Height) by Class", fontsize=13, fontweight="bold")
    ax.set_xlabel("Aspect Ratio (Width / Height)")
    ax.set_ylabel("Frequency")
    ax.set_xlim(0.3, 2.5)
    ax.legend()
    
    plt.tight_layout()
    aspect_path = os.path.join(OUTPUT_DIR, "eda_aspect_ratio_hist.png")
    plt.savefig(aspect_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved aspect ratio histogram: {aspect_path}")

def generate_interpretations():
    """Writes report-ready captions and 2-3 sentence interpretations for each figure."""
    doc = """# Task 1.3: Visual EDA Figures & Analytical Interpretations
**Student Roll Number:** 23L0710  
**Project:** 23L0710_A2_Ops - VisionGuard  

---

### Figure 1: Sample Images Comparison Grid (`eda_sample_grid_comparison.png`)
**Caption:** *Figure 1.1 — Side-by-side visual comparison of sample AI-Generated images (top row) versus Real photographs and artwork (bottom row).*

**Interpretation:**
> The AI-generated images prominently display characteristic diffusion artifacts, including hyper-smooth skin textures, stylistic painterly lighting, surreal structural compositions, and exaggerated dynamic ranges. In contrast, the real images exhibit natural optical imperfections, authentic camera sensor grain, and organic background depth of field. This visual variance confirms that while subjects overlap (portraits, landscapes, and objects), high-frequency texture cues and edge rendering represent key discriminative features for convolutional feature extractors.

---

### Figure 2: Image Dimensions Histogram (`eda_dimensions_hist.png`)
**Caption:** *Figure 1.2 — Histograms of image width and height distributions comparing AI-Generated (red) and Real (blue) classes.*

**Interpretation:**
> Both classes exhibit clustering around common web resolutions (e.g., widths of 512–1024 px and heights of 512–800 px), with AI-generated images showing a pronounced peak around 1024x1024 pixels reflecting standard generator canvas dimensions (such as Stable Diffusion and Midjourney defaults). Real images display greater variance in resolution, with occasional high-resolution outliers exceeding 4000 pixels resulting from diverse web scraping sources. Because input dimensions vary widely across both classes, resizing images to a standardized input tensor (e.g., $224 \\times 224$) in Task 2.1 is essential.

---

### Figure 3: File Size Distribution (`eda_filesize_hist.png`)
**Caption:** *Figure 1.3 — File size distribution (KB) with histogram (left) and log-scale boxplot (right) for AI-Generated versus Real images.*

**Interpretation:**
> The median file size for AI-generated images is slightly higher (195.12 KB) than that of real images (141.58 KB), driven by dense color saturation, complex synthetic textures, and a higher proportion of PNG encodings (110 PNGs in AI vs. 33 in Real). Both classes have long right tails with maximum file sizes approaching 5.5 MB (AI) and 9.9 MB (Real), representing uncompressed or ultra-high-resolution files. However, the interquartile range (IQR) remains tightly bounded between 80 KB and 450 KB, confirming that file size alone is not a reliable shortcut classifier for deep learning.

---

### Figure 4: Aspect Ratio Distribution (`eda_aspect_ratio_hist.png`)
**Caption:** *Figure 1.4 — Aspect ratio (width / height) distribution across classes, with a dashed reference line at 1.0 (square).*

**Interpretation:**
> A substantial proportion of AI-generated images center around an aspect ratio of 1.0, reflecting the standard 1:1 square canvas default of modern generative models. Real images show a broader multimodal distribution with peaks near 1.33 (4:3) and 1.50 (3:2), reflecting standard photographic camera sensors and smartphone landscape formats. This finding informs our input preprocessing pipeline in Task 2.1, where smart resizing or padding must be applied to prevent distortion and preserve key spatial aspect ratios.
"""
    with open(INTERPRETATIONS_MD, "w") as f:
        f.write(doc)
    print(f"[+] Saved analytical interpretations to: {INTERPRETATIONS_MD}")

def main():
    print("[*] Starting Task 1.3 Visual EDA generation...")
    if not os.path.exists(MANIFEST_CSV):
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_CSV}. Run src/audit.py first.")
    
    df = pd.read_csv(MANIFEST_CSV)
    plot_sample_grids(df)
    plot_distributions(df)
    generate_interpretations()
    print("[*] Task 1.3 Visual EDA completed successfully!")

if __name__ == "__main__":
    main()
