# Task 1.3: Visual EDA Figures & Analytical Interpretations
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
> Both classes exhibit clustering around common web resolutions (e.g., widths of 512–1024 px and heights of 512–800 px), with AI-generated images showing a pronounced peak around 1024x1024 pixels reflecting standard generator canvas dimensions (such as Stable Diffusion and Midjourney defaults). Real images display greater variance in resolution, with occasional high-resolution outliers exceeding 4000 pixels resulting from diverse web scraping sources. Because input dimensions vary widely across both classes, resizing images to a standardized input tensor (e.g., $224 \times 224$) in Task 2.1 is essential.

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
