# Feature Detection and Matching

A **Qt6-based interactive desktop application** implementing classical computer vision algorithms entirely from scratch. This project demonstrates complete implementations of:

- **Harris Corner Detection** — Eigenvalue-based corner response with tunable parameters
- **Shi-Tomasi Corner Detection** — Minimum eigenvalue variant for more robust tracking
- **SIFT Feature Extraction** — Full scale-space pyramid with scale-invariant keypoints and 128-D descriptors
- **Descriptor Matching** — SSD and NCC-based correspondence with Lowe's ratio test

> **Key Point:** Every algorithm is implemented in pure C++. OpenCV is used only for matrix storage, I/O, and basic utilities—not for feature detection or matching.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Core Features](#core-features)
- [How It Works (Quick Start)](#how-it-works-quick-start)
- [Algorithm Implementations](#algorithm-implementations)
  - [Harris Corner Detector](#harris-corner-detector)
  - [Shi-Tomasi Corner Detector](#shi-tomasi-corner-detector)
  - [SIFT Feature Extraction](#sift-feature-extraction)
  - [Descriptor Matching](#descriptor-matching)
- [Project Architecture](#project-architecture)
- [Code Organization](#code-organization)
- [Dependencies](#dependencies)
- [Building & Running](#building--running)
- [Usage Guide](#usage-guide)
  - [Tab 1: Corner Detection](#tab-1--corner-detection)
  - [Tab 2: SIFT Extraction](#tab-2--sift-extraction)
  - [Tab 3: Feature Matching](#tab-3--feature-matching)
- [Performance Optimizations](#performance-optimizations)

---

## Project Overview

This desktop application provides an **interactive visual laboratory** for exploring classical feature detection and matching algorithms. Organized into three tabs, each representing a stage of the complete pipeline:

| Tab | Stage | Purpose |
|-----|-------|---------|
| **Tab 1** | Corner Detection | Extract keypoints using Harris or Shi-Tomasi with real-time parameter tuning |
| **Tab 2** | SIFT Extraction | Extract scale-invariant features and build a descriptor database |
| **Tab 3** | Feature Matching | Find correspondences between two images using SSD or NCC metrics |

All computation runs asynchronously via `QtConcurrent::run()` with `QFutureWatcher` to keep the GUI responsive even during intensive processing.

---

## Core Features

### Detection Algorithms
- ✅ **Harris corner detector** — `k`-controlled Harris response with thresholding and NMS
- ✅ **Shi-Tomasi corner detector** — Minimum eigenvalue variant for superior tracking performance

### SIFT (Scale-Invariant Feature Transform)
- ✅ **Gaussian pyramid** — Multi-octave, multi-scale smoothing for scale-space analysis
- ✅ **DoG pyramid** — Difference-of-Gaussians for efficient extrema detection
- ✅ **Extrema detection** — 3-D peak/valley detection with contrast thresholding
- ✅ **Orientation assignment** — 36-bin gradient histogram for rotation-invariance
- ✅ **Descriptor computation** — 128-dimensional local feature descriptor (4×4×8 grid)

### Matching
- ✅ **SSD matching** — Sum of Squared Differences with Lowe's ratio test
- ✅ **NCC matching** — Normalized Cross-Correlation with similarity thresholding
- ✅ **ROI-based filtering** — Draw rectangles to restrict matching to regions of interest

### Interactive Features
- ✅ **Zoomable/pannable image panels** — Scroll wheel to zoom, Shift+drag to pan
- ✅ **ROI drawing** — Click-drag to define search regions for matching
- ✅ **Result caching** — Automatic skip of redundant computation when parameters unchanged
- ✅ **Per-stage timing** — Wall-clock measurements for every pipeline stage
- ✅ **Real-time UI feedback** — Progress status and performance metrics in the status bar

---

## How It Works (Quick Start)

### The Complete Pipeline

```
Load Image → Preprocess → Detect Features → Extract Descriptors → Match
              ↓              ↓                  ↓                    ↓
           Grayscale    Harris/Shi-Tomasi    SIFT Pyramid    SSD/NCC Matching
           + Blur       + Tensor + Response  + DoG Extrema   + Lowe's Ratio Test
                        + NMS + Threshold    + Orientation
                                             + 128-D Descriptor
```

### Typical Workflow

1. **Load an image** via Tab 1
2. **Tune corner detection parameters** (Harris k, threshold, NMS)
3. **Switch to Tab 2** to see SIFT keypoints overlaid on the image
4. **Load a second image** on Tab 3 and draw ROIs to restrict matching
5. **Run matching** to find correspondences between the two images

---

## Algorithm Implementations

### Harris Corner Detector

The detection pipeline consists of **8 sequential stages**, each operating on cached intermediate results:

#### 1. **Grayscale Conversion**
- Input: RGB/BGR image (CV_8UC3) or already-grayscale
- Formula: Gray = 0.299R + 0.587G + 0.114B (ITU-R BT.601 standard)
- Output: CV_32FC1 (float32) for numerical stability in subsequent stages

#### 2. **Gaussian Smoothing**
- Separable 5-tap Gaussian blur: kernel = [1, 4, 6, 4, 1]/16
- Method: Convolve horizontally, then vertically (O(n·k) vs O(n·k²) complexity)
- Purpose: Reduce noise and suppress high-frequency gradients

#### 3. **Gradient Computation**
- Separable Sobel operators: smoothing kernel [1, 2, 1], difference kernel [-1, 0, 1]
- Outputs: Gx (horizontal gradient), Gy (vertical gradient)
- Cached products: Ix² (Gx²), Iy² (Gy²), IxIy (Gx·Gy) for structure tensor

#### 4. **Structure Tensor**
- Accumulate squared gradients with Gaussian smoothing (5-tap kernel)
- Produces: M = [[Sxx, Sxy], [Sxy, Syy]] at each pixel
- M captures local image structure and curvature

#### 5. **Harris Response**
$$R = \det(M) - k \cdot \mathrm{trace}(M)^2$$
where:
  - $\det(M) = S_{xx} S_{yy} - S_{xy}^2$ (product of eigenvalues)
  - $\mathrm{trace}(M) = S_{xx} + S_{yy}$ (sum of eigenvalues)
  - $k$ ≈ 0.04 (sensitivity parameter; higher k suppresses more edge-like structures)

#### 6. **Thresholding**
- Normalize response to [0, 255]
- Zero out pixels with response < user-specified threshold
- Purpose: Select strong corners and reject weak responses

#### 7. **Non-Maximum Suppression (NMS)**
- For each non-zero response, check if it's a local maximum within a square neighborhood
- Suppresses all non-maximum pixels to zero
- Result: One peak per corner cluster

#### 8. **Corner Extraction**
- Scan thresholded response map
- Extract (x, y) coordinates of all non-zero pixels
- Render as colored circles on the original image (orange-red for Harris)

### Shi-Tomasi Corner Detector

Identical to Harris up through the **Structure Tensor** stage. The response is replaced by the **minimum eigenvalue**:

$$R = \frac{1}{2} \left[ (S_{xx} + S_{yy}) - \sqrt{(S_{xx} - S_{yy})^2 + 4S_{xy}^2} \right]$$

**Advantage:** Shi-Tomasi corners are often more stable for feature tracking because the response is directly related to trackability.

Corners detected via Shi-Tomasi are rendered with a different color (green) for visual distinction.

### SIFT Feature Extraction

Complete scale-space feature detection and descriptor extraction:

#### **Stage 1: Gaussian Pyramid Construction**
- **Input:** Float32 grayscale image (normalized to [0, 1])
- **Process:**
  - For octave $o = 0, 1, \ldots, O-1$:
    - Blur base image with $S+3$ Gaussian levels at progressively increasing sigma
    - Scale by $\sigma_s = \sigma_0 \cdot k^s$ where $k = 2^{1/S}$ (S = scales per octave)
    - Downsample by 2× to seed the next octave (using `cv::INTER_AREA`)
  - Base sigma: σ₀ = 1.6; inter-octave scale: 2×; inter-scale ratio: $k = 2^{1/S}$
- **Output:** Pyramid of S+3 levels per octave across O octaves

#### **Stage 2: DoG (Difference-of-Gaussians) Pyramid**
- **Process:** Subtract adjacent Gaussian levels: DoG = G[s+1] - G[s]
- **Result:** S+2 DoG images per octave (enabling 3-D extrema search across S interior layers)
- **Purpose:** Used to detect scale-space keypoint candidates

#### **Stage 3: Extrema Detection**
- **Search:** For each interior DoG pixel, check all 26 neighbors in the 3×3×3 scale-space cube
- **Condition:** Candidate is kept if:
  1. It's a strict local maximum OR minimum (compared to all 26 neighbors)
  2. Its absolute DoG value ≥ contrastThreshold
- **Scale Mapping:** Keypoint coordinates scaled back to image space: (x, y) = (col·2^oct, row·2^oct)
- **Parallelization:** OpenMP across rows; thread-local lists aggregated at the end
- **Typical Yield:** ~1000–5000 keypoints per MegaPixel (depends on contrast threshold)

#### **Stage 4: Orientation Assignment**
- **Local Gradient Histogram:**
  - Compute 36-bin gradient-magnitude histogram (10° per bin) within a 3σ radius
  - Weight each bin by Gaussian window (σ = 1.5 × keypoint_scale)
- **Dominant Orientation:** The peak bin direction becomes `kp.angle`
- **Result:** Each keypoint is rotation-invariant

#### **Stage 5: Descriptor Computation**
- **Spatial Grid:** 4×4 cells centered at the keypoint, each cell spanning D/4 pixels (D = 30 pixels ≈ 1.5 × window radius)
- **Orientation Grid:** 8 bins per cell (45° per bin)
- **Interpolation:** Trilinear (spatial and angular) to smooth descriptor across cell boundaries
- **Gaussian Weighting:** Envelope of σ = 1.0 cell size to emphasize central region
- **Post-processing:**
  1. L2-normalize
  2. Threshold at 0.2 (Lowe's rule; clamps large values)
  3. L2-normalize again
- **Output:** 128-dimensional feature vector per keypoint
- **Parallelization:** OpenMP across keypoints

### Descriptor Matching

Runs in an async QtConcurrent thread pool; keypoints are pre-filtered by ROI bounds.

**SSD (Sum of Squared Differences)**
- For each query descriptor, find the two nearest neighbors in the train set via brute-force L2 distance
- **Lowe's Ratio Test:** Match is accepted if $\sqrt{d_{\text{best}}} < \text{ratio} \times \sqrt{d_{\text{2nd best}}}$
  - Typical ratio: 0.7–0.8 (default: 0.7)
  - Suppresses ambiguous matches where nearest neighbor is not clearly better than second-nearest

**NCC (Normalized Cross-Correlation)**
- Mean-center each descriptor: $d'_i = d_i - \bar{d}$
- Compute dot product and normalize by L2 norms: $\text{NCC} = \frac{d_1' \cdot d_2'}{||d_1'|| \cdot ||d_2'||}$
- Match accepted if NCC ≥ minCorr (scaled 0–255 slider maps to 0.0–1.0)
- **Advantage:** Robust to global brightness shifts; metric is interpretable as a correlation coefficient

**Rendering:** Matches drawn as colored lines connecting corresponding keypoints across a side-by-side image composite.

---

## Project Architecture

### Three-Column Design

```
┌─────────────────────────────────────────────────────────────┐
│                    MainWindow (Qt)                          │
├─────────────────────────────────────────────────────────────┤
│  Tab 1: Corner Detection  │  Tab 2: SIFT  │  Tab 3: Matching │
├────────────────-────────────────────────────────────────────┤
│       Image Cache (Image struct with named cv::Mats)        │
├─────────────────────────────────────────────────────────────┤
│ Harris │ Shi-Tomasi │ SIFT Processor │ Matchers (SSD/NCC)   │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Image Loading** → `Image` struct created; original mat stored
2. **Preprocessing** → Intermediate results cached with string keys
3. **Feature Detection** → Harris/Shi-Tomasi stairs run, keypoints extracted
4. **Descriptor Extraction** → SIFT pyramid built, descriptors computed
5. **Matching** → SSD or NCC compares descriptors; results rendered

The `Image` struct's cache ensures that:
- Switching between Harris and Shi-Tomasi reuses grayscale, gradient, and tensor stages
- Re-running with identical parameters skips computation (checks cache keys)
- Later stages never recompute what earlier stages cached

---

## Code Organization

### Source Code Structure

```
src/
├── main.cpp                        Application entry point; creates MainWindow
├── mainwindow.cpp                  Window lifecycle, destructor, cleanup
├── mainwindow_setup.cpp            Signal/slot connections, UI initialization
├── mainwindow_shared.cpp           Image loading, mat↔pixmap conversion helpers
├── mainwindow_tab1.cpp             Tab 1: corner detection logic & visualization
├── mainwindow_tab2.cpp             Tab 2: SIFT extraction logic & visualization
├── mainwindow_tab3.cpp             Tab 3: matching UI & result rendering
│
├── processors/
│   ├── harris/
│   │   ├── grayscale.cpp           BGR → float32 grayscale (ITU-R BT.601)
│   │   ├── gaussian.cpp            Separable 5-tap Gaussian smoothing
│   │   ├── gradient.cpp            Sobel gradients (Gx, Gy) → squared products
│   │   ├── strcutre_tensor.cpp     Smooth squared gradients → structure tensor M
│   │   ├── harris_response.cpp     Compute R = det(M) − k·trace²(M)
│   │   ├── shi_tomasi.cpp          Compute R = min eigenvalue of M
│   │   ├── threshold.cpp           Normalize & threshold response; extract corners
│   │   └── nms.cpp                 Non-maximum suppression over local window
│   └── harris_main.cpp             Orchestration: runs all 8 stages in sequence
│
├── sift/
│   ├── sift_pyramid.cpp            Build Gaussian & DoG pyramids
│   ├── sift_extrema_orientation.cpp 3-D extrema detection in DoG; orientation assignment
│   ├── sift_descriptor.cpp         Compute 4×4×8 = 128-D descriptors
│   └── sift_extract.cpp            Top-level pipeline orchestration (extractFeatures)
│
├── SiftCore.cpp                    Translation unit; defines SiftProcessor methods
├── Timer.cpp                       High-resolution timer implementation
│
├── io/
│   └── image_handler.cpp           cv::imread wrapper → Image struct
│
├── utils/
│   └── utils.cpp                   Border-reflection helper; explicit template instantiations
│
└── widgets/
    ├── interactive_label.cpp       QLabel subclass for ROI drawing (mouse events)
    └── zoomable_image.cpp          QLabel subclass for image zoom/pan (scroll + drag)

include/
├── mainwindow.h                    Declares MainWindow class & signal handlers
├── model/image.hpp                 Image struct: source mat + named cache (store/get/has)
├── processors/harris_main.hpp      Declares applyHarris() entry point
├── processors/harris/*.hpp         Headers for each Harris stage
├── SiftCore.hpp                    Declares cv_assign::SiftProcessor class
├── Timer.hpp                       Declares ExecutionTimer utility
├── io/image_handler.hpp            Declares loadImage() helper
├── utils/utils.hpp                 Declares convolveH/convolveV & reflectIndex
├── utils/utils_impl.hpp            Template implementations (included by utils.hpp)
└── widgets/
    ├── interactive_label.h         Declares InteractiveLabel class
    └── zoomable_image.hpp          Declares ZoomableLabel class

ui/
└── mainwindow.ui                   Qt Designer UI definition file
```

### File Count & Scope

- **Total implementation files:** 26 (.cpp)
- **Total header files:** 18 (.hpp/.h)
- **Total lines of code:** ~5000+ (algorithms) + ~2000+ (UI) = ~7000+ LOC
- **Documentation:** All files include Doxygen-style docstrings and inline comments

---

## Dependencies

| Dependency | Version | Purpose | Used For |
|---|---|---|---|
| **Qt** | 6.x | GUI framework | MainWindow, tabs, image display, async threading |
| **OpenCV** | 4.x | Matrix library & I/O | cv::Mat, cv::imread, cv::parallel_for_, drawing utilities |
| **OpenMP** | 4.5+ | Parallelism | Multi-threaded SIFT detection and descriptor computation |
| **CMake** | 3.16+ | Build system | Project configuration and compilation |
| **C++ Standard** | C++17 | Language features | std::string_view, std::optional, ranges, structured bindings |

> **Note:** OpenCV's high-level API functions (`cv::SIFT`, `cv::cornerHarris`, `cv::goodFeaturesToTrack`, `cv::BFMatcher`) are intentionally NOT used. All algorithms are custom implementations.

---

## Usage Guide

### Tab 1: Corner Detection

Extract and visualize corner keypoints using Harris or Shi-Tomasi detection.

**Workflow:**
1. Click **Load Image 1** and select any image file (PNG/JPG/BMP/TIFF)
2. Choose **Harris** or **Shi-Tomasi** from the detector dropdown
3. Adjust parameters:
   - **Threshold** (0–255): Higher = fewer, stronger corners
   - **Harris k** (0.01–0.10): Sensitivity parameter (higher suppresses edge-like structures)
   - **NMS Window** (1–20): Half-width for non-maximum suppression
4. Click **Detect Corners** to run the full 8-stage pipeline
5. Results are displayed with:
   - **Orange-red circles:** Harris corners
   - **Green circles:** Shi-Tomasi corners
6. Status bar shows per-stage timing and corner count

**Tips:**
- Start with threshold ≈ 100 and NMS window ≈ 3–5
- Lower Harris k (0.04) gives more corners; higher k (0.08) gives fewer, stronger corners
- Results are cached; re-run with same parameters for instant results

---

### Tab 2: SIFT Extraction

Extract scale-invariant keypoints and compute 128-D feature descriptors.

**Workflow:**
1. Load Image 1 (from Tab 1 or new image)
2. Configure display:
   - Toggle **Show Corners** checkbox to overlay corner detections
   - Select corner detector type from dropdown
3. Adjust SIFT parameters:
   - **Contrast Threshold** (0.001–0.030): Lower = more keypoints
   - **Octaves** (1–8): Depth of scale-space pyramid
   - **Scales per Octave** (2–6): Resolution within each octave
4. Click **Run SIFT** to extract features
5. Keypoints displayed as circles with:
   - **Radius:** Keypoint scale
   - **Line orientation:** Dominant gradient direction
6. Status bar shows keypoint count and extraction time

**Tips:**
- Use octaves=3–4 and scales=3–4 for balanced performance
- Contrast threshold around 0.01–0.02 is typical
- SIFT features are automatically cached for Tab 3

---

### Tab 3: Feature Matching

Find correspondences between two images using their SIFT descriptors.

**Workflow:**
1. Click **Load Image 2** to open the target image
2. Define regions of interest (ROI) on Image 1:
   - **Click and drag** on the left panel to draw rectangles
   - Multiple ROIs can be added; **Undo Last ROI** or **Clear ROI** to modify
   - Only keypoints inside ROI(s) are matched
3. Choose matching method:
   - **SSD** (Sum of Squared Differences): Faster, good for general matching
   - **NCC** (Normalized Cross-Correlation): Robust to lighting changes
4. Adjust **Threshold**:
   - For SSD: Lowe's ratio threshold (lower = stricter)
   - For NCC: Minimum correlation score (higher = stricter)
5. Click **Run Match** to compute correspondences
6. Results shown as:
   - Colored lines connecting matching keypoints
   - Circled matched keypoints on both images
   - Status bar displays match count and timing

**Tips:**
- Use NCC for images with different lighting
- Use SSD for speed-critical applications
- RODs are useful for constraining matches to specific regions
- Lowe's ratio ≈ 0.7 is standard for feature matching

---

### Image Navigation

**All image panels support:**
- **Scroll Wheel:** Zoom in/out (1× to 8×)
- **Shift + Left Drag:** Pan zoomed image
- **Middle-Click Drag:** Pan (alternative method)

---

## Performance Optimizations

### Algorithm-Level

1. **Separable Convolution** — O(n·k) instead of O(n·k²) complexity
2. **Multi-Threading** — cv::parallel_for_ and OpenMP across computation stages
3. **SIMD Vectorization** — GCC ivdep pragma for auto-vectorization
4. **Result Caching** — Intermediate stages reused across runs
5. **Smart Border Handling** — BORDER_REFLECT_101 padding avoids boundary checks

### System-Level

1. **Auto-Downsampling** — Images > 1024 px are automatically reduced
2. **Async UI** — All long operations run in background threads
3. **Reference-Counted Memory** — cv::Mat sharing avoids data copying

---