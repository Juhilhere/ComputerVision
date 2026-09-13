# Find Homography So Dope — Complete Technical Deep Dive

> A line-by-line explanation of every science, math, computer vision, and engineering concept used in this repository. Written so that any human or AI agent can fully understand what was built and why.

---

## Table of Contents

1. [What Is the Problem?](#1-what-is-the-problem)
2. [What Is a Homography? (The Core Math)](#2-what-is-a-homography-the-core-math)
3. [The Dataset Structure](#3-the-dataset-structure)
4. [The Evaluation Metric (How Submissions Are Scored)](#4-the-evaluation-metric)
5. [The Solution Architecture (Big Picture)](#5-the-solution-architecture)
6. [Concept 1: SIFT — Scale-Invariant Feature Transform](#6-sift)
7. [Concept 2: RootSIFT — Hellinger Kernel Normalization](#7-rootsift)
8. [Concept 3: CLAHE — Contrast Limited Adaptive Histogram Equalization](#8-clahe)
9. [Concept 4: Brute-Force Matching + Lowe's Ratio Test](#9-matching)
10. [Concept 5: USAC_MAGSAC — Robust Homography Estimation](#10-usac-magsac)
11. [Concept 6: Geometric Guardrails (Validation Checks)](#11-guardrails)
12. [The 4-Stage Cascade Pipeline (predict_module.py)](#12-pipeline)
13. [The Identity Fallback Prior — Why It's Smart](#13-identity-fallback)
14. [Evaluation Script (evaluate.py)](#14-evaluate)
15. [Submission Generation (generate_submission.py)](#15-generate)
16. [Submission Validation (validate_submission.py)](#16-validate)
17. [The Solution Notebook (solution_notebook.ipynb)](#17-notebook)
18. [Complete Concept Map](#18-concept-map)

---

## 1. What Is the Problem?

You are given **pairs of images** taken from the same scene. The two images in each pair differ because either:
- The **camera moved** (viewpoint change — rotation, translation, zoom), or
- The **lighting changed** (illumination variation — same camera position, different exposure or light source)

**Goal**: For each pair, predict the **3×3 homography matrix H** that maps every pixel coordinate `(x₁, y₁)` in Image 1 to its corresponding pixel coordinate `(x₂, y₂)` in Image 2.

**Why this matters in the real world**: Homography estimation is foundational in:
- Panorama stitching (aligning overlapping photos)
- Augmented reality (overlaying virtual objects on planar surfaces)
- Camera calibration
- Visual SLAM (Simultaneous Localization and Mapping)
- Document scanning and rectification
- Sports analytics (mapping a court/field to a top-down view)

---

## 2. What Is a Homography? (The Core Math)

### Projective Geometry Background

A **homography** (also called a **projective transformation** or **perspective transformation**) is a mapping between two images of the same planar surface, or between two images related by a pure camera rotation (no translation).

Mathematically, it is represented as a **3×3 matrix** that operates on **homogeneous coordinates**:

```
┌ x₂' ┐   ┌ h₁₁  h₁₂  h₁₃ ┐   ┌ x₁ ┐
│ y₂' │ = │ h₂₁  h₂₂  h₂₃ │ × │ y₁ │
└ w₂' ┘   └ h₃₁  h₃₂  h₃₃ ┘   └  1 ┘
```

The actual 2D pixel coordinates in Image 2 are recovered by **perspective division**:
```
x₂ = x₂' / w₂'
y₂ = y₂' / w₂'
```

### Why Homogeneous Coordinates?

In ordinary (Euclidean) coordinates, perspective projection is a **non-linear** operation — dividing by depth makes it impossible to express as a simple matrix multiply. Homogeneous coordinates add a third coordinate `w` so that the point `(x, y)` in 2D is represented as `(x, y, 1)` in 3D. This lifts the non-linearity into a linear matrix multiplication followed by a single division step.

### Key Properties of H

| Property | Explanation |
|----------|-------------|
| **8 degrees of freedom** | H is 3×3 = 9 entries, but it's defined only **up to scale** (multiplying every element by the same constant doesn't change the mapping). So only 8 independent parameters. |
| **h₃₃ = 1 normalization** | The competition fixes `h₃₃ = 1` to remove the scale ambiguity. You only predict the other 8 entries: `h₁₁, h₁₂, h₁₃, h₂₁, h₂₂, h₂₃, h₃₁, h₃₂`. |
| **Invertible** | A valid homography must have `det(H) ≠ 0`. If the determinant is zero, the transformation collapses the image to a line or point. |
| **Preserves straight lines** | Lines in Image 1 map to lines in Image 2 (not curves). This is a defining property of projective transformations. |
| **Does NOT preserve parallelism** | Unlike affine transforms, parallel lines can converge after a homography (think of railroad tracks converging toward a vanishing point). |

### Subgroups of H

The 3×3 homography matrix encodes several nested geometric transformations:

```
Euclidean ⊂ Similarity ⊂ Affine ⊂ Projective (Homography)

Euclidean:    rotation + translation               (3 DOF)
Similarity:   + uniform scaling                    (4 DOF)
Affine:       + non-uniform scaling + shear         (6 DOF)
Projective:   + perspective distortion              (8 DOF)
```

The bottom row `[h₃₁, h₃₂, 1]` controls the **perspective effect**. When `h₃₁ = h₃₂ = 0`, H reduces to an affine transformation. When additionally the top-left 2×2 block is a scaled rotation matrix, it reduces to a similarity.

### The Identity Matrix = No Transformation

```
    ┌ 1  0  0 ┐
I = │ 0  1  0 │  →  maps every point (x, y) to itself
    └ 0  0  1 ┘
```

This is the homography that says "both images are already perfectly aligned" — which is exactly what happens when only the lighting changed and the camera didn't move.

---

## 3. The Dataset Structure

```
data/
├── train/
│   ├── scene_001/   (6 images: 1.png, 2.png, 3.png, 4.png, 5.png, 6.png)
│   ├── scene_002/
│   ├── ...
│   └── scene_038/   (28 scenes total)
└── test/
    ├── scene_004/
    ├── scene_008/
    ├── ...
    └── scene_040/   (12 scenes total)
```

### How Pairs Are Formed

Each scene has 6 images. Image 1 is always the **reference**. Pairs are formed as:
- `(image_1, image_2)`, `(image_1, image_3)`, ..., `(image_1, image_6)` → **5 pairs per scene**

So:
- **Train**: 28 scenes × 5 pairs = **140 pairs** (with known ground-truth H)
- **Test**: 12 scenes × 5 pairs = **60 pairs** (H is hidden)

### CSV Format

**train.csv** — Has the 8 elements of the ground-truth homography:
```
pair_id,image_1,image_2,h11,h12,h13,h21,h22,h23,h31,h32
```

**test.csv** — Only has pair ID and image paths (no H):
```
pair_id,image_1,image_2
```

**submission.csv** — What you predict:
```
pair_id,h11,h12,h13,h21,h22,h23,h31,h32
```

### Critical EDA Discovery

~50% of training scenes are **pure illumination changes** where `H = I` (identity matrix). This means the camera never moved — only the lighting changed. This is a crucial insight because:
1. The identity matrix is already correct for half the data
2. For the other half (viewpoint changes), a wrong prediction that's worse than identity actually *hurts* the score
3. Therefore, using identity as a **fallback** when feature matching fails is not just safe — it's optimal

---

## 4. The Evaluation Metric

### Why Not Compare Matrices Element-by-Element?

Because H is only defined up to scale, and small changes in different elements have vastly different geometric impacts. For example:
- A tiny change in `h₁₃` or `h₂₃` (translation) shifts pixels by a few units
- The same tiny change in `h₃₁` or `h₃₂` (perspective) can shift pixels by hundreds of units in some regions

### Geometric Reprojection Error

Instead, we measure how well the *predicted* H aligns points compared to the *true* H:

**Step-by-step process** (implemented in `evaluate.py`):

1. **Choose 5 canonical test points** in normalized `[0, 1]` coordinates:
   ```
   (0, 0)      — top-left corner
   (1, 0)      — top-right corner
   (1, 1)      — bottom-right corner
   (0, 1)      — bottom-left corner
   (0.5, 0.5)  — center
   ```

2. **Convert to pixel coordinates** in Image 1:
   ```
   pixel_point = normalized_point × (W₁, H₁)
   ```
   For example, `(1, 1)` in a 640×480 image becomes `(640, 480)`.

3. **Warp with both** the predicted H and the ground-truth H:
   ```
   predicted_pixel_2 = H_pred × pixel_1   (then divide by w)
   groundtruth_pixel_2 = H_gt × pixel_1   (then divide by w)
   ```

4. **Convert warped points to normalized coordinates** in Image 2:
   ```
   norm_point_2 = pixel_point_2 / (W₂, H₂)
   ```

5. **Compute Euclidean distance** between predicted and ground-truth normalized points, then average over the 5 points:
   ```
   error = mean(‖pred_norm₂ − gt_norm₂‖₂)   for all 5 points
   ```

6. **Final competition score** (Kaggle Leaderboard):
   ```
   Score = 100 × max(0, 1 − error / 0.2)
   ```
   - Perfect prediction → error = 0 → Score = 100
   - Error ≥ 0.2 → Score = 0
   - Lower error = higher score

### Why Normalized Coordinates?

Normalization by image dimensions makes the metric **resolution-independent**. A 5-pixel error on a 100×100 image is much worse than a 5-pixel error on a 4000×3000 image. Normalization ensures the error reflects the *fraction* of the image that's misaligned, not the absolute pixel count.

---

## 5. The Solution Architecture (Big Picture)

The solution follows the **classical feature-based homography estimation pipeline**:

```
┌────────────┐     ┌─────────────────┐     ┌───────────────┐     ┌──────────────┐
│  Load both │ ──▸ │  Detect features│ ──▸ │ Match features│ ──▸ │ Estimate H   │
│  images    │     │  (keypoints +   │     │ (BFMatcher +  │     │ (USAC_MAGSAC │
│  (gray)    │     │   descriptors)  │     │  Lowe ratio)  │     │  + RANSAC)   │
└────────────┘     └─────────────────┘     └───────────────┘     └──────┬───────┘
                                                                        │
                                                                        ▼
                                                                 ┌──────────────┐
                                                                 │ Validate H   │
                                                                 │ (guardrails) │
                                                                 └──────┬───────┘
                                                                        │
                                                              ┌─────────┴─────────┐
                                                              │ Valid?            │
                                                              │  YES → return H   │
                                                              │  NO  → try next   │
                                                              │        stage or   │
                                                              │        fallback   │
                                                              │        to I₃ₓ₃    │
                                                              └───────────────────┘
```

This pipeline is executed in **4 progressively more aggressive stages**, each with different detector sensitivity, preprocessing, descriptor transforms, and matching thresholds.

---

## 6. SIFT — Scale-Invariant Feature Transform

> **Paper**: David Lowe, "Distinctive Image Features from Scale-Invariant Keypoints" (2004)

SIFT is the most important algorithm in this solution. It detects **keypoints** (interesting, repeatable points) in images and computes a **128-dimensional descriptor** for each keypoint that captures the local image appearance.

### How SIFT Works (Step by Step)

#### Step 1: Scale-Space Construction (Gaussian Pyramid)

SIFT builds a **scale-space** by progressively blurring the image with Gaussian filters of increasing σ (standard deviation):

```
Original image → blur(σ₁) → blur(σ₂) → blur(σ₃) → ...
```

This creates multiple "octaves" (each at half the resolution of the previous), with multiple blur levels within each octave. The idea: a feature that exists at multiple scales is more robust than one that only appears at one scale.

#### Step 2: Difference-of-Gaussians (DoG)

Adjacent blur levels are subtracted to create **Difference-of-Gaussian** images:

```
DoG(σ) = G(σ₂) − G(σ₁)    where σ₂ = k × σ₁
```

DoG is a **band-pass filter** that approximates the Laplacian of Gaussian (LoG) — it highlights regions where intensity changes rapidly (edges, corners, blobs).

#### Step 3: Keypoint Detection (Extrema Finding)

A pixel is a candidate keypoint if it is a **local extremum** (maximum or minimum) compared to its 26 neighbors: 8 in the same DoG image + 9 above + 9 below in scale.

#### Step 4: Keypoint Refinement

Candidates are refined using:
- **Sub-pixel interpolation** (Taylor expansion) to find the exact position and scale
- **Contrast threshold** (`contrastThreshold`): Rejects low-contrast candidates (they're unstable under noise). The code uses three settings: 0.015 (standard), 0.008 (sensitive), 0.005 (aggressive)
- **Edge threshold** (`edgeThreshold=10`): Rejects keypoints on edges using the ratio of principal curvatures (Hessian eigenvalue ratio). Edge points have poor localization along the edge direction.

#### Step 5: Orientation Assignment

For each surviving keypoint, SIFT computes a **dominant orientation** from the gradient directions in its neighborhood. This makes the descriptor **rotation-invariant**: the local patch is rotated to align with this orientation before computing the descriptor.

#### Step 6: Descriptor Computation (128-D Vector)

A 16×16 pixel window around the keypoint (aligned to the dominant orientation) is divided into a 4×4 grid of cells. In each cell, an 8-bin **histogram of gradient orientations** is computed. This gives 4 × 4 × 8 = **128 values**, which are normalized to unit length.

### The Three SIFT Configurations in This Code

| Name | `nfeatures` | `contrastThreshold` | Purpose |
|------|-------------|---------------------|---------|
| `sift_standard` | 5,000 | 0.015 | Default — fast, finds only the most confident keypoints |
| `sift_sensitive` | 8,000 | 0.008 | Finds weaker features in low-contrast regions |
| `sift_aggressive` | 10,000 | 0.005 | Maximum sensitivity — finds almost everything, including noise |

The **cascade strategy**: start with the fastest, most precise detector. If it doesn't find enough matches, progressively relax the detection threshold to find more (potentially noisier) keypoints.

---

## 7. RootSIFT — Hellinger Kernel Normalization

> **Paper**: Arandjelović & Zisserman, "Three things everyone should know to improve object retrieval" (CVPR 2012)

### The Problem with Raw SIFT Descriptors

Standard SIFT descriptors are compared using **Euclidean (L2) distance**. But research showed that using the **Hellinger kernel** (a measure from probability theory) dramatically improves matching accuracy — especially for illumination-invariant matching.

### The RootSIFT Transform

Rather than changing the distance metric (which would require modifying OpenCV internals), RootSIFT transforms the descriptors so that **L2 distance on the transformed descriptors = Hellinger distance on the originals**:

```python
# Step 1: L1-normalize each descriptor
des_norm = des / (‖des‖₁ + ε)     # each row sums to 1 (like a probability distribution)

# Step 2: Element-wise square root
des_sqrt = √(des_norm)              # the Bhattacharyya coefficient step

# Step 3: L2-normalize the result
des_final = des_sqrt / (‖des_sqrt‖₂ + ε)
```

### Why This Works

The Hellinger kernel between two probability distributions `p` and `q` is:

```
H(p, q) = √(1 − Σ √(pᵢ × qᵢ))
```

When you L1-normalize SIFT descriptors (making them probability distributions) and take the square root, the L2 distance between the resulting vectors is monotonically related to the Hellinger distance. The Hellinger metric is more robust to the heavy-tailed gradient distributions that SIFT produces, reducing the influence of high-magnitude bins and giving more weight to low-magnitude but informative dimensions.

### In the Code

```python
def rootsift(des):
    des_norm = des / (np.linalg.norm(des, axis=1, ord=1, keepdims=True) + 1e-7)  # L1 norm
    des_sqrt = np.sqrt(des_norm)                                                   # √
    des_l2   = des_sqrt / (np.linalg.norm(des_sqrt, axis=1, ord=2, keepdims=True) + 1e-7)  # L2 norm
    return des_l2.astype(np.float32)
```

The `1e-7` epsilon prevents division by zero for all-zero descriptors.

---

## 8. CLAHE — Contrast Limited Adaptive Histogram Equalization

### The Problem

Some image pairs have **drastic lighting differences** (e.g., one brightly lit, one in shadow). This causes SIFT to detect different features in each image because the gradient magnitudes and directions change with illumination.

### How Standard Histogram Equalization Works

Histogram equalization remaps pixel intensities so that the histogram of the output image is approximately uniform. This maximizes contrast. But global histogram equalization can over-amplify noise in homogeneous regions and wash out detail in already-bright areas.

### How CLAHE Improves on This

CLAHE applies histogram equalization **locally** to small tiles (typically 8×8 pixel blocks), with a **clip limit** that prevents over-amplification:

1. **Divide the image into tiles** (8×8 grid = 64 regions)
2. **Compute histogram** for each tile
3. **Clip** the histogram at the clip limit — any bin count exceeding the limit is redistributed equally to all bins. This prevents extreme contrast stretching.
4. **Apply equalization** within each tile
5. **Bilinear interpolation** at tile boundaries to avoid seam artifacts

### The Two CLAHE Configurations

| Name | `clipLimit` | Purpose |
|------|-------------|---------|
| `clahe_filter` | 2.5 | Moderate contrast enhancement — reduces lighting variation without amplifying noise too much |
| `clahe_strong` | 3.5 | Aggressive contrast enhancement — last resort for very difficult lighting pairs |

### Why This Helps Homography Estimation

After CLAHE, both images in a pair have more **uniform contrast**. Features that were invisible in dark regions become detectable, and features that were washed out in bright regions become distinct. SIFT can now find **more correct correspondences** between the images.

---

## 9. Brute-Force Matching + Lowe's Ratio Test

### Feature Matching

Given keypoints and descriptors from Image 1 (`des1`) and Image 2 (`des2`), we need to find **corresponding keypoints** — which keypoint in Image 1 matches which keypoint in Image 2.

### Brute-Force k-Nearest Neighbors (k=2)

```python
bf_matcher = cv2.BFMatcher(cv2.NORM_L2)
matches = bf_matcher.knnMatch(des1, des2, k=2)
```

For each descriptor in Image 1, find the **2 closest** descriptors in Image 2 (using L2 distance). This returns pairs `(best_match, second_best_match)` for every keypoint.

**Why L2 distance?** SIFT descriptors are L2-normalized floating-point vectors, so Euclidean distance is the natural metric. (For binary descriptors like ORB, you'd use Hamming distance instead.)

### Lowe's Ratio Test

> **Paper**: Lowe, 2004 (same SIFT paper)

The ratio test is a **match quality filter**. For each keypoint in Image 1:

```python
good = [m for m, n in matches if m.distance < ratio * n.distance]
```

**Intuition**: If the best match is **much closer** than the second-best match, the match is likely correct (the feature is distinctive). If both are similar distances, the match is ambiguous and likely wrong.

```
ratio = d(best_match) / d(second_best_match)
```

- `ratio < 0.75` → confident match (standard threshold)
- `ratio < 0.80` → relaxed — allows more matches but also more false positives
- `ratio < 0.85` → very relaxed — desperation mode

### Why This Works

In a typical image, most keypoints are non-distinctive (common edge patterns, textures). Their descriptors will have many near-equidistant neighbors in the other image. Truly distinctive features (unique corners, logos, texture patches) will have one very close neighbor and all others far away. The ratio test exploits this gap.

### The Code's Threshold Ladder

| Stage | Ratio | Effect |
|-------|-------|--------|
| 1, 2 | 0.75 | Conservative — only very confident matches |
| 3 | 0.80 | Slightly relaxed — a few more matches |
| 4 | 0.85 | Aggressive — accepts weaker matches |

---

## 10. USAC_MAGSAC — Robust Homography Estimation

### The Problem: Outliers

Even after the Lowe ratio test, some matches will be wrong (outliers). If you naively fit a homography to all matches using least squares, a single outlier can catastrophically distort the result. We need a **robust estimator** that finds the correct homography despite outlier contamination.

### RANSAC (Random Sample Consensus) — The Classic Approach

1. **Randomly sample** 4 point correspondences (the minimum needed to compute H)
2. **Compute H** from these 4 points using the Direct Linear Transform (DLT)
3. **Count inliers**: How many of the remaining matches agree with this H (their reprojection error < threshold)?
4. **Repeat** many times, keep the H with the most inliers
5. **Refine**: Refit H using all inliers from the best model

### MAGSAC (Marginalizing Sample Consensus) — The Modern Improvement

> **Paper**: Baráth et al., "MAGSAC: Marginalizing Sample Consensus" (CVPR 2019)

MAGSAC improves on RANSAC in two key ways:

1. **No hard inlier/outlier threshold**: Instead of classifying each match as inlier or outlier based on a fixed threshold, MAGSAC **marginalizes** (integrates) over a range of possible thresholds. Each match gets a **soft weight** based on how well it fits the model across all thresholds.

2. **σ-consensus**: It models the noise as a mixture of inlier and outlier distributions, automatically determining the noise level without user tuning.

### USAC (Universal RANSAC)

OpenCV's `cv2.USAC_MAGSAC` flag combines MAGSAC with additional improvements from the USAC framework:
- **Progressive NAPSAC** spatial sampling (samples nearby points first, exploiting spatial coherence)
- **PROSAC** score-ordered sampling (tries higher-quality matches first)
- **Local optimization** (LO-RANSAC refinement of promising models)
- **Degeneracy checking** (detects degenerate configurations like all points being collinear)

### In the Code

```python
H, mask = cv2.findHomography(src_pts, dst_pts, cv2.USAC_MAGSAC, ransac_thresh)
```

- `src_pts` / `dst_pts`: The matched point correspondences
- `cv2.USAC_MAGSAC`: The robust estimation method
- `ransac_thresh`: Maximum reprojection error (in pixels) for a match to be considered an inlier
  - 3.0 pixels (stages 1-2): strict — only very precise matches
  - 5.0 pixels (stages 3-4): relaxed — accepts slightly imprecise matches

- `mask`: A boolean array indicating which matches are inliers
- `inliers = mask.sum()`: The total inlier count — a key quality metric

### The Direct Linear Transform (DLT) — Under the Hood

When USAC_MAGSAC samples 4 point correspondences `(x₁ᵢ, y₁ᵢ) ↔ (x₂ᵢ, y₂ᵢ)`, it solves for H using the DLT:

Each correspondence gives 2 equations:
```
x₂ = (h₁₁·x₁ + h₁₂·y₁ + h₁₃) / (h₃₁·x₁ + h₃₂·y₁ + 1)
y₂ = (h₂₁·x₁ + h₂₂·y₁ + h₂₃) / (h₃₁·x₁ + h₃₂·y₁ + 1)
```

Rearranging into a linear system `Ah = 0` and solving via SVD gives the homography.

With 4 correspondences → 8 equations for 8 unknowns → unique solution (up to scale).

---

## 11. Geometric Guardrails (Validation Checks)

Even USAC_MAGSAC can sometimes produce a homography that is numerically "valid" but **physically nonsensical**. The code implements 5 layers of geometric sanity checking:

### Check 1: Finiteness

```python
if H is None or not np.isfinite(H).all():
    return False
```

Rejects matrices containing NaN or Inf values (numerical failure).

### Check 2: Determinant Bounds

```python
det = np.linalg.det(H)
if not (0.005 < abs(det) < 50.0):
    return False
```

**Why**: The determinant of H represents the **area scaling factor** of the transformation.
- `det ≈ 0` → the image collapses to a line (degenerate)
- `det < 0` → the image is reflected/flipped (usually wrong)
- `|det| >> 1` or `|det| << 1` → extreme zoom that's physically implausible for typical scenes

The bounds `0.005 < |det| < 50` allow moderate zoom (up to ~7× linear scale in either direction) while rejecting degeneracies.

### Check 3: Positive Projective Depth

```python
warped_homo = (H @ corners_1.T).T
if (warped_homo[:, 2] <= 1e-4).any():
    return False
```

When we warp the 4 corners of Image 1 using H, the homogeneous coordinate `w` (the third component) must be **positive** for all corners. If `w ≤ 0` for any corner, that point is "behind the camera" in projective geometry — the image would appear flipped inside-out or infinitely stretched. This is a physically impossible configuration for a valid photograph.

### Check 4: Convex Quadrilateral

```python
if not is_convex_quad(warped_pts):
    return False
```

The 4 warped corners should form a **convex quadrilateral** (like a trapezoid). If the quad is non-convex (concave) or self-intersecting, the homography has produced a "bow-tie" or "butterfly" shape — the image is folded onto itself, which is physically impossible.

**How convexity is checked**: Compute the cross product of consecutive edge vectors. If all cross products have the same sign (all positive or all negative), the polygon is convex. If signs differ, it's concave or self-intersecting.

```python
def is_convex_quad(pts):
    for i in range(4):
        v1 = pts[(i+1)%4] - pts[i]        # edge vector 1
        v2 = pts[(i+2)%4] - pts[(i+1)%4]  # edge vector 2
        cross = v1[0]*v2[1] - v1[1]*v2[0]  # 2D cross product (z-component)
        # All must be same sign for convexity
```

### Check 5: Bounding Box Sanity

```python
if box_w < 10 or box_h < 10:        # collapsed to nearly nothing
    return False
if box_w > 20*w2 or box_h > 20*h2:  # exploded to absurd size
    return False
```

The warped image shouldn't shrink to less than 10 pixels (degenerate) or expand to more than 20× the target image size (extreme perspective that's physically implausible).

---

## 12. The 4-Stage Cascade Pipeline

The core prediction logic in `predict_module.py` uses a **progressive escalation strategy**:

### Stage 1: Standard SIFT (the fast path)

```
Detector:     SIFT (5,000 features, contrast=0.015)
Preprocessing: None (raw grayscale)
Descriptor:   Standard SIFT (128-D)
Lowe ratio:   0.75 (strict)
RANSAC thresh: 3.0 pixels (strict)
```

**Triggers**: Always runs first.
**Logic**: For easy pairs with good lighting and moderate viewpoint change, this finds plenty of high-quality matches quickly.

### Stage 2: CLAHE + RootSIFT (lighting robustness)

```
Detector:     SIFT (8,000 features, contrast=0.008)
Preprocessing: CLAHE (clipLimit=2.5)
Descriptor:   RootSIFT (Hellinger-normalized)
Lowe ratio:   0.75 (strict)
RANSAC thresh: 3.0 pixels
```

**Triggers**: Only if Stage 1 produced `H = None` or `inliers < 15`.
**Logic**: CLAHE equalizes contrast differences between images, and RootSIFT makes descriptors more illumination-invariant. This combination rescues pairs where lighting variation caused Stage 1 to fail.

### Stage 3: CLAHE + Relaxed Matching (more permissive)

```
Detector:     SIFT (8,000 features, contrast=0.008)
Preprocessing: CLAHE (clipLimit=2.5)
Descriptor:   Standard SIFT
Lowe ratio:   0.80 (relaxed)
RANSAC thresh: 5.0 pixels (relaxed)
```

**Triggers**: Only if still `H = None` or `inliers < 10`.
**Logic**: By relaxing the Lowe ratio and RANSAC threshold, we accept more (potentially noisier) matches. USAC_MAGSAC's robustness can handle a higher outlier ratio.

### Stage 4: Aggressive CLAHE + Maximum Sensitivity (last resort)

```
Detector:     SIFT (10,000 features, contrast=0.005)
Preprocessing: CLAHE (clipLimit=3.5, stronger)
Descriptor:   Standard SIFT
Lowe ratio:   0.85 (very relaxed)
RANSAC thresh: 5.0 pixels
```

**Triggers**: Only if still `H = None` or `inliers < 10`.
**Logic**: Maximum detector sensitivity finds even the faintest features. Strong CLAHE aggressively equalizes contrast. Very relaxed matching accepts almost any plausible correspondence. This is the "throw everything at the wall" stage.

### Stage Interaction Logic

Each stage only **improves** on the previous result — it never makes things worse:

```python
if H_c is not None and inliers_c > max(inliers, 0):
    H_pred = H_c       # only accept if strictly more inliers
    inliers = inliers_c
```

### Final Fallback: Identity Matrix

```python
if H_pred is None or inliers < 10:
    H_pred = np.eye(3, dtype=np.float64)
```

If all 4 stages fail to find a reliable homography (< 10 inliers), fall back to the identity matrix. See next section for why.

---

## 13. The Identity Fallback Prior — Why It's Smart

This is a **Bayesian prior** based on domain knowledge from the EDA:

1. **~50% of scenes are illumination-only** (H = I is the exact answer)
2. **Even for viewpoint scenes**, the median reprojection error of identity is moderate — the camera usually hasn't moved dramatically
3. **A bad homography is worse than identity** — a wildly wrong H can produce errors > 1.0, while identity's error on a moderate viewpoint pair is typically 0.05-0.15

So the decision is:
- If we found a reliable H (≥10 inliers, passes all guardrails) → use it
- If we're not confident → identity is the safest bet

This is mathematically equivalent to a **MAP estimate** where the prior strongly favors identity and we only deviate when the evidence (inlier count) is strong enough.

---

## 14. Evaluation Script (evaluate.py)

`evaluate.py` implements the exact same metric that Kaggle uses:

1. Reads predicted and ground-truth CSVs
2. Merges on `pair_id`
3. For each pair: reads image dimensions (to convert between pixel and normalized coordinates), computes reprojection error on the 5 canonical points
4. Reports:
   - **Mean Reprojection Error** (lower is better)
   - **Leaderboard Score** = `100 × max(0, 1 − error/0.2)` (higher is better, 0-100 scale)

Usage:
```bash
python evaluate.py --predictions train_predictions.csv --ground_truth train.csv --data_dir data/train
```

---

## 15. Submission Generation (generate_submission.py)

`generate_submission.py` is a simple orchestrator:

1. Reads `test.csv` to get the 60 test pairs
2. For each pair, calls `predict(image_1_path, image_2_path)` from `predict_module.py`
3. Extracts the 8 H elements (`h₃₃ = 1` is omitted)
4. Saves to `submission.csv`
5. Measures total inference time
6. Automatically runs `validate_submission.py` on the output

---

## 16. Submission Validation (validate_submission.py)

`validate_submission.py` performs 5 checks before you upload to Kaggle:

| Check | What It Validates |
|-------|-------------------|
| Column headers | Exactly `pair_id,h11,h12,h13,h21,h22,h23,h31,h32` in that order |
| Row count | Exactly matches `test.csv` (60 rows) |
| pair_id order | Values and order match `test.csv` exactly |
| NaN/Inf | No null, NaN, or infinite values in any H element |
| Determinant | `|det(H)| > 1e-8` for all matrices (non-singular, invertible) |

---

## 17. The Solution Notebook (solution_notebook.ipynb)

`solution_notebook.ipynb` is a single, self-contained notebook that combines everything:

| Section | Contents |
|---------|----------|
| 0. Setup | Imports, Kaggle/local auto-detection path configuration, load CSVs |
| 1. Evaluation Metric | `warp_points`, `compute_reprojection_error`, `lb_score_from_error` |
| 2. EDA | Identity pair discovery, scene type breakdown, identity baseline score, error histogram, image dimension scatter plot |
| 3. Pipeline | All functions: `is_convex_quad`, `is_valid_homography`, `rootsift`, detector initialization, `match_and_estimate`, `predict` |
| 4. Validation | Full pipeline on 140 train pairs, prints score, worst pairs analysis, error distribution comparison histogram |
| 5. Visualization | Warped overlay alignment + SIFT keypoint match drawing |
| 6. Submission | Inference on 60 test pairs → `submission.csv` |
| 7. Validation | Inline column/row/NaN/determinant checks |

---

## 18. Complete Concept Map

```
                            Image Pair (Image 1 + Image 2)
                                        │
                                        ▼
                               Grayscale Conversion
                                        │
                                        ▼
                    ┌───── Stage 1: Standard SIFT ──────┐
                    │  SIFT (5K) → BFMatcher → Lowe 0.75 │
                    │  → USAC_MAGSAC → Guardrails         │
                    └──────────────┬────────────────────────┘
                                   │
                    (Valid + inliers ≥ 15?) ──YES──→ Return H
                                   │ NO
                                   ▼
                    ┌── Stage 2: CLAHE + RootSIFT ──────┐
                    │  CLAHE(2.5) → SIFT(8K) → RootSIFT  │
                    │  → BFMatcher → Lowe 0.75 → MAGSAC   │
                    └──────────────┬────────────────────────┘
                                   │
                    (Better?) ────YES──→ Return H
                                   │ NO
                                   ▼
                    ┌── Stage 3: Relaxed Matching ───────┐
                    │  CLAHE(2.5) → SIFT(8K) → Lowe 0.80 │
                    │  → RANSAC thresh 5.0                 │
                    └──────────────┬────────────────────────┘
                                   │
                    (Better?) ────YES──→ Return H
                                   │ NO
                                   ▼
                    ┌── Stage 4: Aggressive ─────────────┐
                    │  CLAHE(3.5) → SIFT(10K) → Lowe 0.85│
                    │  → RANSAC thresh 5.0                 │
                    └──────────────┬────────────────────────┘
                                   │
                    (Better?) ────YES──→ Return H
                                   │ NO
                                   ▼
                         Fallback: Identity I₃ₓ₃
                                   │
                                   ▼
                               Return H
```

### Summary of All Concepts Used

| Category | Concept | Where Used |
|----------|---------|-----------|
| **Linear Algebra** | Homogeneous coordinates, matrix multiplication, determinant, SVD | Homography definition, DLT, guardrails |
| **Projective Geometry** | Projective transformation, perspective division, vanishing points | Core problem formulation |
| **Feature Detection** | Scale-space, Gaussian pyramid, Difference-of-Gaussians, extrema detection | SIFT keypoint detection |
| **Feature Description** | Gradient histograms, orientation assignment, 128-D descriptor | SIFT descriptors |
| **Descriptor Transforms** | L1 normalization, Hellinger kernel, Bhattacharyya coefficient | RootSIFT |
| **Image Processing** | Histogram equalization, adaptive local enhancement, clip limiting | CLAHE |
| **Feature Matching** | k-Nearest Neighbors, L2 distance, Lowe's ratio test | BFMatcher |
| **Robust Estimation** | RANSAC, MAGSAC, marginal weighting, soft inlier scoring | USAC_MAGSAC |
| **Computational Geometry** | Cross products, convexity testing, bounding boxes | Guardrails |
| **Bayesian Reasoning** | Prior distribution (identity), evidence threshold (inlier count) | Fallback strategy |
| **Evaluation** | Reprojection error, normalized coordinates, Euclidean distance | Scoring metric |
| **Software Engineering** | Multi-stage cascade, early stopping, lazy evaluation | Pipeline architecture |
