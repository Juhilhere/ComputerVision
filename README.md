# Computer Vision Projects & Assignments

This repository contains solutions, experiments, pipelines, and submission notebooks for Computer Vision assignments and Kaggle benchmarks.

---

## Repository Structure

```
ComputerVision/
├── Assignment1/
│   └── find-homography-so-dope/
│       ├── data/                     # Image sequences for homography estimation
│       ├── evaluate.py               # Geometric reprojection error evaluation
│       ├── generate_submission.py    # Submission generation pipeline
│       ├── predict_module.py         # Prediction module & homography estimation
│       ├── solution_notebook.ipynb   # Complete walkthrough & experimental notebook
│       └── validate_submission.py    # Submission format validator
│
└── Assignment2/
    ├── build_mvit_notebook.py        # MViT temporal ensemble notebook builder
    ├── build_temporal_notebook.py    # Temporal domain shift experiment notebook builder
    ├── domain_audit.py               # Domain distribution analysis
    ├── exp.py / pair_exp.py          # Ensembling & fusion experiments
    ├── motion_domain.py              # Motion event transfer experiments
    ├── mvit_domain.py                # MViT feature evaluation across temporal slices
    ├── weight_exp.py                 # Multi-model weight optimization
    ├── kaggle_motion/                # Motion event Kaggle kernels
    ├── kaggle_mvit/                  # MViT temporal ensemble Kaggle kernels
    └── kaggle_output/ / mvit_output/ # Features, embeddings, and submissions
```

---

## Assignment 1: Homography Estimation (`find-homography-so-dope`)

- **Objective**: Predict 3×3 planar homography matrices between image pairs mapping $I_1 \rightarrow I_2$.
- **Approaches**: Feature detection and matching (SIFT / ORB), robust estimation (RANSAC), and learning-based refinement.
- **Metric**: Geometric reprojection error across normalized test coordinate points.

See [Assignment1/README.md](Assignment1/README.md) for detailed instructions, mathematical formulation, and evaluation scripts.

---

## Assignment 2: Video Classification & Temporal Modeling (`cooked-or-not`)

- **Objective**: Classify video actions and handle domain shifts in cooking/kitchen action datasets.
- **Pipelines**:
  - **MViT (Multiscale Vision Transformers)**: Temporal sampling across early, tail, and uniform segments.
  - **3D CNNs / R3D**: Spatio-temporal representation learning.
  - **Motion Event Transfer**: Optical flow and motion feature representations.
  - **Ensembling**: Gradient boosted trees (LightGBM) and weighted soft-voting over multi-backbone representations.
