"""
Evaluation script for 'Find Homography So Dope' competition.
Computes mean geometric reprojection error on 5 normalized points:
    (0,0), (1,0), (1,1), (0,1), (0.5, 0.5)
and the Kaggle Leaderboard Score (0-100):
    score = 100 * max(0, 1 - error / 0.2)
"""

import argparse
import os
import cv2
import numpy as np
import pandas as pd

POINTS_NORM = np.array([
    [0.0, 0.0],
    [1.0, 0.0],
    [1.0, 1.0],
    [0.0, 1.0],
    [0.5, 0.5]
], dtype=np.float64)


def parse_h(row):
    return np.array([
        [float(row['h11']), float(row['h12']), float(row['h13'])],
        [float(row['h21']), float(row['h22']), float(row['h23'])],
        [float(row['h31']), float(row['h32']), 1.0]
    ], dtype=np.float64)


def warp_points(H, pts):
    pts_homo = np.hstack([pts, np.ones((len(pts), 1), dtype=np.float64)])
    warped = (H @ pts_homo.T).T
    w = warped[:, 2:3]
    w = np.where(np.abs(w) < 1e-8, 1e-8, w)
    return warped[:, :2] / w


def compute_reprojection_error(pred_h, gt_h, w1, h1, w2, h2):
    pts_px_1 = POINTS_NORM * np.array([w1, h1], dtype=np.float64)
    pred_px_2 = warp_points(pred_h, pts_px_1)
    gt_px_2 = warp_points(gt_h, pts_px_1)
    pred_norm_2 = pred_px_2 / np.array([w2, h2], dtype=np.float64)
    gt_norm_2 = gt_px_2 / np.array([w2, h2], dtype=np.float64)
    return np.linalg.norm(pred_norm_2 - gt_norm_2, axis=1).mean()


def evaluate(predictions_path, ground_truth_path, data_dir=None):
    pred_df = pd.read_csv(predictions_path)
    gt_df = pd.read_csv(ground_truth_path)
    
    merged = pd.merge(pred_df, gt_df, on='pair_id', suffixes=('_pred', '_gt'))
    if len(merged) == 0:
        raise ValueError("No matching pair_ids between predictions and ground truth!")
        
    errors = []
    for _, row in merged.iterrows():
        pred_h = np.array([
            [float(row['h11_pred']), float(row['h12_pred']), float(row['h13_pred'])],
            [float(row['h21_pred']), float(row['h22_pred']), float(row['h23_pred'])],
            [float(row['h31_pred']), float(row['h32_pred']), 1.0]
        ], dtype=np.float64)
        
        gt_h = np.array([
            [float(row['h11_gt']), float(row['h12_gt']), float(row['h13_gt'])],
            [float(row['h21_gt']), float(row['h22_gt']), float(row['h23_gt'])],
            [float(row['h31_gt']), float(row['h32_gt']), 1.0]
        ], dtype=np.float64)
        
        # Read dimensions if data_dir is provided and images exist
        w1, h1, w2, h2 = 1000, 1000, 1000, 1000
        if data_dir and 'image_1' in row and 'image_2' in row:
            p1 = os.path.join(data_dir, row['image_1'])
            p2 = os.path.join(data_dir, row['image_2'])
            if os.path.exists(p1) and os.path.exists(p2):
                im1 = cv2.imread(p1)
                im2 = cv2.imread(p2)
                if im1 is not None and im2 is not None:
                    h1, w1 = im1.shape[:2]
                    h2, w2 = im2.shape[:2]
                    
        err = compute_reprojection_error(pred_h, gt_h, w1, h1, w2, h2)
        errors.append(err)
        
    mean_err = float(np.mean(errors))
    lb_score = float(100.0 * max(0.0, 1.0 - mean_err / 0.2))
    
    print(f"Evaluated {len(errors)} pairs:")
    print(f"Mean Reprojection Error: {mean_err:.5f} (lower is better)")
    print(f"Leaderboard Match Score: {lb_score:.2f} / 100 (higher is better)")
    return mean_err, lb_score


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate homography predictions against ground truth.")
    parser.add_argument('--predictions', required=True, help="Path to predictions CSV")
    parser.add_argument('--ground_truth', required=True, help="Path to ground truth CSV (e.g. train.csv)")
    parser.add_argument('--data_dir', default=None, help="Root directory containing images")
    args = parser.parse_args()
    
    evaluate(args.predictions, args.ground_truth, args.data_dir)
