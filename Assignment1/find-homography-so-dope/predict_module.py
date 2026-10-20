"""
Predict Module for 'Find Homography So Dope' Competition
Exposes:
    predict(image_1_path: str, image_2_path: str) -> np.ndarray
Returning a 3x3 homography matrix H with H[2, 2] = 1.0, mapping Image 1 -> Image 2.
"""

import os
import cv2
import numpy as np


def is_convex_quad(pts: np.ndarray) -> bool:
    """
    Checks if a 4-point polygon (pts shape (4, 2)) is strictly convex.
    """
    cross_products = []
    for i in range(4):
        p1 = pts[i]
        p2 = pts[(i + 1) % 4]
        p3 = pts[(i + 2) % 4]
        v1 = p2 - p1
        v2 = p3 - p2
        cp = v1[0] * v2[1] - v1[1] * v2[0]
        cross_products.append(cp)
    all_pos = all(cp > 0 for cp in cross_products)
    all_neg = all(cp < 0 for cp in cross_products)
    return all_pos or all_neg


def is_valid_homography(H: np.ndarray, w1: int, h1: int, w2: int, h2: int) -> bool:
    """
    Validates physical feasibility of the homography:
    - Finite numerical values
    - Determinant within realistic perspective transformation bounds
    - Warped corners have positive projective depth (w > 0)
    - Warped image boundary forms a convex quadrilateral
    - Warped bounding box within reasonable coordinate range
    """
    if H is None or not np.isfinite(H).all():
        return False
    
    det = np.linalg.det(H)
    if not (0.005 < abs(det) < 50.0):
        return False
        
    corners_1 = np.array([
        [0.0, 0.0, 1.0],
        [float(w1), 0.0, 1.0],
        [float(w1), float(h1), 1.0],
        [0.0, float(h1), 1.0]
    ], dtype=np.float64)
    
    warped_homo = (H @ corners_1.T).T
    if (warped_homo[:, 2] <= 1e-4).any():
        return False
        
    warped_pts = warped_homo[:, :2] / warped_homo[:, 2:3]
    
    if not is_convex_quad(warped_pts):
        return False
        
    x_min, y_min = warped_pts.min(axis=0)
    x_max, y_max = warped_pts.max(axis=0)
    box_w = x_max - x_min
    box_h = y_max - y_min
    
    if box_w < 10 or box_h < 10:
        return False
    if box_w > 20 * w2 or box_h > 20 * h2:
        return False
        
    return True


def rootsift(des: np.ndarray) -> np.ndarray:
    """
    Applies Hellinger kernel / RootSIFT transformation to SIFT descriptors.
    """
    if des is None or len(des) == 0:
        return des
    des_norm = des / (np.linalg.norm(des, axis=1, ord=1, keepdims=True) + 1e-7)
    des_sqrt = np.sqrt(des_norm)
    des_l2 = des_sqrt / (np.linalg.norm(des_sqrt, axis=1, ord=2, keepdims=True) + 1e-7)
    return des_l2.astype(np.float32)


# Initialize detectors and matchers
_sift_standard = cv2.SIFT_create(nfeatures=5000, contrastThreshold=0.015, edgeThreshold=10)
_sift_sensitive = cv2.SIFT_create(nfeatures=8000, contrastThreshold=0.008, edgeThreshold=10)
_sift_aggressive = cv2.SIFT_create(nfeatures=10000, contrastThreshold=0.005, edgeThreshold=10)
_bf_matcher = cv2.BFMatcher(cv2.NORM_L2)
_clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
_clahe_strong = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))


def _match_and_estimate(kp1, des1, kp2, des2, w1, h1, w2, h2, ratio=0.75, ransac_thresh=3.0):
    if des1 is None or des2 is None or len(des1) < 4 or len(des2) < 4:
        return None, 0
        
    matches = _bf_matcher.knnMatch(des1, des2, k=2)
    good = [m for m, n in matches if len((m, n)) == 2 and m.distance < ratio * n.distance]
    if len(good) < 4:
        return None, 0
        
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.USAC_MAGSAC, ransac_thresh)
    inliers = int(mask.sum()) if mask is not None else 0
    
    if H is not None and np.isfinite(H).all() and abs(H[2, 2]) > 1e-8:
        H_normalized = H / H[2, 2]
        if is_valid_homography(H_normalized, w1, h1, w2, h2):
            return H_normalized, inliers
            
    return None, 0


def predict(image_1_path: str, image_2_path: str) -> np.ndarray:
    """
    Predicts the 3x3 homography matrix mapping image 1 to image 2.
    Returns:
        np.ndarray of shape (3, 3) with H[2, 2] == 1.0.
    """
    img1 = cv2.imread(image_1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image_2_path, cv2.IMREAD_GRAYSCALE)
    
    if img1 is None or img2 is None:
        return np.eye(3, dtype=np.float64)
        
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # Stage 1: Standard SIFT with strict Lowe ratio 0.75
    kp1, des1 = _sift_standard.detectAndCompute(img1, None)
    kp2, des2 = _sift_standard.detectAndCompute(img2, None)
    
    H_pred, inliers = _match_and_estimate(kp1, des1, kp2, des2, w1, h1, w2, h2, ratio=0.75, ransac_thresh=3.0)
    
    # Stage 2: If Stage 1 had fewer than 15 inliers, try CLAHE + RootSIFT
    if H_pred is None or inliers < 15:
        img1_clahe = _clahe.apply(img1)
        img2_clahe = _clahe.apply(img2)
        
        kp1_c, des1_c = _sift_sensitive.detectAndCompute(img1_clahe, None)
        kp2_c, des2_c = _sift_sensitive.detectAndCompute(img2_clahe, None)
        
        des1_r = rootsift(des1_c)
        des2_r = rootsift(des2_c)
        
        H_c, inliers_c = _match_and_estimate(kp1_c, des1_r, kp2_c, des2_r, w1, h1, w2, h2, ratio=0.75, ransac_thresh=3.0)
        
        if H_c is not None and inliers_c > max(inliers, 0):
            H_pred = H_c
            inliers = inliers_c

    # Stage 3: If still weak, try CLAHE + standard SIFT descriptors (no RootSIFT) + relaxed ratio
    if H_pred is None or inliers < 10:
        img1_clahe = _clahe.apply(img1)
        img2_clahe = _clahe.apply(img2)
        
        kp1_c, des1_c = _sift_sensitive.detectAndCompute(img1_clahe, None)
        kp2_c, des2_c = _sift_sensitive.detectAndCompute(img2_clahe, None)
        
        H_c, inliers_c = _match_and_estimate(kp1_c, des1_c, kp2_c, des2_c, w1, h1, w2, h2, ratio=0.80, ransac_thresh=5.0)
        
        if H_c is not None and inliers_c > max(inliers, 0):
            H_pred = H_c
            inliers = inliers_c

    # Stage 4: Aggressive CLAHE + ultra-sensitive SIFT + very relaxed matching
    if H_pred is None or inliers < 10:
        img1_clahe = _clahe_strong.apply(img1)
        img2_clahe = _clahe_strong.apply(img2)
        
        kp1_c, des1_c = _sift_aggressive.detectAndCompute(img1_clahe, None)
        kp2_c, des2_c = _sift_aggressive.detectAndCompute(img2_clahe, None)
        
        H_c, inliers_c = _match_and_estimate(kp1_c, des1_c, kp2_c, des2_c, w1, h1, w2, h2, ratio=0.85, ransac_thresh=5.0)
        
        if H_c is not None and inliers_c > max(inliers, 0):
            H_pred = H_c
            inliers = inliers_c
            
    # Final Fallback: If inliers < 10 or H is invalid, fallback to Identity (optimal prior)
    if H_pred is None or inliers < 10:
        H_pred = np.eye(3, dtype=np.float64)
        
    # Ensure exact H[2, 2] == 1.0
    H_pred = H_pred / H_pred[2, 2]
    return H_pred.astype(np.float64)
