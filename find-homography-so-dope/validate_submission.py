"""
Validation script for 'Find Homography So Dope' submission file.
Checks:
- Column names: pair_id,h11,h12,h13,h21,h22,h23,h31,h32
- Row count matches test.csv exactly
- Pair IDs match test.csv exactly in same order
- No nulls, NaNs, or Infs
- All homography matrices are invertible (non-zero det)
"""

import argparse
import sys
import numpy as np
import pandas as pd

EXPECTED_COLS = ['pair_id', 'h11', 'h12', 'h13', 'h21', 'h22', 'h23', 'h31', 'h32']


def validate_submission(submission_path: str, test_path: str):
    print(f"Validating submission: {submission_path}")
    print(f"Against test reference: {test_path}")
    
    try:
        sub_df = pd.read_csv(submission_path)
    except Exception as e:
        print(f"[ERROR] Failed to read submission CSV: {e}")
        sys.exit(1)
        
    try:
        test_df = pd.read_csv(test_path)
    except Exception as e:
        print(f"[ERROR] Failed to read test reference CSV: {e}")
        sys.exit(1)
        
    # Check columns
    if list(sub_df.columns) != EXPECTED_COLS:
        print(f"[ERROR] Columns mismatch!")
        print(f"Expected: {EXPECTED_COLS}")
        print(f"Found:    {list(sub_df.columns)}")
        sys.exit(1)
    print("  [OK] Column headers match expected format.")
    
    # Check row count
    if len(sub_df) != len(test_df):
        print(f"[ERROR] Row count mismatch! Expected {len(test_df)} rows, found {len(sub_df)} rows.")
        sys.exit(1)
    print(f"  [OK] Row count matches ({len(sub_df)} rows).")
    
    # Check pair_ids
    if not (sub_df['pair_id'].values == test_df['pair_id'].values).all():
        print("[ERROR] pair_id values or order do not match test.csv!")
        sys.exit(1)
    print("  [OK] pair_id values and order match test.csv perfectly.")
    
    # Check NaNs / Infs
    for col in EXPECTED_COLS[1:]:
        if sub_df[col].isnull().any():
            print(f"[ERROR] Column {col} contains null / NaN values!")
            sys.exit(1)
        if np.isinf(sub_df[col]).any():
            print(f"[ERROR] Column {col} contains Infinite values!")
            sys.exit(1)
    print("  [OK] No nulls, NaNs, or Infs found.")
    
    # Check non-degeneracy
    degenerate_count = 0
    for idx, row in sub_df.iterrows():
        H = np.array([
            [float(row['h11']), float(row['h12']), float(row['h13'])],
            [float(row['h21']), float(row['h22']), float(row['h23'])],
            [float(row['h31']), float(row['h32']), 1.0]
        ])
        det = np.linalg.det(H)
        if abs(det) < 1e-8:
            degenerate_count += 1
            print(f"  [WARNING] Pair {row['pair_id']} has near-zero determinant ({det:.2e})!")
            
    if degenerate_count == 0:
        print("  [OK] All homography matrices are strictly non-singular and invertible.")
    else:
        print(f"  [WARNING] Found {degenerate_count} singular/degenerate matrices.")
        
    print("\nSUCCESS: The submission file is completely valid and ready for Kaggle upload!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Validate competition submission format.")
    parser.add_argument('--submission', default='submission.csv', help="Path to submission CSV")
    parser.add_argument('--test', default='test.csv', help="Path to test CSV")
    args = parser.parse_args()
    
    validate_submission(args.submission, args.test)
