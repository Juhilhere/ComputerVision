"""
Generate Submission CSV for 'Find Homography So Dope'
Uses predict_module to predict homographies for test.csv and saves to submission.csv.
"""

import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from predict_module import predict
from validate_submission import validate_submission


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    test_csv = os.path.join(base_dir, 'test.csv')
    test_img_dir = os.path.join(base_dir, 'data', 'test')
    output_csv = os.path.join(base_dir, 'submission.csv')
    
    print(f"Reading test data from {test_csv}...")
    test_df = pd.read_csv(test_csv)
    print(f"Found {len(test_df)} test pairs.")
    
    rows = []
    start_time = time.time()
    
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Predicting homographies"):
        pair_id = row['pair_id']
        p1 = os.path.join(test_img_dir, row['image_1'])
        p2 = os.path.join(test_img_dir, row['image_2'])
        
        H = predict(p1, p2)
        
        rows.append({
            'pair_id': pair_id,
            'h11': float(H[0, 0]),
            'h12': float(H[0, 1]),
            'h13': float(H[0, 2]),
            'h21': float(H[1, 0]),
            'h22': float(H[1, 1]),
            'h23': float(H[1, 2]),
            'h31': float(H[2, 0]),
            'h32': float(H[2, 1])
        })
        
    sub_df = pd.DataFrame(rows)
    cols = ['pair_id', 'h11', 'h12', 'h13', 'h21', 'h22', 'h23', 'h31', 'h32']
    sub_df = sub_df[cols]
    
    sub_df.to_csv(output_csv, index=False)
    elapsed = time.time() - start_time
    print(f"\nSaved predictions to {output_csv}")
    print(f"Total inference time: {elapsed:.2f}s ({elapsed/len(test_df):.3f}s / pair)")
    
    # Validate the generated submission
    print("\n--- Running Validation ---")
    validate_submission(output_csv, test_csv)


if __name__ == '__main__':
    main()
