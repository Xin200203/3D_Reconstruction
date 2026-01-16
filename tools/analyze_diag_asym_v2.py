
import os
import glob
import numpy as np
import argparse

def analyze_diag(data_dir):
    print(f"Analyzing {data_dir}...")
    files = glob.glob(os.path.join(data_dir, "*.npz"))
    print(f"Found {len(files)} files.")

    all_pos_det = []
    all_neg_det = []
    
    all_pos_tt = []
    all_neg_tt = []

    all_dino_pos_det = []
    all_dino_neg_det = []
    all_dino_pos_tt = []
    all_dino_neg_tt = []

    for f in files:
        try:
            d = np.load(f, allow_pickle=True)
            if 'records' not in d:
                # Try fallback legacy
                recs = [d]
            else:
                recs = d['records']
                # recs might be 0-d array wrapping list or something
                if recs.ndim == 0:
                    recs = recs.item()
                if isinstance(recs, np.ndarray):
                    # It might be array of dicts
                    pass 
            
            for r in recs:
                # r should be dict
                if not isinstance(r, dict):
                    continue

                # Helper to safe extend
                def safe_extend(target_list, key):
                    if key in r and r[key] is not None:
                         arr = np.array(r[key]).flatten()
                         if arr.size > 0:
                             target_list.extend(arr)

                # 1. Det-Track
                safe_extend(all_pos_det, 'M_pos')
                safe_extend(all_neg_det, 'M_neg')
                safe_extend(all_dino_pos_det, 'M_dino_pos')
                safe_extend(all_dino_neg_det, 'M_dino_neg')

                # 2. Track-Track Asymmetric
                if 'M_tt_ij' in r and r['M_tt_ij'] is not None:
                    m = np.array(r['M_tt_ij'])
                    if m.ndim == 2 and m.shape[0] > 1:
                        n = m.shape[0]
                        # Loop over unique pairs i < j
                        for i in range(n):
                            for j in range(i+1, n):
                                val1 = m[i, j]
                                val2 = m[j, i]
                                all_pos_tt.append(max(val1, val2))
                elif 'M_tt' in r and r['M_tt'] is not None: # Fallback legacy
                    pass
                
                # Dino Positive
                if 'M_dino_tt' in r and r['M_dino_tt'] is not None:
                    m = np.array(r['M_dino_tt'])
                    if m.ndim == 2 and m.shape[0] > 1:
                        n = m.shape[0]
                        for i in range(n):
                            for j in range(i+1, n):
                                all_dino_pos_tt.append(m[i,j])
                
                # Negative Pairs
                if 'M_tt_neg_ij' in r and 'M_tt_neg_ji' in r and r['M_tt_neg_ij'] is not None and r['M_tt_neg_ji'] is not None:
                    m_ij = np.array(r['M_tt_neg_ij'])
                    m_ji = np.array(r['M_tt_neg_ji'])
                    
                    if m_ij.size > 0 and m_ji.size > 0:
                         # Ensure shapes match expectation
                         # m_ij: (nP, nN), m_ji: (nN, nP)
                         if m_ji.shape == (m_ij.shape[1], m_ij.shape[0]):
                            for i in range(m_ij.shape[0]):
                                for j in range(m_ij.shape[1]):
                                    v1 = m_ij[i, j]
                                    v2 = m_ji[j, i]
                                    all_neg_tt.append(max(v1, v2))

                if 'M_dino_tt_neg' in r and r['M_dino_tt_neg'] is not None:
                     safe_extend(all_dino_neg_tt, 'M_dino_tt_neg')

        except Exception as e:
            # print(f"Error reading {f}: {e}")
            import traceback
            traceback.print_exc()

    # Analyze
    all_pos_det = np.array(all_pos_det)
    all_neg_det = np.array(all_neg_det)
    all_pos_tt = np.array(all_pos_tt)
    all_neg_tt = np.array(all_neg_tt)
    all_dino_pos_tt = np.array(all_dino_pos_tt)
    all_dino_neg_tt = np.array(all_dino_neg_tt)

    print(f"\nStats for {data_dir}")
    print(f"Det-Track Pos: {len(all_pos_det)}, Neg: {len(all_neg_det)}")
    print(f"Track-Track Pos (Triang): {len(all_pos_tt)}, Neg: {len(all_neg_tt)}")

    # Thresholds
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    print("\n--- Det-Track Geometric Containment ---")
    print("Thresh | TPR   | FPR")
    for t in thresholds:
        tpr = (all_pos_det > t).mean() if len(all_pos_det) else 0
        fpr = (all_neg_det > t).mean() if len(all_neg_det) else 0
        print(f"{t:.1f}    | {tpr:.3f} | {fpr:.3f}")

    print("\n--- Track-Track Asymmetric Containment (Max(A in B, B in A)) ---")
    print("Thresh | TPR   | FPR")
    for t in thresholds:
        tpr = (all_pos_tt > t).mean() if len(all_pos_tt) else 0
        fpr = (all_neg_tt > t).mean() if len(all_neg_tt) else 0
        print(f"{t:.1f}    | {tpr:.3f} | {fpr:.3f}")

    print("\n--- Track-Track DINO Similarity ---")
    print("Thresh | TPR   | FPR")
    dino_thresh = [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for t in dino_thresh:
        tpr = (all_dino_pos_tt > t).mean() if len(all_dino_pos_tt) else 0
        fpr = (all_dino_neg_tt > t).mean() if len(all_dino_neg_tt) else 0
        print(f"{t:.1f}    | {tpr:.3f} | {fpr:.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', type=str)
    args = parser.parse_args()
    analyze_diag(args.dir)
