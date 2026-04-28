import os
import glob
import numpy as np
from scipy.io import loadmat
from sklearn.metrics import accuracy_score, cohen_kappa_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # Change this folder path to the specific subfolder you want to test
    folder_path = r"M:\Alex\REST-Testing\CD1"
    
    print(f"\n==================================")
    print(f"Starting Evaluation for Folder: {folder_path}")
    
    rest_files = glob.glob(os.path.join(folder_path, "*_REST_V1.52.mat"))
    if not rest_files:
        print("No inference files found in this folder.")
        return

    print(f"Found {len(rest_files)} inference files.")

    all_true = []
    all_pred = []

    for rest_p in rest_files:
        # ref_p = rest_p.replace("_data_REST_V1.5.mat", "_scores.mat") # for fmr1
        ref_p = rest_p.replace("_REST_V1.52.mat", "_reference.mat") # for others
        if not os.path.exists(ref_p):
            print(f"  -> Reference missing for {os.path.basename(rest_p)}")
            continue

        try:
            d_rest = loadmat(rest_p)
            d_ref = loadmat(ref_p)

            pred_score = d_rest['score'].flatten()
            true_score = d_ref['score'].flatten()

            final_len = min(len(pred_score), len(true_score))
            pred_score = pred_score[:final_len]
            true_score = true_score[:final_len]
            
            all_pred.extend(pred_score)
            all_true.extend(true_score)

        except Exception as e:
            print(f"Error processing {rest_p}: {e}")

    if len(all_true) == 0:
        print("No valid paired data found to evaluate in this folder.")
        return

    # To 0-indexed for classification_report mapping
    y_true = np.array(all_true) - 1
    y_pred = np.array(all_pred) - 1

    # Map state labels
    # Just in case there are 4s or out of bound values
    y_true[y_true > 2] = 0

    acc = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)

    print("\n==================================")
    print(f"Accuracy: {acc:.4f}")
    print(f"Cohen's Kappa: {kappa:.3f}\n")
    
    print("Classification Report:")
    report = classification_report(y_true, y_pred, target_names=["Wake", "NREM", "REM"], labels=[0, 1, 2], zero_division=0)
    print(report)
    
    print("Confusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    print("==================================\n")

if __name__ == "__main__":
    main()
