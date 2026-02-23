import numpy as np
import nibabel as nib
from pathlib import Path

def dice(a, b):
    a = a.astype(bool); b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    denom = a.sum() + b.sum()
    return (2 * inter / denom) if denom > 0 else 1.0

def main():
    pred = nib.load("outputs/segmentation.nii.gz").get_fdata().astype(np.int16)
    gt   = nib.load("data/raw_case/seg.nii").get_fdata().astype(np.int16)

    # BraTS labels: 1=NET/NCR, 2=ED, 4=ET
    wt_pred = pred > 0
    wt_gt   = gt > 0
    tc_pred = np.logical_or(pred == 1, pred == 4)
    tc_gt   = np.logical_or(gt == 1, gt == 4)
    et_pred = pred == 4
    et_gt   = gt == 4

    print("Dice WT:", dice(wt_pred, wt_gt))
    print("Dice TC:", dice(tc_pred, tc_gt))
    print("Dice ET:", dice(et_pred, et_gt))

if __name__ == "__main__":
    main()
