from pathlib import Path
from brats.preprocessing import preprocess_coreg_sri24reg_bet

def main():
    raw = Path("data/raw_case")
    out = Path("data/preprocessed_case")
    out.mkdir(parents=True, exist_ok=True)

    preprocess_coreg_sri24reg_bet(
        t1_input=raw / "t1n.nii",
        t1c_input=raw / "t1c.nii",
        t2_input=raw / "t2w.nii",
        flair_input=raw / "t2f.nii",
        t1_output=out / "t1n.nii",
        t1c_output=out / "t1c.nii",
        t2_output=out / "t2w.nii",
        flair_output=out / "t2f.nii",
    )

    print("Preprocessing complete:", out)

if __name__ == "__main__":
    main()
