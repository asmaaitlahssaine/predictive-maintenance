# scripts/preprocess.py

import pandas as pd
from pathlib import Path
import argparse

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocessing pipeline extracted from the notebook.
    Includes:
    - Feature engineering:
        * Engine_power
        * Temperature_difference
    """

    # ------------------------------
    # 1) Engine Power Feature
    # ------------------------------
    if "Engine rpm" in df.columns and "Lub oil pressure" in df.columns:
        df["Engine_power"] = df["Engine rpm"] * df["Lub oil pressure"]
    else:
        print("[WARNING] Missing columns for Engine_power")

    # ------------------------------
    # 2) Temperature Difference Feature
    # ------------------------------
    if "Coolant temp" in df.columns and "lub oil temp" in df.columns:
        df["Temperature_difference"] = df["Coolant temp"] - df["lub oil temp"]
    else:
        print("[WARNING] Missing columns for Temperature_difference")

    # No cleaning / No dropping NAs in your notebook
    # No encoding / No scaling
    # Everything else was only descriptive statistics or visualization

    return df


def preprocess(input_csv: str, output_csv: str = "data/processed/processed.csv"):
    """
    Complete preprocessing pipeline.
    Loads raw CSV → applies feature engineering → saves processed CSV.
    """

    print(f"[INFO] Loading raw data from: {input_csv}")
    df = pd.read_csv(input_csv)

    print("[INFO] Applying preprocessing steps...")
    df_processed = build_features(df)

    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_processed.to_csv(out_path, index=False)

    print(f"[INFO] Processed data saved to: {out_path}")
    return str(out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to raw CSV")
    parser.add_argument("--output", default="data/processed/processed.csv", help="Output CSV path")
    args = parser.parse_args()

    preprocess(args.input, args.output)
