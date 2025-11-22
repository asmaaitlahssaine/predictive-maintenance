# scripts/data_ingest.py
import shutil
from pathlib import Path
import argparse

def ingest_local(src: str, dest_dir: str = "data/raw"):
    """
    Copies the source CSV into the raw data directory.
    """
    src_path = Path(src).expanduser().resolve()
    dest_path = Path(dest_dir).resolve()

    dest_path.mkdir(parents=True, exist_ok=True)
    final_path = dest_path / src_path.name

    shutil.copy2(src_path, final_path)
    print(f"[INFO] Copied {src_path} → {final_path}")
    return str(final_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="Path to the source CSV file")
    args = parser.parse_args()

    ingest_local(args.src)
