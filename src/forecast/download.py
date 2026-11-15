"""Download the M5 raw files into data/raw/.

Source: the Zenodo mirror of the M5 Forecasting Accuracy dataset
(DOI 10.5281/zenodo.12636070, CC BY 4.0). It needs no login, unlike Kaggle.
Data credit: Walmart, the M5 organizers (MOFC, University of Nicosia), Kaggle.

Run with:  python -m forecast.download
"""

from __future__ import annotations

import hashlib
import shutil
import urllib.request
import zipfile
from pathlib import Path

from forecast import RAW_DIR

ZENODO_URL = (
    "https://zenodo.org/api/records/12636070/files/m5-forecasting-accuracy.zip/content"
)
ZIP_MD5 = "86f57416a314197f40a17cc6fc60cbb4"

# The only files the pipeline uses. The zip also holds the validation file
# (a shorter copy of the evaluation file) and a sample submission.
NEEDED_FILES = ("calendar.csv", "sell_prices.csv", "sales_train_evaluation.csv")


def md5_of(path: Path, chunk_size: int = 1 << 20) -> str:
    """Return the MD5 hex digest of a file, read in chunks."""
    digest = hashlib.md5()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def download_zip(dest: Path, url: str = ZENODO_URL, expected_md5: str = ZIP_MD5) -> Path:
    """Download the M5 zip to `dest` and verify its checksum."""
    print(f"Downloading {url} ...")
    with urllib.request.urlopen(url, timeout=60) as response, dest.open("wb") as f:
        shutil.copyfileobj(response, f)
    actual = md5_of(dest)
    if actual != expected_md5:
        dest.unlink()
        raise ValueError(f"Checksum mismatch: expected {expected_md5}, got {actual}")
    return dest


def extract_needed(zip_path: Path, out_dir: Path, names: tuple[str, ...] = NEEDED_FILES) -> None:
    """Extract only the named files, each written directly into `out_dir`.

    Files are written by their known names, not by paths stored in the zip,
    so a malformed archive cannot write outside `out_dir`.
    """
    with zipfile.ZipFile(zip_path) as zf:
        for name in names:
            with zf.open(name) as src, (out_dir / name).open("wb") as dst:
                shutil.copyfileobj(src, dst)
            print(f"  extracted {name}")


def main(raw_dir: Path = RAW_DIR) -> None:
    """Fetch the M5 files into `raw_dir`, skipping the download if they exist."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    if all((raw_dir / name).exists() for name in NEEDED_FILES):
        print(f"All M5 files already present in {raw_dir}; nothing to do.")
        return

    zip_path = raw_dir / "m5-forecasting-accuracy.zip"
    try:
        download_zip(zip_path)
        extract_needed(zip_path, raw_dir)
    finally:
        zip_path.unlink(missing_ok=True)
    print(f"Done. Files are in {raw_dir}")


if __name__ == "__main__":
    main()
