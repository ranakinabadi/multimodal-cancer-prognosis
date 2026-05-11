"""
TCIA download script.

Usage:
    python scripts/download_data.py --n_patients 200 --output data/raw_ct/

Requires:
    pip install tcia_utils
"""

import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download NSCLC-Radiomics patients from TCIA."
    )
    parser.add_argument(
        "--n_patients", type=int, default=200,
        help="Number of patients to download (default: 200)"
    )
    parser.add_argument(
        "--output", type=str, default="data/raw_ct/",
        help="Output directory for DICOM files (default: data/raw_ct/)"
    )
    parser.add_argument(
        "--collection", type=str, default="NSCLC-Radiomics",
        help="TCIA collection name (default: NSCLC-Radiomics)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        from tcia_utils import nbia
    except ImportError:
        raise ImportError("Install tcia_utils: pip install tcia_utils")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching series list from collection: {args.collection}")
    series_data = nbia.getSeries(collection=args.collection)

    if series_data is None:
        raise RuntimeError(
            f"No series returned for collection '{args.collection}'. "
            "Check the collection name with nbia.getCollections()."
        )

    print(f"Total series available : {len(series_data)}")
    print(f"Downloading first      : {args.n_patients}")
    print(f"Output directory       : {output_dir.resolve()}")

    nbia.downloadSeries(
        series_data,
        number=args.n_patients,
        path=str(output_dir),
    )

    print(f"\nDownload complete. Files saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
