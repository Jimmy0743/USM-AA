"""Prepare reference genome data for USM-AA pretraining.

This script processes a GRCh38 reference genome FASTA file, translating
IUPAC ambiguous nucleotide codes to standard bases (A, C, G, T).

Example:
    $ python prepare_ref.py --input /path/to/grch38.fasta --output_dir /path/to/output
"""
import os
import argparse
from tqdm import tqdm
from data_utils import (
    IUPAC_TRANSLATOR,
    MMapIndexedFastaDataset,
    MMapIndexedFastaDatasetBuilder
)


def main() -> None:
    """Main entry point for reference genome preparation."""
    parser = argparse.ArgumentParser(
        description="Prepare reference genome data for USM-AA pretraining"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input GRCh38 FASTA file path"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for processed data"
    )
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    ds = MMapIndexedFastaDataset(args.input, verbose=False)
    builder = MMapIndexedFastaDatasetBuilder(f"{args.output_dir}/ref.fasta")
    
    for data in tqdm(ds, desc="Processing reference"):
        # Translate IUPAC codes to standard bases
        data["seq"] = data["seq"].translate(IUPAC_TRANSLATOR)
        builder.add_item(data)
        builder.end_document()
    
    builder.finalize()


if __name__ == "__main__":
    main()
