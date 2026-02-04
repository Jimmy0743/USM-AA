"""
Prepare GUE (Genomic Understanding Evaluation) dataset for USM-AA training.

This script processes the GUE benchmark dataset, adding alignment information
using Minimap2 for various genomic tasks including promoter prediction,
transcription factor binding, splice site prediction, etc.
"""
import argparse
import glob
import os

import tqdm
import pandas as pd

from data_utils import MMapIndexedJsonlDataset, MMapIndexedJsonlDatasetBuilder
from prepare_dna import MinimapAligner


def prepare_gue(data_dir: str, ref_file: str, best_n: int = 10):
    """Process GUE dataset with alignment information."""
    # Collect all CSV files from different task categories
    files = []
    files += list(glob.glob(f"{data_dir}/prom/*/*.csv"))
    files += list(glob.glob(f"{data_dir}/tf/*/*.csv"))
    files += list(glob.glob(f"{data_dir}/splice/*/*.csv"))
    files += list(glob.glob(f"{data_dir}/mouse/*/*.csv"))
    files += list(glob.glob(f"{data_dir}/fungi/*/*.csv"))
    files += list(glob.glob(f"{data_dir}/EMP/*/*.csv"))
    files += list(glob.glob(f"{data_dir}/virus/*/*.csv"))
    
    print(f"Found {len(files)} CSV files")
    
    aligner = MinimapAligner(ref_file, best_n=best_n)
    
    for path in files:
        print(f"\nInput: {path}")
        name = os.path.basename(path).split('.')[0]
        output = f"{os.path.dirname(path)}/{name}.jsonl"
        print(f"Output: {output}")
        
        builder = MMapIndexedJsonlDatasetBuilder(output)
        df = pd.read_csv(path)
        n_aligned = 0
        
        for _, row in tqdm.tqdm(df.iterrows(), total=len(df)):
            sequence = row['sequence']
            alignments = aligner(sequence)
            if len(alignments) > 0:
                n_aligned += 1
            label = row['label']
            data = {"sequence": sequence, "alignments": alignments, "label": label}
            builder.add_item(data)
            builder.end_document()
            
        print(f"Aligned: {n_aligned}/{len(df)} ({n_aligned/len(df)*100:.1f}%)")
        builder.finalize()


def prepare_epi(data_dir: str, ref_file: str, best_n: int = 10):
    """Process EPI (Enhancer-Promoter Interaction) dataset."""
    files = list(glob.glob(f"{data_dir}/EPI/*/*.csv"))
    print(f"Found {len(files)} EPI CSV files")
    
    aligner = MinimapAligner(ref_file, best_n=best_n)
    
    for path in files:
        print(f"\nInput: {path}")
        name = os.path.basename(path).split('.')[0]
        output = f"{os.path.dirname(path)}/{name}.jsonl"
        print(f"Output: {output}")
        
        builder = MMapIndexedJsonlDatasetBuilder(output)
        df = pd.read_csv(path)
        n_aligned = 0
        
        for _, row in tqdm.tqdm(df.iterrows(), total=len(df)):
            # Process enhancer sequence
            enhancer_seq = row['enhancer']
            enhancer_alignments = aligner(enhancer_seq)
            if len(enhancer_alignments) > 0:
                n_aligned += 1
            enhancer = {"sequence": enhancer_seq, "alignments": enhancer_alignments}
            
            # Process promoter sequence
            promoter_seq = row['promoter']
            promoter_alignments = aligner(promoter_seq)
            promoter = {"sequence": promoter_seq, "alignments": promoter_alignments}
            
            label = row['label']
            builder.add_item({
                "enhancer": enhancer, 
                "promoter": promoter, 
                "label": label
            })
            builder.end_document()

        print(f"Aligned: {n_aligned}/{len(df)} ({n_aligned/len(df)*100:.1f}%)")
        builder.finalize()


def analyze_gue(data_dir: str):
    """Analyze processed GUE dataset."""
    datasets = []
    datasets += list(glob.glob(f"{data_dir}/prom/*"))
    datasets += list(glob.glob(f"{data_dir}/tf/*"))
    datasets += list(glob.glob(f"{data_dir}/splice/*"))
    datasets += list(glob.glob(f"{data_dir}/mouse/*"))
    datasets += list(glob.glob(f"{data_dir}/EMP/*"))
    datasets += list(glob.glob(f"{data_dir}/virus/*"))
    
    print(f"Found {len(datasets)} dataset directories")
    
    all_datasets = {}
    for ds_path in datasets:
        if not os.path.isdir(ds_path):
            continue
            
        ds_name, subset_name = ds_path.split("/")[-2:]
        if ds_name == "prom":
            ds_name = ds_name + subset_name.split("_")[1]

        if ds_name not in all_datasets:
            all_datasets[ds_name] = [0, 0]

        num_aligned = 0
        num_samples = 0
        subsets = [f"{ds_path}/train.jsonl", f"{ds_path}/dev.jsonl", f"{ds_path}/test.jsonl"]
        
        for subset in subsets:
            if not os.path.exists(subset):
                continue
            ds = MMapIndexedJsonlDataset(subset, verbose=False)
            num_samples += len(ds)
            for data in ds:
                if len(data['alignments']) > 0:
                    num_aligned += 1

        all_datasets[ds_name][0] += num_aligned
        all_datasets[ds_name][1] += num_samples
        
        if num_samples > 0:
            ratio = num_aligned / num_samples * 100
            print(f"{ds_name}-{subset_name}: {num_aligned}/{num_samples} ({ratio:.1f}%)")

    print("\n=== Summary by Task ===")
    for ds_name in sorted(all_datasets.keys()):
        if all_datasets[ds_name][1] > 0:
            ratio = all_datasets[ds_name][0] / all_datasets[ds_name][1] * 100
            print(f"{ds_name}: {ratio:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Prepare GUE dataset")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing GUE dataset")
    parser.add_argument("--ref", type=str, required=True,
                        help="Reference genome FASTA file (e.g., grch38.fasta)")
    parser.add_argument("--best_n", type=int, default=10,
                        help="Number of best alignments to keep")
    parser.add_argument("--mode", type=str, default="gue",
                        choices=["gue", "epi", "analyze"],
                        help="Processing mode")
    
    args = parser.parse_args()
    
    if args.mode == "gue":
        prepare_gue(args.data_dir, args.ref, args.best_n)
    elif args.mode == "epi":
        prepare_epi(args.data_dir, args.ref, args.best_n)
    elif args.mode == "analyze":
        analyze_gue(args.data_dir)


if __name__ == '__main__':
    main()
