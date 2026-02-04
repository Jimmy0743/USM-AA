"""
Prepare Genomic Benchmarks dataset for USM-AA training.

This script downloads and processes the Genomic Benchmarks dataset,
adding alignment information using Minimap2.
"""
import argparse
import glob
import os

import numpy as np
import tqdm
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import StratifiedKFold

from data_utils import MMapIndexedJsonlDataset, MMapIndexedJsonlDatasetBuilder
from prepare_dna import MinimapAligner

# HuggingFace dataset names
DATASETS = {
    "demo_coding_vs_intergenomic_seqs": "katarinagresova/Genomic_Benchmarks_demo_coding_vs_intergenomic_seqs",
    "demo_human_or_worm": "katarinagresova/Genomic_Benchmarks_demo_human_or_worm",
    "drosophila_enhancers_stark": "katarinagresova/Genomic_Benchmarks_drosophila_enhancers_stark",
    "dummy_mouse_enhancers_ensembl": "katarinagresova/Genomic_Benchmarks_dummy_mouse_enhancers_ensembl",
    "human_enhancers_cohn": "katarinagresova/Genomic_Benchmarks_human_enhancers_cohn",
    "human_enhancers_ensembl": "katarinagresova/Genomic_Benchmarks_human_enhancers_ensembl",
    "human_nontata_promoters": "katarinagresova/Genomic_Benchmarks_human_nontata_promoters",
    "human_ocr_ensembl": "katarinagresova/Genomic_Benchmarks_human_ocr_ensembl",
    "human_ensembl_regulatory": "katarinagresova/Genomic_Benchmarks_human_ensembl_regulatory"
}


def download_dataset(data_dir: str):
    """Download Genomic Benchmarks dataset from HuggingFace to CSV files."""
    os.makedirs(data_dir, exist_ok=True)
    
    for name, hf_name in DATASETS.items():
        print(f"\nDownloading {name} from {hf_name}...")
        try:
            ds = load_dataset(hf_name)
        except Exception as e:
            print(f"Failed to download {name}: {e}")
            continue
        
        save_dir = f"{data_dir}/{name}"
        os.makedirs(save_dir, exist_ok=True)
        
        for split in ["train", "test"]:
            if split not in ds:
                print(f"  Split '{split}' not found, skipping...")
                continue
            
            split_data = ds[split]
            seqs = []
            labels = []
            
            for item in tqdm.tqdm(split_data, desc=f"{name}-{split}"):
                seqs.append(item['seq'])
                labels.append(item['label'])
            
            df = pd.DataFrame({'sequence': seqs, 'label': labels})
            df.to_csv(f"{save_dir}/{split}.csv", index=False)
            print(f"  Saved {len(df)} samples to {save_dir}/{split}.csv")


def prepare_gb(data_dir: str, ref_file: str, best_n: int = 10):
    """Process Genomic Benchmarks with alignment information."""
    aligner = MinimapAligner(ref_file, best_n=best_n)
    files = list(glob.glob(f"{data_dir}/*/*.csv"))
    
    for path in files:
        print("Input:", path)
        name = os.path.basename(path).split('.')[0]
        output = f"{os.path.dirname(path)}/{name}.jsonl"
        print("Output:", output)
        
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


def prepare_gb_cv(data_dir: str, ref_file: str, best_n: int = 10, 
                  n_folds: int = 10, seed: int = 42):
    """Process Genomic Benchmarks with cross-validation splits (train only)."""
    aligner = MinimapAligner(ref_file, best_n=best_n)
    dataset_paths = list(glob.glob(f"{data_dir}/*"))
    
    for path in dataset_paths:
        if not os.path.isdir(path):
            continue
        print(f"\nProcessing: {path}")
        output = f"{path}/cv.jsonl"
        
        # Only use train data for cross-validation
        train_file = f"{path}/train.csv"
        if not os.path.exists(train_file):
            print(f"Skipping: missing train file")
            continue
        
        builder = MMapIndexedJsonlDatasetBuilder(output)
        df = pd.read_csv(train_file)
        
        # Create stratified k-fold splits
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for fold_number, (_, test_index) in enumerate(
                skf.split(X=df.index, y=df["label"])
        ):
            df.loc[df.index[test_index], "fold_id"] = fold_number

        n_aligned = 0
        for _, row in tqdm.tqdm(df.iterrows(), total=len(df)):
            sequence = row['sequence']
            alignments = aligner(sequence)
            if len(alignments) > 0:
                n_aligned += 1
            label = row['label']
            data = {
                "sequence": sequence, 
                "alignments": alignments, 
                "label": label, 
                "fold_id": int(row["fold_id"])
            }
            builder.add_item(data)
            builder.end_document()
            
        print(f"Aligned: {n_aligned}/{len(df)} ({n_aligned/len(df)*100:.1f}%)")
        builder.finalize()


def analyze_dataset(data_dir: str):
    """Analyze processed Genomic Benchmarks dataset."""
    dataset_paths = list(glob.glob(f"{data_dir}/*"))
    
    for path in dataset_paths:
        if not os.path.isdir(path):
            continue
        name = path.split('/')[-1]
        num_aligned = 0
        num_samples = 0
        lengths = []
        
        subsets = [f"{path}/train.jsonl", f"{path}/test.jsonl"]
        for subset in subsets:
            if not os.path.exists(subset):
                continue
            ds = MMapIndexedJsonlDataset(subset)
            num_samples += len(ds)
            for data in ds:
                lengths.append(len(data["sequence"]))
                if len(data['alignments']) > 0:
                    num_aligned += 1

        if num_samples > 0:
            lengths = np.array(lengths)
            print(f"{name}: aligned={num_aligned}/{num_samples} ({num_aligned/num_samples*100:.1f}%)")
            print(f"  Length: max={lengths.max()}, min={lengths.min()}, "
                  f"mean={lengths.mean():.1f}, q75={np.quantile(lengths, 0.75):.1f}")


def main():
    parser = argparse.ArgumentParser(description="Prepare Genomic Benchmarks dataset")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory to store/read dataset")
    parser.add_argument("--ref", type=str, required=True,
                        help="Reference genome FASTA file (e.g., grch38.fasta)")
    parser.add_argument("--best_n", type=int, default=10,
                        help="Number of best alignments to keep")
    parser.add_argument("--mode", type=str, default="prepare",
                        choices=["download", "prepare", "cv", "analyze"],
                        help="Processing mode")
    parser.add_argument("--n_folds", type=int, default=10,
                        help="Number of cross-validation folds")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for CV splits")
    
    args = parser.parse_args()
    
    if args.mode == "download":
        download_dataset(args.data_dir)
    elif args.mode == "prepare":
        prepare_gb(args.data_dir, args.ref, args.best_n)
    elif args.mode == "cv":
        prepare_gb_cv(args.data_dir, args.ref, args.best_n, args.n_folds, args.seed)
    elif args.mode == "analyze":
        analyze_dataset(args.data_dir)


if __name__ == "__main__":
    main()
