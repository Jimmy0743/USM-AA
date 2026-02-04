"""
Prepare Nucleotide Transformer (NT) benchmark dataset for USM-AA training.

This script processes the NT benchmark dataset, adding alignment information
using Minimap2 for various genomic tasks including enhancer prediction,
histone modification, promoter detection, and splice site classification.
"""
import argparse
import glob
import os

import tqdm
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from data_utils import MMapIndexedJsonlDataset, MMapIndexedJsonlDatasetBuilder
from prepare_dna import MinimapAligner


def download_dataset(data_dir: str):
    """Download NT benchmark dataset from HuggingFace and split by task."""
    from datasets import load_dataset
    
    os.makedirs(data_dir, exist_ok=True)
    
    print("Downloading NT benchmark from HuggingFace...")
    print("Dataset: InstaDeepAI/nucleotide_transformer_downstream_tasks_revised")
    print("")
    
    ds = load_dataset("InstaDeepAI/nucleotide_transformer_downstream_tasks_revised")
    
    # Process train and test splits
    for split_name in ds.keys():
        print(f"\n=== Processing split: {split_name} ===")
        split_data = ds[split_name]
        
        # Convert to DataFrame
        df = split_data.to_pandas()
        print(f"Total samples: {len(df)}")
        print(f"Columns: {list(df.columns)}")
        
        # Ensure sequence column is preserved
        if 'seq' in df.columns and 'sequence' not in df.columns:
            df = df.rename(columns={'seq': 'sequence'})
        
        # Get all unique tasks
        if 'task' not in df.columns:
            print("WARNING: No 'task' column found!")
            continue
            
        tasks = df['task'].unique()
        print(f"Found {len(tasks)} tasks: {list(tasks)}")
        
        # Split by task and save
        for task_name in tasks:
            task_df = df[df['task'] == task_name].copy()
            
            # Create task directory
            task_dir = f"{data_dir}/{task_name}"
            os.makedirs(task_dir, exist_ok=True)
            
            # Remove 'task' column since it's now part of directory structure
            if 'task' in task_df.columns:
                task_df = task_df.drop(columns=['task'])
            
            # Remove 'name' column if exists (contains chr position info)
            if 'name' in task_df.columns:
                task_df = task_df.drop(columns=['name'])
            
            # Determine output filename based on split
            if split_name == 'train':
                output_file = f"{task_dir}/train.csv"
            else:
                output_file = f"{task_dir}/test.csv"
            
            task_df.to_csv(output_file, index=False)
            print(f"  {task_name}: {len(task_df)} samples -> {output_file}")
    
    print("\n=== Download complete! ===")
    print(f"Data saved to: {data_dir}")


def prepare_nt(data_dir: str, ref_file: str, best_n: int = 10):
    """Process NT benchmark dataset with alignment information."""
    # Collect all csv files from different task categories
    files = []
    files += list(glob.glob(f"{data_dir}/enhancers*/*.csv"))
    files += list(glob.glob(f"{data_dir}/H*/*.csv"))
    files += list(glob.glob(f"{data_dir}/promoter*/*.csv"))
    files += list(glob.glob(f"{data_dir}/splice_sites*/*.csv"))
    
    print(f"Found {len(files)} csv files")
    
    aligner = MinimapAligner(ref_file, best_n=best_n)
    
    for path in files:
        print(f"\nInput: {path}")
        name = os.path.basename(path).split('.')[0]
        output = f"{os.path.dirname(path)}/{name}.jsonl"
        print(f"Output: {output}")
        
        builder = MMapIndexedJsonlDatasetBuilder(output)
        
        # Read CSV file
        df = pd.read_csv(path)
            
        n_aligned = 0
        for _, row in tqdm.tqdm(df.iterrows(), total=len(df)):
            # Handle different column names
            sequence = row['sequence'] if "sequence" in row else row['seq']
            alignments = aligner(sequence)
            if len(alignments) > 0:
                n_aligned += 1
            label = row['label']
            data = {"sequence": sequence, "alignments": alignments, "label": label}
            builder.add_item(data)
            builder.end_document()
            
        print(f"Aligned: {n_aligned}/{len(df)} ({n_aligned/len(df)*100:.1f}%)")
        builder.finalize()


def prepare_nt_cv(data_dir: str, ref_file: str, best_n: int = 10,
                  n_folds: int = 10, seed: int = 42):
    """Process NT benchmark with cross-validation splits (train only)."""
    # Get dataset directories
    files = []
    files += list(glob.glob(f"{data_dir}/enhancers*"))
    files += list(glob.glob(f"{data_dir}/H*"))
    files += list(glob.glob(f"{data_dir}/promoter*"))
    files += list(glob.glob(f"{data_dir}/splice_sites*"))
    
    print(f"Found {len(files)} dataset directories")
    
    aligner = MinimapAligner(ref_file, best_n=best_n)
    
    for path in files:
        if not os.path.isdir(path):
            continue
            
        print(f"\nProcessing: {path}")
        output = f"{path}/cv.jsonl"
        print(f"Output: {output}")
        
        # Check for required files
        train_file = f"{path}/train.csv"
        if not os.path.exists(train_file):
            print(f"Skipping: missing train file")
            continue
        
        builder = MMapIndexedJsonlDatasetBuilder(output)
        
        # Only use train data for cross-validation
        df = pd.read_csv(train_file)
        
        # Create stratified k-fold splits
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for fold_number, (_, test_index) in enumerate(
                skf.split(X=df.index, y=df["label"])
        ):
            df.loc[df.index[test_index], "fold_id"] = fold_number

        n_aligned = 0
        for _, row in tqdm.tqdm(df.iterrows(), total=len(df)):
            sequence = row['sequence'] if "sequence" in row else row['seq']
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


def analyze_nt(data_dir: str):
    """Analyze processed NT benchmark dataset."""
    files = []
    files += list(glob.glob(f"{data_dir}/enhancers*"))
    files += list(glob.glob(f"{data_dir}/H*"))
    files += list(glob.glob(f"{data_dir}/promoter*"))
    files += list(glob.glob(f"{data_dir}/splice_sites*"))
    
    print(f"Found {len(files)} dataset directories\n")
    
    for path in files:
        if not os.path.isdir(path):
            continue
            
        output = f"{path}/cv.jsonl"
        if not os.path.exists(output):
            continue
            
        ds = MMapIndexedJsonlDataset(output)
        num_aligned = 0
        num_samples = len(ds)
        
        for data in ds:
            if len(data["alignments"]) > 0:
                num_aligned += 1
                
        name = os.path.basename(path)
        ratio = num_aligned / num_samples * 100 if num_samples > 0 else 0
        print(f"{name}: {num_aligned}/{num_samples} ({ratio:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Prepare NT benchmark dataset")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing NT benchmark dataset")
    parser.add_argument("--ref", type=str, default="",
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
        prepare_nt(args.data_dir, args.ref, args.best_n)
    elif args.mode == "cv":
        prepare_nt_cv(args.data_dir, args.ref, args.best_n, args.n_folds, args.seed)
    elif args.mode == "analyze":
        analyze_nt(args.data_dir)


if __name__ == '__main__':
    main()
