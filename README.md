<div align="center">

# USM-AA

**Alignment-Aware Genomic Coordinate Encoding for Long-Range DNA Modeling**

[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![DeepSpeed](https://img.shields.io/badge/DeepSpeed-Enabled-green.svg)](https://www.deepspeed.ai/)
[![License](https://img.shields.io/badge/License-PolyForm%20NC%201.0-lightgrey.svg)](https://polyformproject.org/licenses/noncommercial/1.0.0/)

*A powerful DNA sequence pretrain model with Flash Attention support*

[Quick Start](#quick-start) · [Data Preparation](#data-preparation) · [Training](#training) · [Usage](#extract-embedding)

</div>

---

## Overview

<div align="center">
<img src="fig/usmaa.png" alt="USM-AA Architecture" width="800"/>
</div>

USM-AA is a Transformer-based encoder model pretrained on a combination of the **human reference genome**, **population-scale genomes from 1000 Genomes Project**, and **reference genomes from multiple species**.

### Key Innovations

- **AARoPE (Alignment-Aware Rotary Position Embedding)**: A novel position encoding method designed specifically for genomic coordinates, enabling the model to capture long-range dependencies across both contiguous and non-contiguous genomic regions.

- **Alignment Embedding Module**: Processes alignment information including CIGAR strings, strand orientation, chromosome identifiers, and absolute genomic coordinates from Minimap2 alignment results.

- **RC-MLM (Reverse-Complement Masked Language Modeling)**: A specialized pretraining objective where forward-strand sequences are predicted normally, while reverse-strand sequences require reverse-complement predictions at masked positions.

As a result, USM-AA is better suited to interpret genomic sequences in the context of the reference genome, capturing both sequence content and coordinate-aware genomic structure. It achieves **excellent performance on multiple downstream tasks**.

---

## Features

- **DNA Sequence Modeling** - State-of-the-art pretrained model for DNA sequences
- **AARoPE** - Alignment-aware position encoding for genomic coordinates
- **Alignment Embedding** - Incorporates alignment information (CIGAR, strand, chromosome)
- **Flash Attention** - Optimized attention mechanism for faster training
- **DeepSpeed Integration** - Distributed training with ZeRO optimization
- **Multi-species Support** - Train on diverse genomic datasets
- **Easy to Use** - Simple API for embedding extraction

---

## Quick Start

### Installation

```bash
# Create conda environment
conda create -n usmaa python=3.9 -y
conda activate usmaa

# Install dependencies
pip install -r requirements.txt

# Install Flash Attention (for faster training)
pip install flash-attn --no-build-isolation
```

### Download Pretrained Models

```bash
# Download USM-AA 100M model (~204MB)
bash models/download.sh 100m

# Download USM-AA 1B model (~2GB)
bash models/download.sh 1b

# Download all models
bash models/download.sh all
```

| Model | Parameters | Download |
|-------|------------|----------|
| USM-AA 100M | 107M | [Google Drive](https://drive.google.com/file/d/1AzskU3T7As_nHW0HzwcvfOmzDBm7pLGw/view) |
| USM-AA 1B | 1B | [Google Drive](https://drive.google.com/file/d/1E3D4ty8QVHmzpascb7wcCWIrJe3af0RV/view) |

---

## Data Preparation

> All data processing scripts are located in the `scripts/` directory.

### Quick Start

```bash
# Step 1: Download data
bash scripts/download_pre_data.sh all   # Download pretrain data (ref + 1kg)
bash scripts/download_ft_data.sh all    # Download finetune data (gb + gue + nt)

# Step 2: Process data
bash scripts/prepare_pre_data.sh all    # Process pretrain data
bash scripts/prepare_ft_data.sh all     # Process finetune data
```

### Pretrain Data

#### Download

```bash
bash scripts/download_pre_data.sh ref   # Download GRCh38 reference genome
bash scripts/download_pre_data.sh 1kg   # Download 1000 Genomes VCF files
bash scripts/download_pre_data.sh all   # Download all
```

#### Process

```bash
bash scripts/prepare_pre_data.sh ref    # Process reference genome
bash scripts/prepare_pre_data.sh 1kg    # Process 1000 Genomes data
bash scripts/prepare_pre_data.sh dna    # Process DNA alignment data
bash scripts/prepare_pre_data.sh all    # Process all
```

### Finetune Data

#### Download

```bash
bash scripts/download_ft_data.sh gb     # Download Genomic Benchmarks
bash scripts/download_ft_data.sh gue    # Download GUE dataset
bash scripts/download_ft_data.sh nt     # Download NT benchmark
bash scripts/download_ft_data.sh all    # Download all
```

| Dataset | Source |
|---------|--------|
| **Genomic Benchmarks** | [HuggingFace](https://huggingface.co/datasets/katarinagresova) |
| **GUE** | [Google Drive](https://drive.google.com/file/d/1uOrwlf07qGQuruXqGXWMpPn8avBoW7T-/view) |
| **NT Benchmark** | [HuggingFace](https://huggingface.co/datasets/InstaDeepAI/nucleotide_transformer_downstream_tasks_revised) |

#### Process

```bash
bash scripts/prepare_ft_data.sh gb      # Process Genomic Benchmarks (10-fold CV)
bash scripts/prepare_ft_data.sh gue     # Process GUE dataset
bash scripts/prepare_ft_data.sh nt      # Process NT benchmark (10-fold CV)
bash scripts/prepare_ft_data.sh all     # Process all
```

### Configuration

Edit the paths in the scripts:

```bash
# scripts/download_pre_data.sh
DATA_DIR="../data"                      # Output directory

# scripts/prepare_pre_data.sh
REF_FILE="/path/to/grch38.fasta"        # Reference genome for alignment
```

---

### 1. Reference Genome

Download and process the GRCh38 human reference genome:

```bash
# Download
bash scripts/download_pre_data.sh ref

# Process
bash scripts/prepare_pre_data.sh ref
```

| Type | Description |
|------|-------------|
| **Source** | [NCBI GRCh38](https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/000/001/405/GCA_000001405.15_GRCh38/seqs_for_alignment_pipelines.ucsc_ids/GCA_000001405.15_GRCh38_no_alt_analysis_set.fna.gz) |
| **Output** | `data/ref.fasta` + `data/ref.fasta.idx` |

---

### 2. 1000 Genomes Project

Download and extract SNP and INDEL variants from 1000 Genomes Project VCF files:

```bash
# Download VCF files
bash scripts/download_pre_data.sh 1kg

# Process variants
bash scripts/prepare_pre_data.sh 1kg
```

| Type | Description |
|------|-------------|
| **Source** | [1000 Genomes FTP](http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000G_2504_high_coverage/working/20220422_3202_phased_SNV_INDEL_SV) |
| **Output** | `data/1kg.jsonl` + `data/1kg.jsonl.idx` |

> **Note**: This step may take a long time due to the large data volume (~3,200 samples).

<details>
<summary>Output Format</summary>

Each record contains:
- `sample_id` - Sample identifier
- `chrom` - Chromosome number
- `variants` - List of variant positions

</details>

---

### 3. Multi-species DNA

Process multi-species DNA sequences with alignment information:

```bash
bash scripts/prepare_pre_data.sh dna
```

| Type | Description |
|------|-------------|
| **Input** | Text file with one DNA sequence per line |
| **Output** | `data/dna.jsonl` + `data/dna.jsonl.idx` |

---

## Training

### Configuration Files

USM-AA uses YAML configuration files to control training. Different configs are provided for pretraining and finetuning:

| Config | Description |
|--------|-------------|
| `configs/usmaa_100m_pretrain.yaml` | Pretraining on reference genome + 1KG + multi-species DNA |
| `configs/usmaa_100m_finetune.yaml` | Finetuning 100M model on downstream classification tasks |
| `configs/usmaa_1b_finetune.yaml` | Finetuning 1B model on downstream classification tasks |

### Pretraining

```bash
bash scripts/train.sh configs/usmaa_100m_pretrain.yaml
```

### Finetuning

```bash
# Modify configs for your task:
#   - dataset.data_dir: path to your data
#   - dataset.num_classes: number of classes
#   - model.checkpoint: path to pretrained model

# Finetune 100M model
bash scripts/train.sh configs/usmaa_100m_finetune.yaml

# Finetune 1B model
bash scripts/train.sh configs/usmaa_1b_finetune.yaml
```

### Evaluation

```bash
# Evaluate best model on test set
bash scripts/train.sh configs/usmaa_100m_finetune.yaml --eval
```

| Argument | Description |
|----------|-------------|
| `--eval` | Evaluation only mode. Loads best model from `checkpoints/best_model.txt` and runs evaluation on test set. |

## Project Structure

```
usm-aa/
├── configs/          # Model & training configurations
├── scripts/          # Data processing & training scripts
├── src/
│   ├── bio/          # Bioinformatics utilities
│   ├── config/       # Configuration management
│   ├── data/         # Dataset classes
│   ├── distributed/  # Distributed training utilities
│   ├── engine/       # Training engine
│   ├── evaluator/    # Evaluation metrics
│   ├── loss/         # Loss functions
│   ├── model/        # Model architecture
│   ├── tokenizer/    # K-mer tokenizer
│   └── utils/        # General utilities
└── requirements.txt  # Dependencies
```

---

## License

This project is licensed under the [PolyForm Noncommercial License 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0/).

You are free to:
- **Use** — for any noncommercial purpose
- **Share** — copy and redistribute the material
- **Adapt** — make changes and new works based on the software

Under the following terms:
- **Noncommercial** — You may not use the material for commercial purposes
- **Notice** — You must ensure that anyone who gets a copy also gets a copy of these terms

See the [license](license) file for full details.