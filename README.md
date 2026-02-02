# USM-AA

DNA sequence pretrain model.

## Install

```bash
conda create -n usmaa python=3.9 -y
conda activate usmaa
pip install -r requirements.txt
#install flash attention
pip install flash-attn --no-build-isolation
```

## Data Preparation

Data processing scripts are located in the `dataprocess/` directory.

### 1. HG38 Reference Genome

Process GRCh38 human reference genome.

```bash
python dataprocess/prepare_ref.py
```

- **Input**: `path/to/grch38.fasta`
- **Output**: `data/ref.fasta` + `data/ref.fasta.idx`

### 2. 1000 Genomes

Extract SNP and INDEL variants from 1000 Genomes Project VCF files.

```bash
python dataprocess/prepare_1kg.py
```

- **Input**: 1000 Genomes phased VCF directory
- **Output**: `data/1kg/1kg.jsonl` + `data/1kg/1kg.jsonl.idx`
- **Format**: Each record contains `sample_id`, `chrom`, `variants` (list of variant positions)

### 3. Multi-species DNA

Process multi-species DNA sequences for pretraining.

```bash
python dataprocess/prepare_dna.py
```

- **Input**: `dnabert/train.txt` (one DNA sequence per line)
- **Output**: `train.jsonl` + `train.jsonl.idx`

## Training

Modify `CONFIG_NAME` in `train.sh` to switch between pretraining and finetuning:

```bash
# For pretraining (MLM)
CONFIG_NAME="usmaa_100m_pretrain"

# For finetuning (Classification/Regression)
CONFIG_NAME="usmaa_100m_finetune"
```

Run training:
```bash
bash train.sh
```

## Extract Embedding

Extract sequence embeddings with alignment information:

```bash
bash embedding.sh
```

### Parameters

| Parameter | Description |
|-----------|-------------|
| `SEQUENCE` | DNA sequence (e.g., 60bp → 10 tokens) |
| `CHR_IDS` | Chromosome ID per token (1-22, 23=X, 24=Y, 0=unknown) |
| `POSITIONS` | Genomic position (bp) per token |
| `STRAND_IDS` | Strand per token (0=forward, 1=reverse) |
| `CIGAR_IDS` | CIGAR operation per token (7=match) |

### Token Alignment Format

Each 6mer token needs corresponding alignment info. For a 60bp sequence:
- Tokenized: `[BOS]` + 10 tokens + `[EOS]` = 12 tokens total
- Each alignment array should have 12 values

```bash
# Example: chr1, positions 1000-1054, forward strand
SEQUENCE="ACGTAGCATCGGATCTATCTATCGACACTTGGTTATCGATCTACGAGCATCTCGTTAGC"
CHR_IDS="1,1,1,1,1,1,1,1,1,1,1,1"
POSITIONS="0,1000,1006,1012,1018,1024,1030,1036,1042,1048,1054,248956422"
STRAND_IDS="0,0,0,0,0,0,0,0,0,0,0,0"
CIGAR_IDS="7,7,7,7,7,7,7,7,7,7,7,7"
```

CIGAR operations: `["M", "I", "D", "N", "S", "H", "P", "=", "X", "B"]` (index 7 = "=" match)