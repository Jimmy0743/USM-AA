# USM-AA

DNA sequence pretrain model.

## Install

```bash
conda create -n usmaa python=3.9 -y
conda activate usmaa
pip install -r requirements.txt
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

### DDP Training
```bash
bash train.sh
```

### DeepSpeed Training
```bash
bash train_deepspeed.sh
```

## Extract Embedding

```python
from embedding import USMAA

model, tokenizer = USMAA.from_pretrained(
    config_path="config.yaml", 
    checkpoint_path="/path/to/model.safetensors",  # your model checkpoint
    device="cuda"
)

dna = "ACGTAGCATCGGATCTATCTATCGACACTTGGTTATCGATCTACGAGCATCTCGTTAGC"
inputs = tokenizer(dna, return_tensors='pt')
hidden_states = model(inputs["input_ids"].to("cuda"))[0]

# Mean pooling
embedding = hidden_states.mean(dim=1)
print(f"Embedding shape: {embedding.shape}")
```