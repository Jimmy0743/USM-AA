# USM-AA

DNA sequence pretrain model.

## Install

```bash
conda create -n usmaa python=3.9 -y
conda activate usmaa
pip install -r requirements.txt
```

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