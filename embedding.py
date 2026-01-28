# Extract embeddings from USM-AA model
# Usage similar to DNABERT-2:
#   from embedding import USMAA
#   model, tokenizer = USMAA.from_pretrained(config_path, checkpoint_path)
#   inputs = tokenizer(dna, return_tensors='pt')
#   hidden_states = model(inputs["input_ids"])[0]
#   embedding_mean = torch.mean(hidden_states[0], dim=0)

import torch
from typing import List, Union, Dict, Optional

try:
    from .src.tokenizer import DNAKmerTokenizer
    from .src.build_model import build_model_from_yaml, MODEL_CONFIGS
except ImportError:
    from src.tokenizer import DNAKmerTokenizer
    from src.build_model import build_model_from_yaml, MODEL_CONFIGS


class USMAATokenizer:
    """Tokenizer wrapper with HuggingFace-like interface."""
    
    def __init__(self, tokenizer: DNAKmerTokenizer):
        self._tokenizer = tokenizer
    
    @property
    def vocab_size(self) -> int:
        return self._tokenizer.vocab_size
    
    @property
    def pad_token_id(self) -> int:
        return self._tokenizer.pad_id
    
    @property
    def bos_token_id(self) -> int:
        return self._tokenizer.bos_id
    
    @property
    def eos_token_id(self) -> int:
        return self._tokenizer.eos_id
    
    def __call__(
        self,
        text: Union[str, List[str]],
        return_tensors: str = "pt",
        padding: bool = True,
        max_length: int = -1,
        add_special_tokens: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Tokenize DNA sequences.
        
        Args:
            text: DNA sequence or list of sequences
            return_tensors: Return type ("pt" for PyTorch tensor)
            padding: Whether to pad sequences to same length
            max_length: Maximum length (-1 for no limit)
            add_special_tokens: Whether to add BOS/EOS tokens
        
        Returns:
            Dict with "input_ids" and "attention_mask"
        """
        if isinstance(text, str):
            text = [text]
        
        # Tokenize each sequence
        token_ids_list = []
        for seq in text:
            seq = seq.upper().replace(" ", "")
            tokens = self._tokenizer.encode(
                seq,
                bos=add_special_tokens,
                eos=add_special_tokens,
                max_length=max_length,
                to_tensor=True,
            )
            token_ids_list.append(tokens)
        
        # Pad to same length
        max_len = max(t.size(0) for t in token_ids_list)
        padded_tokens = []
        attention_masks = []
        
        for tokens in token_ids_list:
            seq_len = tokens.size(0)
            mask = torch.ones(seq_len, dtype=torch.long)
            
            if padding and seq_len < max_len:
                pad_len = max_len - seq_len
                tokens = torch.cat([tokens, torch.full((pad_len,), self.pad_token_id, dtype=torch.long)])
                mask = torch.cat([mask, torch.zeros(pad_len, dtype=torch.long)])
            
            padded_tokens.append(tokens)
            attention_masks.append(mask)
        
        input_ids = torch.stack(padded_tokens, dim=0)
        attention_mask = torch.stack(attention_masks, dim=0)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
    
    def decode(self, token_ids: torch.Tensor) -> str:
        return self._tokenizer.decode(token_ids)


class USMAA:
    """USM-AA model loader with convenience methods.
    
    Example:
        model, tokenizer = USMAA.from_pretrained("config.yaml", "model.safetensors")
        
        dna = "ACGTAGCATCGGATCTATCTATCGACACTTGGTTATCGATCTACGAGCATCTCGTTAGC"
        inputs = tokenizer(dna, return_tensors='pt')
        hidden_states = model(inputs["input_ids"])[0]  # [1, seq_len, dim]
        
        # Mean pooling
        embedding_mean = torch.mean(hidden_states[0], dim=0)
        print(embedding_mean.shape)  # [dim]
        
        # Max pooling
        embedding_max = torch.max(hidden_states[0], dim=0)[0]
        print(embedding_max.shape)  # [dim]
    """
    
    @staticmethod
    def from_pretrained(
        config_path: str, 
        checkpoint_path: str, 
        device: str = "cpu",
        hf_tokenizer: bool = True,
    ) -> tuple:
        """Load model and tokenizer from config and checkpoint.
        
        Args:
            config_path: Path to YAML config file
            checkpoint_path: Path to safetensors checkpoint
            device: Device to load model on
            hf_tokenizer: If True, wrap tokenizer with HF-like interface
        
        Returns:
            (model, tokenizer) tuple
        """
        # Load model from yaml config
        model, tokenizer, config = build_model_from_yaml(
            config_path=config_path,
            pretrained_path=checkpoint_path,
            device=device,
        )
        model.eval()
        
        # Optionally wrap tokenizer with HF-like interface
        if hf_tokenizer:
            tokenizer = USMAATokenizer(tokenizer)
        
        return model, tokenizer


# Convenience function for backward compatibility
def load_usmaa_model(config_path: str, checkpoint_path: str, device: str = "cpu"):
    """Load USM-AA model (backward compatible interface)."""
    return USMAA.from_pretrained(config_path, checkpoint_path, device)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract embeddings from USM-AA model")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to safetensors checkpoint")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--sequence", type=str, default=None, help="DNA sequence to embed")
    args = parser.parse_args()
    
    # Load model (similar to DNABERT-2)
    model, tokenizer = USMAA.from_pretrained(args.config, args.checkpoint, args.device)
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")
    
    # Example usage
    if args.sequence:
        dna = args.sequence
    else:
        dna = "ACGTAGCATCGGATCTATCTATCGACACTTGGTTATCGATCTACGAGCATCTCGTTAGC"
        print(f"\nNo sequence provided, using demo: {dna}")
    
    # Tokenize (like HuggingFace)
    inputs = tokenizer(dna, return_tensors='pt')
    input_ids = inputs["input_ids"].to(args.device)
    
    print(f"Input shape: {input_ids.shape}")
    
    # Get hidden states (like DNABERT-2)
    hidden_states = model(input_ids)[0]  # [1, sequence_length, hidden_dim]
    print(f"Hidden states shape: {hidden_states.shape}")
    
    # Embedding with mean pooling
    embedding_mean = hidden_states.mean(dim=1)
    print(f"Mean embedding shape: {embedding_mean.shape}")
    print(f"Embedding: {embedding_mean}")
