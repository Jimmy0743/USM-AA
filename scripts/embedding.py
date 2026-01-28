# Extract embeddings from USM-AA model
# Usage similar to DNABERT-2:
#   from embedding import USMAA
#   model, tokenizer = USMAA.from_pretrained(config_path, checkpoint_path)
#   inputs = tokenizer(dna, return_tensors='pt')
#   hidden_states = model(inputs["input_ids"])[0]
#   embedding_mean = torch.mean(hidden_states[0], dim=0)
try:
    from .src.tokenizer import DNAKmerTokenizer
    from .src.build_model import build_model_from_yaml, MODEL_CONFIGS
except ImportError:
    from src.tokenizer import DNAKmerTokenizer
    from src.build_model import build_model_from_yaml, MODEL_CONFIGS


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
