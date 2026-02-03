"""Extract embeddings from USM-AA model.

This module provides a simple interface for loading USM-AA models and
extracting DNA sequence embeddings.

"""
try:
    from .src.tokenizer import DNAKmerTokenizer
    from .src.build_model import build_model_from_yaml, MODEL_CONFIGS
except ImportError:
    from src.tokenizer import DNAKmerTokenizer
    from src.build_model import build_model_from_yaml, MODEL_CONFIGS


class USMAA:
    """USM-AA model loader with convenience methods.
    
    Provides a simple interface for loading pretrained USM-AA models
    and extracting embeddings from DNA sequences.
    
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
            config_path: Path to YAML configuration file.
            checkpoint_path: Path to safetensors checkpoint file.
            device: Device to load the model on ("cpu" or "cuda").
            hf_tokenizer: If True, wrap tokenizer with HuggingFace-like interface.
        
        Returns:
            Tuple of (model, tokenizer) ready for inference.
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


def load_usmaa_model(config_path: str, checkpoint_path: str, device: str = "cpu") -> tuple:
    """Load USM-AA model (backward compatible interface).
    
    Args:
        config_path: Path to YAML configuration file.
        checkpoint_path: Path to safetensors checkpoint file.
        device: Device to load the model on.
        
    Returns:
        Tuple of (model, tokenizer).
    """
    return USMAA.from_pretrained(config_path, checkpoint_path, device)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Extract embeddings from USM-AA model"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to safetensors checkpoint file"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use (cpu/cuda)"
    )
    parser.add_argument(
        "--sequence",
        type=str,
        default=None,
        help="DNA sequence to embed (optional)"
    )
    args = parser.parse_args()
    
    # Load model
    model, tokenizer = USMAA.from_pretrained(args.config, args.checkpoint, args.device)
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")

    # Use provided sequence or demo sequence
    if args.sequence:
        dna = args.sequence
    else:
        dna = "ACGTAGCATCGGATCTATCTATCGACACTTGGTTATCGATCTACGAGCATCTCGTTAGC"
        print(f"\nNo sequence provided, using demo: {dna}")

    # Tokenize input
    inputs = tokenizer(dna, return_tensors='pt')
    input_ids = inputs["input_ids"].to(args.device)
    print(f"Input shape: {input_ids.shape}")

    # Get hidden states
    hidden_states = model(input_ids)[0]  # [1, sequence_length, hidden_dim]
    print(f"Hidden states shape: {hidden_states.shape}")

    # Compute mean pooled embedding
    embedding_mean = hidden_states.mean(dim=1)
    print(f"Mean embedding shape: {embedding_mean.shape}")
    print(f"Embedding: {embedding_mean}")
