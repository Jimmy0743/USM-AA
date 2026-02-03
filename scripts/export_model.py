
import torch
from src.model.modeling_usmaa import USMAAModel, USMAAConfig


def convert_usmaa_to_hf(ckpt_path: str, output_dir: str, model_type: str = "usm-aa-100m"):
    """Convert original USM-AA checkpoint to HuggingFace format.

    Args:
        ckpt_path: Path to original checkpoint (.pt file)
        output_dir: Output directory for HuggingFace model
        model_type: One of "usm-aa-100m", "usm-aa-500m", "usm-aa-1b"
    """
    import os
    CONFIG_MAP = {
        "usm-aa-100m": dict(num_layers=12, num_heads=12, embedding_dim=768, ffn_dim=4096, use_gate=True),
        "usm-aa-500m": dict(num_layers=24, num_heads=20, embedding_dim=1280, ffn_dim=6912, use_gate=True),
        "usm-aa-1b": dict(num_layers=36, num_heads=32, embedding_dim=1536, ffn_dim=8192, use_gate=True),
    }
    mcfg = CONFIG_MAP[model_type]
    config = USMAAConfig(
        vocab_size=512,
        embedding_dim=mcfg["embedding_dim"],
        num_layers=mcfg["num_layers"],
        num_heads=mcfg["num_heads"],
        ffn_dim=mcfg["ffn_dim"],
        use_gate=mcfg["use_gate"],
    )
    model = USMAAModel(config)
    # Load original checkpoint
    state_dict = torch.load(ckpt_path, map_location="cpu")
    if "model" in state_dict:
        state_dict = state_dict["model"]

    # Convert state dict keys if needed
    new_state_dict = {}
    for k, v in state_dict.items():
        # Remove "module." prefix if present (from DDP)
        if k.startswith("module."):
            k = k[7:]
        new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=False)
    # Save to HuggingFace format
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    config.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")
    return model
