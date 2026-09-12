"""
Utility functions: Device setup, reproducibility, LLRD Optimizer, MLflow logging, and Checkpointing.
"""

import os
import json
import random
import numpy as np
import torch
from transformers import AutoTokenizer

from .config import (
    CATEGORIES, INTENTS, FLAGS, TRAJECTORIES,
    cat2id, intent2id, traj2id, flag2id,
    id2cat, id2intent, id2traj, id2flag
)

try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False


def set_seed(seed: int = 42):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Detects available hardware accelerator."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Hardware Accelerator: GPU ({torch.cuda.get_device_name(0)}) | CUDA {torch.version.cuda}")
    else:
        device = torch.device("cpu")
        print("Hardware Accelerator: CPU")
    return device


def build_llrd_optimizer(
    model: torch.nn.Module,
    base_lr: float = 2e-5,
    head_lr: float = 6e-5,
    decay_rate: float = 0.9,
    weight_decay: float = 0.01,
    num_layers: int = 12
) -> torch.optim.AdamW:
    """
    Layer-wise Learning Rate Decay (LLRD) Optimizer:
    Assigns higher learning rates to downstream perception heads and top transformer layers,
    decaying geometrically down to the lowest embedding layer.
    """
    param_groups = []

    # 1. Perception Heads, Turn-Transformer, Attention Pooler, and Uncertainty Loss
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "sentence_encoder" not in name:
            head_params.append(param)

    if head_params:
        param_groups.append({
            "params": head_params,
            "lr": head_lr,
            "weight_decay": weight_decay
        })

    # 2. Transformer layers in reverse order (top to bottom)
    if hasattr(model, "sentence_encoder"):
        for layer_idx in range(num_layers - 1, -1, -1):
            layer_lr = base_lr * (decay_rate ** (num_layers - 1 - layer_idx))
            layer_params = [
                p for n, p in model.sentence_encoder.named_parameters()
                if f"layer.{layer_idx}." in n and p.requires_grad
            ]
            if layer_params:
                param_groups.append({
                    "params": layer_params,
                    "lr": layer_lr,
                    "weight_decay": weight_decay
                })

        # 3. Word embeddings & position embeddings
        embed_lr = base_lr * (decay_rate ** num_layers)
        embed_params = [
            p for n, p in model.sentence_encoder.named_parameters()
            if ("embeddings" in n or "word_embeddings" in n) and p.requires_grad
        ]
        if embed_params:
            param_groups.append({
                "params": embed_params,
                "lr": embed_lr,
                "weight_decay": weight_decay
            })
    else:
        # Fallback for standard models
        param_groups.append({
            "params": [p for p in model.parameters() if p.requires_grad],
            "lr": base_lr,
            "weight_decay": weight_decay
        })

    return torch.optim.AdamW(param_groups)


def init_mlflow_run(experiment_name: str = "DeBERTa-Hierarchical-NLU", run_name: str = None):
    """Initializes MLflow run if MLflow is installed and available (standard in Azure ML)."""
    if not HAS_MLFLOW:
        print("[MLflow] MLflow not installed. Skipping MLflow tracking.")
        return None
    try:
        mlflow.set_experiment(experiment_name)
        run = mlflow.start_run(run_name=run_name)
        print(f"[MLflow] Started MLflow Run: {run.info.run_id} under experiment: '{experiment_name}'")
        return run
    except Exception as e:
        print(f"[MLflow] Warning: Could not start MLflow run: {e}")
        return None


def log_mlflow_metrics(metrics: dict, step: int = None):
    """Logs a dictionary of metrics to MLflow."""
    if not HAS_MLFLOW:
        return
    try:
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(k, float(v), step=step)
    except Exception as e:
        print(f"[MLflow] Warning: Metric logging failed: {e}")


def log_mlflow_params(params: dict):
    """Logs hyperparameters to MLflow."""
    if not HAS_MLFLOW:
        return
    try:
        for k, v in params.items():
            mlflow.log_param(k, str(v))
    except Exception as e:
        print(f"[MLflow] Warning: Parameter logging failed: {e}")


def save_nlu_checkpoint(model, tokenizer, output_dir: str, metadata: dict = None):
    """Saves the PyTorch model state, tokenizer, and complete label mappings to disk."""
    os.makedirs(output_dir, exist_ok=True)

    # Save PyTorch weights
    model_path = os.path.join(output_dir, "best_deberta_hierarchical_nlu.pt")
    torch.save(model.state_dict(), model_path)

    # Save Tokenizer
    tokenizer.save_pretrained(output_dir)

    # Save Label Mappings & Metadata
    config_dict = {
        "categories": CATEGORIES,
        "intents": INTENTS,
        "flags": FLAGS,
        "trajectories": TRAJECTORIES,
        "cat2id": cat2id,
        "id2cat": id2cat,
        "intent2id": intent2id,
        "id2intent": id2intent,
        "traj2id": traj2id,
        "id2traj": id2traj,
        "flag2id": flag2id,
        "id2flag": id2flag,
        "metadata": metadata or {}
    }
    with open(os.path.join(output_dir, "nlu_config.json"), "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)

    print(f"Saved complete NLU artifacts and checkpoint to: {output_dir}")
