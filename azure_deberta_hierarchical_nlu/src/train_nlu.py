"""
Stage 2: Hierarchical Turn-Transformer NLU Multi-Task Fine-Tuning.
Trains Utterance Encoder (DeBERTa-v3) + Turn-Transformer + Kendall Homoscedastic Loss.
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from .config import (
    CATEGORIES, INTENTS, FLAGS, TRAJECTORIES,
    DEFAULT_PRETRAINED_MODEL, DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS, DEFAULT_BASE_LR, DEFAULT_HEAD_LR,
    DEFAULT_LLRD_DECAY, DEFAULT_MAX_TURNS, DEFAULT_MAX_TURN_LEN
)
from .dataset import HierarchicalDialogueDataset, load_and_prepare_nlu_dataframe
from .model import DeBERTaHierarchicalCustomerSupportTransformer
from .utils import (
    get_device, set_seed, build_llrd_optimizer,
    init_mlflow_run, log_mlflow_metrics, log_mlflow_params,
    save_nlu_checkpoint
)


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 2: DeBERTa Hierarchical Turn-Transformer NLU Fine-Tuning")
    parser.add_argument("--bitext_data", type=str, required=True, help="Path to Bitext customer support dataset CSV")
    parser.add_argument("--twitter_data", type=str, default=None, help="Optional path to Twitter augmented dialogue dataset CSV")
    parser.add_argument("--pretrained_backbone", type=str, default=DEFAULT_PRETRAINED_MODEL, help="Path to adapted MLM backbone or HuggingFace model ID")
    parser.add_argument("--output_dir", type=str, default="./outputs/deberta_hierarchical_nlu", help="Output directory for trained model")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of fine-tuning epochs")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="Training batch size")
    parser.add_argument("--base_lr", type=float, default=DEFAULT_BASE_LR, help="Base learning rate for lower DeBERTa layers")
    parser.add_argument("--head_lr", type=float, default=DEFAULT_HEAD_LR, help="Head learning rate for turn transformer and perception heads")
    parser.add_argument("--llrd_decay", type=float, default=DEFAULT_LLRD_DECAY, help="Layer-wise LR decay factor")
    parser.add_argument("--patience", type=int, default=2, help="Early stopping patience")
    parser.add_argument("--max_turns", type=int, default=DEFAULT_MAX_TURNS, help="Maximum dialogue turns per conversation")
    parser.add_argument("--max_turn_len", type=int, default=DEFAULT_MAX_TURN_LEN, help="Maximum token length per turn")
    parser.add_argument("--sample_size", type=int, default=None, help="Optional maximum row limit for Twitter data")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    init_mlflow_run(experiment_name="DeBERTa-Hierarchical-NLU", run_name="Stage2-Hierarchical-FineTuning")
    log_mlflow_params(vars(args))

    print("=" * 75)
    print("STAGE 2: HIERARCHICAL TURN-TRANSFORMER NLU MULTI-TASK FINE-TUNING")
    print(f"Bitext Dataset     : {args.bitext_data}")
    print(f"Twitter Dataset    : {args.twitter_data}")
    print(f"Pretrained Backbone: {args.pretrained_backbone}")
    print(f"Output Directory   : {args.output_dir}")
    print(f"Epochs             : {args.epochs} | Batch Size: {args.batch_size}")
    print(f"Base LR: {args.base_lr} | Head LR: {args.head_lr} | LLRD Decay: {args.llrd_decay}")
    print("=" * 75)

    # 1. Load and Prepare Blended Dataset
    df = load_and_prepare_nlu_dataframe(
        bitext_path=args.bitext_data,
        twitter_path=args.twitter_data,
        sample_size=args.sample_size
    )

    train_df, val_df = train_test_split(df, test_size=0.15, random_state=args.seed, stratify=df["category"])
    print(f"Train Dialogues: {len(train_df):,} | Validation Dialogues: {len(val_df):,}")

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_backbone)

    train_dataset = HierarchicalDialogueDataset(
        train_df, tokenizer, max_turns=args.max_turns, max_turn_len=args.max_turn_len
    )
    val_dataset = HierarchicalDialogueDataset(
        val_df, tokenizer, max_turns=args.max_turns, max_turn_len=args.max_turn_len
    )

    use_amp = (device.type == "cuda")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=use_amp)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size * 2, shuffle=False, pin_memory=use_amp)

    # 2. Instantiate Model
    print(f"Initializing DeBERTa Hierarchical Customer Support Transformer from: {args.pretrained_backbone}...")
    model = DeBERTaHierarchicalCustomerSupportTransformer(
        model_name=args.pretrained_backbone,
        num_categories=len(CATEGORIES),
        num_intents=len(INTENTS),
        num_flags=len(FLAGS),
        num_trajectories=len(TRAJECTORIES),
        max_turns=args.max_turns
    ).to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Initialized with {trainable_params:,} trainable parameters.")

    # 3. Optimizer & Scheduler
    optimizer = build_llrd_optimizer(
        model,
        base_lr=args.base_lr,
        head_lr=args.head_lr,
        decay_rate=args.llrd_decay,
        weight_decay=0.01
    )

    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps
    )

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val_loss = float("inf")
    patience_counter = 0
    task_names = ["Intent", "Category", "Flags", "Trajectory", "Escalation", "Effort"]

    # 4. Training Loop
    for epoch in range(args.epochs):
        model.train()
        total_train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")

        for batch in pbar:
            optimizer.zero_grad()
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
                out = model(**batch)
                loss = out["loss"]

            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            scheduler.step()
            total_train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = total_train_loss / max(1, len(train_loader))

        # 5. Validation Loop
        model.eval()
        total_val_loss = 0.0
        correct_intent = 0
        correct_cat = 0
        correct_traj = 0
        total_samples = 0
        esc_errors = []
        eff_errors = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]"):
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

                with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
                    out = model(**batch)

                total_val_loss += out["loss"].item()
                pred_intent = out["intent_logits"].argmax(dim=-1)
                pred_cat = out["cat_logits"].argmax(dim=-1)
                pred_traj = out["traj_logits"].argmax(dim=-1)

                correct_intent += (pred_intent == batch["intent_label"]).sum().item()
                correct_cat += (pred_cat == batch["category_label"]).sum().item()
                correct_traj += (pred_traj == batch["trajectory_label"]).sum().item()
                total_samples += batch["intent_label"].size(0)

                esc_errors.extend(torch.abs(out["escalation_pred"] - batch["escalation_label"]).cpu().numpy())
                eff_errors.extend(torch.abs(out["effort_pred"] - batch["effort_label"]).cpu().numpy())

        avg_val_loss = total_val_loss / max(1, len(val_loader))
        val_intent_acc = correct_intent / max(1, total_samples)
        val_cat_acc = correct_cat / max(1, total_samples)
        val_traj_acc = correct_traj / max(1, total_samples)
        val_esc_mae = float(np.mean(esc_errors))
        val_eff_mae = float(np.mean(eff_errors))

        # Learned task standard deviations from Kendall loss
        sigmas = model.uncertainty_loss.get_task_sigmas()
        sigma_summary = " | ".join([f"{task_names[i]}: {sigmas[i]:.2f}" for i in range(len(sigmas))])

        print(f"\n--- Epoch {epoch+1} Results ---")
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"Val Intent Accuracy     : {val_intent_acc * 100:.2f}%")
        print(f"Val Category Accuracy   : {val_cat_acc * 100:.2f}%")
        print(f"Val Trajectory Accuracy : {val_traj_acc * 100:.2f}%")
        print(f"Val Escalation MAE      : {val_esc_mae:.4f}")
        print(f"Val Effort MAE          : {val_eff_mae:.4f}")
        print(f"Kendall Learned Uncertainties: {sigma_summary}")
        print("-" * 55)

        # Log to MLflow
        log_mlflow_metrics({
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_intent_accuracy": val_intent_acc,
            "val_category_accuracy": val_cat_acc,
            "val_trajectory_accuracy": val_traj_acc,
            "val_escalation_mae": val_esc_mae,
            "val_effort_mae": val_eff_mae
        }, step=epoch + 1)

        # Checkpointing & Early Stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            save_nlu_checkpoint(
                model=model,
                tokenizer=tokenizer,
                output_dir=args.output_dir,
                metadata={
                    "best_epoch": epoch + 1,
                    "val_loss": avg_val_loss,
                    "val_intent_accuracy": val_intent_acc,
                    "val_category_accuracy": val_cat_acc,
                    "val_trajectory_accuracy": val_traj_acc,
                    "val_escalation_mae": val_esc_mae
                }
            )
        else:
            patience_counter += 1
            print(f"⚠️ No improvement for {patience_counter}/{args.patience} epoch(s).")
            if patience_counter >= args.patience:
                print(f"🛑 [EARLY STOPPING] Triggered at Epoch {epoch+1}.")
                break

    print(f"\n✅ Fine-tuning complete. Best model saved in: {args.output_dir}")


if __name__ == "__main__":
    main()
