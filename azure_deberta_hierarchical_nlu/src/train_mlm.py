"""
Stage 1: Masked Language Model (MLM) Domain Adaptation for Customer Support DeBERTa Backbone.
"""

import os
import re
import math
import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForMaskedLM, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from .config import DEFAULT_PRETRAINED_MODEL, DEFAULT_MLM_MAX_LEN, DEFAULT_MLM_BATCH_SIZE, DEFAULT_MLM_EPOCHS, DEFAULT_MLM_LR
from .dataset import CustomerSupportMLMDataset, clean_tweet_text
from .utils import get_device, set_seed, init_mlflow_run, log_mlflow_metrics, log_mlflow_params


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 1: Customer Support MLM Domain Adaptation")
    parser.add_argument("--train_data", type=str, required=True, help="Path to raw or augmented Twitter customer support CSV")
    parser.add_argument("--model_name", type=str, default=DEFAULT_PRETRAINED_MODEL, help="Pretrained model identifier or path")
    parser.add_argument("--output_dir", type=str, default="./outputs/deberta_twitter_adapted", help="Output directory for adapted weights")
    parser.add_argument("--epochs", type=int, default=DEFAULT_MLM_EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_MLM_BATCH_SIZE, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=DEFAULT_MLM_LR, help="AdamW learning rate")
    parser.add_argument("--max_len", type=int, default=DEFAULT_MLM_MAX_LEN, help="Maximum token length per tweet")
    parser.add_argument("--sample_size", type=int, default=None, help="Optional maximum row sample size for rapid training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    init_mlflow_run(experiment_name="DeBERTa-CustomerSupport-MLM", run_name="Stage1-MLM-Domain-Adaptation")
    log_mlflow_params(vars(args))

    print("=" * 70)
    print("STAGE 1: MASKED LANGUAGE MODEL DOMAIN ADAPTATION")
    print(f"Data Path     : {args.train_data}")
    print(f"Base Model    : {args.model_name}")
    print(f"Output Path   : {args.output_dir}")
    print(f"Epochs        : {args.epochs} | Batch Size: {args.batch_size} | LR: {args.learning_rate}")
    print("=" * 70)

    # 1. Load and clean corpus
    if not os.path.exists(args.train_data):
        raise FileNotFoundError(f"Input training dataset not found at: {args.train_data}")

    print("Loading raw customer support text corpus...")
    # Determine columns
    df_preview = pd.read_csv(args.train_data, nrows=5)
    text_col = "instruction" if "instruction" in df_preview.columns else ("text" if "text" in df_preview.columns else df_preview.columns[0])
    
    df = pd.read_csv(args.train_data, usecols=[text_col], dtype={text_col: str})
    print(f"Loaded {len(df):,} raw utterances from column '{text_col}'.")

    if args.sample_size and len(df) > args.sample_size:
        df = df.sample(args.sample_size, random_state=args.seed).reset_index(drop=True)
        print(f"Subsampled to {len(df):,} records for training.")

    print("Normalizing and cleaning text...")
    df["clean"] = df[text_col].apply(clean_tweet_text)
    df_clean = df[df["clean"].str.len() > 12].drop_duplicates(subset=["clean"])
    all_texts = df_clean["clean"].tolist()

    train_texts, val_texts = train_test_split(all_texts, test_size=0.05, random_state=args.seed)
    print(f"Train Corpus: {len(train_texts):,} | Val Corpus: {len(val_texts):,}")

    # 2. Tokenize in batches
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    print("Pre-tokenizing corpus using fast tokenizer...")

    def batch_tokenize(texts, chunk_size=100000):
        all_ids, all_masks = [], []
        for i in range(0, len(texts), chunk_size):
            chunk = texts[i:i + chunk_size]
            enc = tokenizer(chunk, padding="max_length", truncation=True, max_length=args.max_len, return_tensors="pt")
            all_ids.append(enc["input_ids"].to(torch.int32))
            all_masks.append(enc["attention_mask"].to(torch.int8))
        return torch.cat(all_ids, dim=0), torch.cat(all_masks, dim=0)

    train_ids, train_masks = batch_tokenize(train_texts)
    val_ids, val_masks = batch_tokenize(val_texts)

    train_dataset = CustomerSupportMLMDataset(train_ids, train_masks, tokenizer, mlm_probability=0.15)
    val_dataset = CustomerSupportMLMDataset(val_ids, val_masks, tokenizer, mlm_probability=0.15)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size * 2, shuffle=False, pin_memory=(device.type == "cuda"))

    # 3. Model & Optimizer Setup
    print(f"Instantiating Masked Language Model backbone from {args.model_name}...")
    model = AutoModelForMaskedLM.from_pretrained(args.model_name).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.05),
        num_training_steps=total_steps
    )

    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val_loss = float("inf")
    os.makedirs(args.output_dir, exist_ok=True)

    # 4. Training Loop
    for epoch in range(args.epochs):
        model.train()
        total_train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"MLM Epoch {epoch+1}/{args.epochs} [Train]")

        for batch in pbar:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device, dtype=torch.long, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, dtype=torch.long, non_blocking=True)
            labels = batch["labels"].to(device, dtype=torch.long, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

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

        # Validation
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"MLM Epoch {epoch+1}/{args.epochs} [Val]"):
                input_ids = batch["input_ids"].to(device, dtype=torch.long, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, dtype=torch.long, non_blocking=True)
                labels = batch["labels"].to(device, dtype=torch.long, non_blocking=True)

                with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    total_val_loss += outputs.loss.item()

        avg_val_loss = total_val_loss / max(1, len(val_loader))
        perplexity = math.exp(min(20.0, avg_val_loss))

        print(f"\n--- Epoch {epoch+1} Results ---")
        print(f"Train Loss : {avg_train_loss:.4f}")
        print(f"Val Loss   : {avg_val_loss:.4f} | Perplexity: {perplexity:.2f}")

        log_mlflow_metrics({
            "mlm_train_loss": avg_train_loss,
            "mlm_val_loss": avg_val_loss,
            "mlm_perplexity": perplexity
        }, step=epoch + 1)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"🌟 [CHECKPOINT] Saving domain-adapted DeBERTa backbone to: {args.output_dir}")
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)

    print(f"\n✅ Stage 1 Domain Adaptation Complete. Adapted model ready at: {args.output_dir}")


if __name__ == "__main__":
    main()
