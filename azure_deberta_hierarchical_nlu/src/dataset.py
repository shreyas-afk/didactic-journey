"""
Dataset loaders and pre-processing for MLM Domain Adaptation and Hierarchical Multi-Turn NLU.
"""

import re
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, TensorDataset

from .config import (
    CATEGORIES, INTENTS, FLAGS, TRAJECTORIES,
    cat2id, intent2id, traj2id, flag2id,
    DEFAULT_MAX_TURNS, DEFAULT_MAX_TURN_LEN
)

# ---------------------------------------------------------------------------
# Synthetic Entity Generator for Customer Support Slots
# ---------------------------------------------------------------------------
SYNTHETIC_ENTITIES = {
    "Order Number": lambda: f"ORD-{random.randint(10000, 99999)}",
    "Refund Amount": lambda: f"${random.randint(10, 500)}.{random.randint(10, 99):02d}",
    "Invoice Number": lambda: f"INV-{random.randint(10000, 99999)}",
    "Delivery City": lambda: random.choice(["New York", "Chicago", "London", "Berlin", "Austin", "Seattle"]),
    "Delivery Country": lambda: random.choice(["USA", "Canada", "Germany", "UK", "Australia"]),
    "Person Name": lambda: random.choice(["John Doe", "Sarah Connor", "Alex Smith", "Emma Watson"]),
    "Account Category": lambda: random.choice(["Personal", "Business", "VIP", "Enterprise"]),
    "Account Type": lambda: random.choice(["Standard", "Premium", "Pro"]),
    "Currency Symbol": lambda: "$"
}

AGENT_CANNED_REPLIES = [
    "Could you please confirm your registered account email and order ID?",
    "Please provide your details so I can check your account records.",
    "I understand. Could you clarify the issue in more detail so I can assist?",
    "Let me pull up your account records to check this for you."
]


def clean_tweet_text(text: str) -> str:
    """Normalizes tweet text: standardizes mentions, URLs, and excessive whitespace."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"@[A-Za-z0-9_]+", "@User", text)
    text = re.sub(r"https?://\S+", "http://url", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def inject_synthetic_entities(text: str):
    """Replaces {{Entity_Name}} template tags with randomized realistic entities."""
    pattern = re.compile(r"\{\{([^}]+)\}\}")
    spans = []
    new_text = ""
    last_end = 0
    for match in pattern.finditer(text):
        ent_name = match.group(1).strip()
        generator = SYNTHETIC_ENTITIES.get(ent_name, lambda: ent_name)
        val = generator()
        new_text += text[last_end:match.start()]
        start_char = len(new_text)
        new_text += val
        end_char = len(new_text)
        spans.append((start_char, end_char, ent_name.replace(" ", "_")))
        last_end = match.end()
    new_text += text[last_end:]
    return new_text, spans


def build_multi_turn_dialogue(row):
    """
    Transforms a single customer ticket row into a rich 3-turn chronological conversation
    with calibrated escalation, customer effort score (CES), and trajectory momentum label.
    """
    # 1. Pre-labeled Twitter multi-turn dialogue
    if pd.notna(row.get("trajectory")) and pd.notna(row.get("escalation")):
        traj_name = str(row["trajectory"])
        traj_label = traj2id.get(traj_name, traj2id["STABLE_INQUIRY"])
        esc_score = float(row["escalation"])
        ces_score = float(row.get("customer_effort", 0.5))
        text_t1 = str(row["instruction"])
        turns = [
            {"speaker": 0, "text": text_t1},
            {"speaker": 1, "text": "Thanks for reaching out. Please provide your account details so we can check."},
            {"speaker": 0, "text": "Thank you, I sent the details." if esc_score < 0.40 else "I have been waiting forever, please fix this now!"}
        ]
        return turns, [], traj_label, esc_score, ces_score

    # 2. Bitext row with calibrated multi-turn synthesis
    text_t1, spans_t1 = inject_synthetic_entities(str(row["instruction"]))
    flags = str(row.get("flags", ""))
    intent = str(row.get("intent", "general_inquiry"))
    intent_clean = intent.replace("_", " ")

    agent_reply = random.choice(AGENT_CANNED_REPLIES)
    rand_val = random.random()

    if "W" in flags:  # Severe Anger / Profanity
        text_t3 = random.choice([
            f"I already provided that! Why is this taking so damn long, get me a supervisor for my {intent_clean}!",
            f"This is ridiculous, stop sending canned messages and fix my {intent_clean} right now!",
            f"Unacceptable service! Stop asking stupid questions and resolve my {intent_clean} immediately!"
        ])
        traj_label = traj2id["CRITICAL_CHURN_RISK"]
        esc_score = 0.88
        ces_score = 0.85
    elif rand_val < 0.15:  # Chronic Service Incompetence / Repetition
        text_t3 = random.choice([
            f"I have tried for 60 days to resolve this. No service, several transfers, nobody can find my {intent_clean}.",
            f"This cuts out constantly. For the third time this year I am facing this issue with {intent_clean}.",
            f"Zero stars. Challenging, frustrating, and your update broke my {intent_clean}.",
            f"I sent several messages and no one responds. Next time I am switching to your competitor."
        ])
        traj_label = traj2id["ESCALATING_FRICTION"]
        esc_score = 0.72
        ces_score = 0.78
    elif "M" in flags or "E" in flags:  # Distress / Impatience
        text_t3 = random.choice([
            f"I have been waiting for hours, can you please assist with my {intent_clean}?",
            f"I really need this resolved today, please check on my {intent_clean} again.",
            f"Why is this taking so long? I already provided all details for my {intent_clean}."
        ])
        traj_label = traj2id["ESCALATING_FRICTION"]
        esc_score = 0.48
        ces_score = 0.52
    elif rand_val < 0.40:  # Casual Clarification
        text_t3 = random.choice([
            f"Hi, I realized I made a mistake on my end regarding {intent_clean}, can you help me update it?",
            f"Could you clarify how to select the open option for my {intent_clean}?",
            f"Thanks for the info! Just wanted to confirm where to find {intent_clean}."
        ])
        traj_label = traj2id["STABLE_INQUIRY"]
        esc_score = 0.04
        ces_score = 0.12
    else:  # Stable Routine Inquiry
        text_t3 = random.choice([
            f"Sure, my email is user@example.com regarding my {intent_clean}.",
            f"Thanks, I have provided the details for my {intent_clean} as requested.",
            "Okay, let me know what the diagnostic check shows."
        ])
        traj_label = traj2id["STABLE_INQUIRY"]
        esc_score = 0.03
        ces_score = 0.10

    turns = [
        {"speaker": 0, "text": text_t1},
        {"speaker": 1, "text": agent_reply},
        {"speaker": 0, "text": text_t3}
    ]
    return turns, spans_t1, traj_label, esc_score, ces_score


# ---------------------------------------------------------------------------
# Hierarchical Multi-Turn NLU Dataset
# ---------------------------------------------------------------------------
class HierarchicalDialogueDataset(Dataset):
    """
    PyTorch Dataset for Hierarchical Turn-Transformer NLU.
    Yields batched multi-turn dialogue tensor structures.
    """
    def __init__(self, df: pd.DataFrame, tokenizer, max_turns=DEFAULT_MAX_TURNS, max_turn_len=DEFAULT_MAX_TURN_LEN):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_turns = max_turns
        self.max_turn_len = max_turn_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        turns, _, traj_label, esc_score, ces_score = build_multi_turn_dialogue(row)

        turn_input_ids = []
        turn_attention_masks = []
        speaker_ids = []

        for t in turns[:self.max_turns]:
            enc = self.tokenizer(
                t["text"],
                padding="max_length",
                truncation=True,
                max_length=self.max_turn_len,
                return_tensors="pt"
            )
            turn_input_ids.append(enc["input_ids"].squeeze(0))
            turn_attention_masks.append(enc["attention_mask"].squeeze(0))
            speaker_ids.append(t["speaker"])

        num_actual_turns = len(turn_input_ids)
        turn_positions = list(range(num_actual_turns))
        flags_vec = torch.tensor([float(row.get(f"flag_{f}", 0.0)) for f in FLAGS], dtype=torch.float)

        category_val = row.get("category", "TECH_SUPPORT")
        intent_val = row.get("intent", "general_inquiry")

        cat_id = cat2id.get(category_val, 0)
        intent_id = intent2id.get(intent_val, 0)

        return {
            "turn_input_ids": torch.stack(turn_input_ids),
            "turn_attention_mask": torch.stack(turn_attention_masks),
            "speaker_ids": torch.tensor(speaker_ids, dtype=torch.long),
            "turn_positions": torch.tensor(turn_positions, dtype=torch.long),
            "category_label": torch.tensor(cat_id, dtype=torch.long),
            "intent_label": torch.tensor(intent_id, dtype=torch.long),
            "flags_label": flags_vec,
            "trajectory_label": torch.tensor(traj_label, dtype=torch.long),
            "escalation_label": torch.tensor(esc_score, dtype=torch.float),
            "effort_label": torch.tensor(ces_score, dtype=torch.float)
        }


# ---------------------------------------------------------------------------
# Masked Language Modeling (MLM) Dataset
# ---------------------------------------------------------------------------
class CustomerSupportMLMDataset(Dataset):
    """
    High-throughput PyTorch Dataset for 15% dynamic Masked Language Modeling (MLM).
    80% [MASK], 10% random token, 10% unchanged.
    """
    def __init__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, tokenizer, mlm_probability=0.15):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability

    def __len__(self):
        return self.input_ids.size(0)

    def __getitem__(self, idx):
        input_ids = self.input_ids[idx].clone()
        attention_mask = self.attention_mask[idx]
        labels = input_ids.clone()

        # Probability matrix for masking
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        # Do not mask special tokens (e.g. [CLS], [SEP], [PAD])
        special_tokens_mask = self.tokenizer.get_special_tokens_mask(
            labels.tolist(), already_has_special_tokens=True
        )
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        probability_matrix.masked_fill_(attention_mask == 0, value=0.0)

        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # Only compute loss on masked tokens

        # 80% of the time, replace with [MASK]
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.mask_token_id

        # 10% of the time, replace with random token
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        input_ids[indices_random] = random_words[indices_random]

        # 10% of the time, keep original token (already in input_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


def load_and_prepare_nlu_dataframe(bitext_path: str, twitter_path: str = None, sample_size: int = None) -> pd.DataFrame:
    """Loads, cleans, and blends Bitext and Twitter datasets into a unified DataFrame."""
    print(f"Loading Bitext customer support data from: {bitext_path}")
    df_bitext = pd.read_csv(bitext_path)

    if twitter_path and os.path.exists(twitter_path):
        print(f"Loading Twitter multi-turn data from: {twitter_path}")
        df_twitter = pd.read_csv(twitter_path)
        if sample_size and len(df_twitter) > sample_size:
            df_twitter = df_twitter.groupby("category", group_keys=False).apply(
                lambda x: x.sample(min(len(x), int(sample_size * len(x) / len(df_twitter))), random_state=42)
            ).reset_index(drop=True)
            print(f"Sampled {len(df_twitter):,} Twitter rows.")
        df = pd.concat([df_bitext, df_twitter], ignore_index=True)
    else:
        df = df_bitext

    # Compute flag columns if not present
    for f in FLAGS:
        col = f"flag_{f}"
        if col not in df.columns:
            df[col] = df["flags"].apply(lambda x: 1.0 if f in str(x) else 0.0)

    # Filter invalid categories / intents if necessary
    df = df[df["category"].isin(CATEGORIES) & df["intent"].isin(INTENTS)].reset_index(drop=True)
    print(f"Unified Dataset ready with {len(df):,} records across {len(CATEGORIES)} categories and {len(INTENTS)} intents.")
    return df
