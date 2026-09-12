"""
Central Configuration & Constants for DeBERTa-v3 Hierarchical NLU Perception Engine.
"""

import os
import torch

# Base Pretrained Model
DEFAULT_PRETRAINED_MODEL = "microsoft/deberta-v3-base"

# Level 1: Categories (12 Macro-Categories)
CATEGORIES = [
    "ACCOUNT",
    "APP_SUPPORT",
    "BILLING",
    "CANCELLATION",
    "DELIVERY",
    "FEEDBACK",
    "ORDER",
    "PAYMENT",
    "PRODUCT_INQUIRY",
    "REFUND",
    "SUBSCRIPTION",
    "TECH_SUPPORT"
]

# Level 1: 38 Expanded Enterprise Intents
INTENTS = [
    "cancel_order",
    "change_address",
    "change_payment_method",
    "change_shipping",
    "check_cancellation_fee",
    "check_invoice",
    "check_order_status",
    "check_payment_status",
    "check_refund_policy",
    "check_refund_status",
    "close_account",
    "contact_agent",
    "create_account",
    "delete_data",
    "dispute_charge",
    "edit_account",
    "escalate_issue",
    "get_app_help",
    "get_discount",
    "get_invoice",
    "get_refund",
    "leave_feedback",
    "log_in",
    "log_out",
    "modify_order",
    "network_outage",
    "pause_subscription",
    "product_availability",
    "product_details",
    "recover_password",
    "renew_subscription",
    "report_bug",
    "report_damaged_item",
    "report_fraud",
    "report_late_delivery",
    "restart_device",
    "track_package",
    "upgrade_subscription"
]

# Level 1: 14 Emotion & Nuance Linguistic Flags
FLAGS = ["B", "L", "Q", "I", "Z", "M", "C", "K", "E", "P", "W", "N", "S", "V"]

FLAG_DESCRIPTIONS = {
    "B": "Basic / Direct Request",
    "L": "Location Reference",
    "Q": "Question / Inquiry",
    "I": "Informational Statement",
    "Z": "Short / Terse Utterance",
    "M": "Frustration / Impatience",
    "C": "Clarification / Correction",
    "K": "Keyword Heavy",
    "E": "Distress / Urgent Emotion",
    "P": "Politeness / Gratitude",
    "W": "Severe Anger / Hostility / Profanity",
    "N": "Negative Sentiment",
    "S": "Sarcasm / Irony",
    "V": "Verification Request"
}

# Level 2: 4 Dialogue Trajectory Momentum Classes
TRAJECTORIES = [
    "STABLE_INQUIRY",          # Routine troubleshooting / standard back-and-forth / casual inquiry
    "ESCALATING_FRICTION",      # Customer getting frustrated / repeating details / chronic failure
    "CRITICAL_CHURN_RISK",     # Profanity / severe hostility / churn threat / cancellation
    "DE_ESCALATING_RESOLVED"   # Issue being fixed / customer satisfied / gratitude
]

# Index Mappings
cat2id = {c: i for i, c in enumerate(CATEGORIES)}
id2cat = {i: c for c, i in cat2id.items()}

intent2id = {intent: i for i, intent in enumerate(INTENTS)}
id2intent = {i: intent for intent, i in intent2id.items()}

traj2id = {t: i for i, t in enumerate(TRAJECTORIES)}
id2traj = {i: t for t, i in traj2id.items()}

flag2id = {f: i for i, f in enumerate(FLAGS)}
id2flag = {i: f for f, i in flag2id.items()}

# Default Class Weightings for Imbalanced Multi-Label Flags (14 flags)
DEFAULT_FLAG_POS_WEIGHTS = torch.tensor([
    1.2,  # B: Basic
    3.5,  # L: Location
    1.5,  # Q: Question
    1.8,  # I: Info
    2.5,  # Z: Terse
    4.0,  # M: Frustration
    3.0,  # C: Clarification
    2.8,  # K: Keyword
    4.5,  # E: Distress
    2.0,  # P: Politeness
    6.5,  # W: Severe Anger
    3.2,  # N: Negative
    8.0,  # S: Sarcasm
    3.0   # V: Verification
], dtype=torch.float)

# Training Defaults
DEFAULT_MAX_TURNS = 5
DEFAULT_MAX_TURN_LEN = 48
DEFAULT_MLM_MAX_LEN = 48
DEFAULT_BATCH_SIZE = 32
DEFAULT_MLM_BATCH_SIZE = 128
DEFAULT_EPOCHS = 5
DEFAULT_MLM_EPOCHS = 3
DEFAULT_BASE_LR = 2e-5
DEFAULT_HEAD_LR = 6e-5
DEFAULT_MLM_LR = 5e-5
DEFAULT_LLRD_DECAY = 0.9
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_DROPOUT = 0.2
