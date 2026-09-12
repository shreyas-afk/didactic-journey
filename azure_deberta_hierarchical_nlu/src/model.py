"""
DeBERTa-v3 Hierarchical Turn-Transformer NLU Architecture & Kendall Multi-Task Uncertainty Loss.
"""

import math
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig, AutoModelForMaskedLM

from .config import (
    CATEGORIES, INTENTS, FLAGS, TRAJECTORIES,
    DEFAULT_FLAG_POS_WEIGHTS, DEFAULT_DROPOUT,
    DEFAULT_PRETRAINED_MODEL
)


# ---------------------------------------------------------------------------
# 1. Masked Attention Pooling over DeBERTa Token States
# ---------------------------------------------------------------------------
class MaskedAttentionPooling(nn.Module):
    """
    Learned Attention Pooling layer over DeBERTa-v3 token representations.
    Replaces naive [CLS] token slicing to capture comprehensive sentence semantics
    while respecting token padding masks.
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            last_hidden_state: [N, Seq_Len, Hidden]
            attention_mask:    [N, Seq_Len]
        Returns:
            pooled:            [N, Hidden]
        """
        scores = self.scorer(last_hidden_state).squeeze(-1)  # [N, Seq_Len]
        scores = scores.masked_fill(attention_mask == 0, -1e4)
        attn_weights = torch.softmax(scores, dim=-1).unsqueeze(-1)  # [N, Seq_Len, 1]
        pooled = torch.sum(last_hidden_state * attn_weights, dim=1)  # [N, Hidden]
        return pooled


# ---------------------------------------------------------------------------
# 2. Homoscedastic Multi-Task Uncertainty Loss Module (Kendall et al.)
# ---------------------------------------------------------------------------
class KendallMultiTaskUncertaintyLoss(nn.Module):
    """
    Homoscedastic Multi-Task Uncertainty Loss Weighting (Kendall et al., CVPR 2018).
    Dynamically balances classification losses with continuous regression losses
    via learnable task log-variance parameters (log_vars).
    Total Loss = sum_i [ (1 / (2 * exp(log_var_i))) * L_i + 0.5 * log_var_i ]
    """
    def __init__(self, num_tasks: int = 6):
        super().__init__()
        # Initialize log_vars to 0.0 (initial sigma_i = 1.0)
        self.log_vars = nn.Parameter(torch.zeros(num_tasks, dtype=torch.float))

    def forward(self, loss_list: list) -> torch.Tensor:
        total_loss = 0.0
        for i, loss in enumerate(loss_list):
            precision = torch.exp(-self.log_vars[i])
            total_loss += 0.5 * precision * loss + 0.5 * self.log_vars[i]
        return total_loss

    def get_task_sigmas(self) -> list:
        """Returns current learned task standard deviations (sigma_i = exp(0.5 * log_var_i))."""
        with torch.no_grad():
            sigmas = torch.exp(0.5 * self.log_vars).cpu().tolist()
        return sigmas


# ---------------------------------------------------------------------------
# 3. Complete DeBERTa-v3 Hierarchical Perception Engine
# ---------------------------------------------------------------------------
class DeBERTaHierarchicalCustomerSupportTransformer(nn.Module):
    """
    Enterprise Dual-Level Perception Engine:
    Level 1: Sentence-Level Utterance Encoder (DeBERTa-v3 + Masked Attention Pooling)
    Level 2: Dialogue-Level Turn-Transformer (with Speaker & Temporal Turn Positional Embeddings)
    Level 3: Multi-Task Heads (Root-Grounded Intent & Category, Emotion Nuance, Trajectory Momentum, Escalation, Effort)
    Level 4: Homoscedastic Multi-Task Uncertainty Loss (Kendall et al.)
    """
    def __init__(
        self,
        model_name: str = DEFAULT_PRETRAINED_MODEL,
        num_categories: int = len(CATEGORIES),
        num_intents: int = len(INTENTS),
        num_flags: int = len(FLAGS),
        num_trajectories: int = len(TRAJECTORIES),
        pos_weights: torch.Tensor = DEFAULT_FLAG_POS_WEIGHTS,
        turn_layers: int = 2,
        max_turns: int = 16,
        dropout_rate: float = DEFAULT_DROPOUT
    ):
        super().__init__()
        self.model_name = model_name
        self.num_categories = num_categories
        self.num_intents = num_intents
        self.num_flags = num_flags
        self.num_trajectories = num_trajectories

        # LEVEL 1: Sentence-Level Utterance Encoder
        self.sentence_encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.sentence_encoder.config.hidden_size  # 768 for base
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout_rate)

        # Masked Attention Pooling for Utterance Tokens
        self.attention_pooler = MaskedAttentionPooling(hidden_size)

        # LEVEL 2: Dialogue-Level Turn-Transformer
        turn_encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=8,
            dim_feedforward=hidden_size * 2,
            dropout=dropout_rate,
            activation="gelu",
            batch_first=True
        )
        self.turn_transformer = nn.TransformerEncoder(turn_encoder_layer, num_layers=turn_layers)
        self.speaker_embedding = nn.Embedding(num_embeddings=2, embedding_dim=hidden_size)  # 0=Cust, 1=Agent
        self.turn_pos_embedding = nn.Embedding(num_embeddings=max_turns, embedding_dim=hidden_size)

        # LEVEL 3: PERCEPTION & TELEMETRY HEADS
        # 🎯 Root-Grounded Intent & Category: [d_root || d_latest] -> 1536-D
        self.category_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, num_categories)
        )
        self.intent_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, num_intents)
        )

        # 🌟 Instantaneous Emotion Nuance Flags: u_latest -> 768-D
        self.flags_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, num_flags)
        )

        # 📊 Global Momentum Trajectory & Effort: [d_global || d_latest] -> 1536-D
        self.trajectory_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, num_trajectories)
        )
        self.effort_head = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # 🚨 Real-time Escalation Regressor: d_latest -> 768-D
        self.escalation_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # Loss base functions
        self.ce_loss = nn.CrossEntropyLoss()
        self.register_buffer("flag_pos_weights", pos_weights)
        self.smooth_l1 = nn.SmoothL1Loss()

        # Kendall Homoscedastic Multi-Task Uncertainty Loss
        self.uncertainty_loss = KendallMultiTaskUncertaintyLoss(num_tasks=6)

    def forward(
        self,
        turn_input_ids: torch.Tensor,
        turn_attention_mask: torch.Tensor,
        speaker_ids: torch.Tensor,
        turn_positions: torch.Tensor = None,
        category_label: torch.Tensor = None,
        intent_label: torch.Tensor = None,
        flags_label: torch.Tensor = None,
        trajectory_label: torch.Tensor = None,
        escalation_label: torch.Tensor = None,
        effort_label: torch.Tensor = None
    ):
        batch_size, num_turns, max_len = turn_input_ids.shape

        flat_input_ids = turn_input_ids.view(batch_size * num_turns, max_len)
        flat_attention_mask = turn_attention_mask.view(batch_size * num_turns, max_len)

        # Level 1: Sentence Encoding with Attention Pooling
        encoder_out = self.sentence_encoder(input_ids=flat_input_ids, attention_mask=flat_attention_mask)
        flat_pooled = self.attention_pooler(encoder_out.last_hidden_state.float(), flat_attention_mask)

        turn_vectors = flat_pooled.view(batch_size, num_turns, -1)

        # Level 2: Speaker + Temporal Turn Positional Embeddings
        if turn_positions is None:
            turn_positions = torch.arange(num_turns, device=turn_vectors.device).unsqueeze(0).expand(batch_size, -1)

        turns_augmented = (
            turn_vectors +
            self.speaker_embedding(speaker_ids) +
            self.turn_pos_embedding(turn_positions)
        )

        dialogue_context = self.turn_transformer(turns_augmented)  # [Batch, Num_Turns, 768]

        # Representation Dissection
        d_root = dialogue_context[:, 0, :]       # Turn 1 Root Goal Vector [Batch, 768]
        d_latest = dialogue_context[:, -1, :]    # Latest Turn State Vector [Batch, 768]
        d_global = dialogue_context.mean(dim=1)  # All-Turns Global Momentum Summary [Batch, 768]
        u_latest = turn_vectors[:, -1, :]        # Instantaneous Utterance Vector [Batch, 768]

        # Root-Grounded Intent & Category Concatenation
        h_intent = self.dropout(torch.cat([d_root, d_latest], dim=-1))  # [Batch, 1536]
        # Global Momentum Trajectory Concatenation
        h_traj = self.dropout(torch.cat([d_global, d_latest], dim=-1))    # [Batch, 1536]

        cat_logits = self.category_head(h_intent)
        intent_logits = self.intent_head(h_intent)
        flags_logits = self.flags_head(self.dropout(u_latest))

        traj_logits = self.trajectory_head(h_traj)
        effort_pred = self.effort_head(h_traj).squeeze(-1)
        escalation_pred = self.escalation_head(self.dropout(d_latest)).squeeze(-1)

        total_loss = None
        losses = {}

        if category_label is not None:
            l_intent = self.ce_loss(intent_logits, intent_label)
            l_cat = self.ce_loss(cat_logits, category_label)
            flags_bce = nn.BCEWithLogitsLoss(pos_weight=self.flag_pos_weights.to(flags_logits.device))
            l_flags = flags_bce(flags_logits, flags_label)
            l_traj = self.ce_loss(traj_logits, trajectory_label)
            l_esc = self.smooth_l1(escalation_pred, escalation_label)
            l_eff = self.smooth_l1(effort_pred, effort_label)

            loss_list = [l_intent, l_cat, l_flags, l_traj, l_esc, l_eff]
            total_loss = self.uncertainty_loss(loss_list)

            losses = {
                "total": total_loss.item(),
                "intent": l_intent.item(),
                "category": l_cat.item(),
                "flags": l_flags.item(),
                "trajectory": l_traj.item(),
                "escalation": l_esc.item(),
                "effort": l_eff.item()
            }

        return {
            "loss": total_loss,
            "losses": losses,
            "cat_logits": cat_logits,
            "intent_logits": intent_logits,
            "flags_logits": flags_logits,
            "traj_logits": traj_logits,
            "escalation_pred": escalation_pred,
            "effort_pred": effort_pred
        }


# ---------------------------------------------------------------------------
# 4. Masked Language Model Wrapper
# ---------------------------------------------------------------------------
class DeBERTaMLMWrapper(nn.Module):
    """
    Stage 1 Masked Language Model backbone using Hugging Face AutoModelForMaskedLM.
    """
    def __init__(self, model_name: str = DEFAULT_PRETRAINED_MODEL):
        super().__init__()
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor = None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def save_pretrained(self, save_directory: str):
        self.model.save_pretrained(save_directory)
