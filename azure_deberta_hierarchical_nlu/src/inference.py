"""
Real-time Structured Telemetry Engine for Downstream LLM Agents.
Loads fine-tuned DeBERTa Hierarchical model and outputs rich JSON telemetry payloads.
"""

import os
import re
import json
import torch
from transformers import AutoTokenizer

from .config import (
    CATEGORIES, INTENTS, FLAGS, TRAJECTORIES,
    cat2id, intent2id, traj2id, flag2id,
    id2cat, id2intent, id2traj, id2flag,
    DEFAULT_MAX_TURN_LEN
)
from .model import DeBERTaHierarchicalCustomerSupportTransformer
from .utils import get_device


def extract_slots_from_text(text: str) -> dict:
    """Extracts common customer support entity slots (Order IDs, Invoice IDs, Amounts, etc.)."""
    slots = {}
    order_match = re.search(r"\b(ORD-\d+|#\d{5,8}|PO-\d+|CAS-\d+)\b", text, re.IGNORECASE)
    if order_match:
        slots["Order_Number"] = order_match.group(1)

    amount_match = re.search(r"(\$|€|£)\s*\d+(\.\d{2})?", text)
    if amount_match:
        slots["Refund_Amount"] = amount_match.group(0)

    inv_match = re.search(r"\b(INV-\d+)\b", text, re.IGNORECASE)
    if inv_match:
        slots["Invoice_Number"] = inv_match.group(1)

    return slots


class DeBERTaHierarchicalNLUPredictor:
    """
    Production-ready inference wrapper for DeBERTa-v3 Hierarchical NLU Perception Engine.
    """
    def __init__(self, model_dir: str, device: torch.device = None):
        self.model_dir = model_dir
        self.device = device or get_device()

        # Load Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

        # Load Model
        checkpoint_path = os.path.join(model_dir, "best_deberta_hierarchical_nlu.pt")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Model checkpoint not found at: {checkpoint_path}")

        print(f"Loading trained weights from: {checkpoint_path}")
        self.model = DeBERTaHierarchicalCustomerSupportTransformer(
            model_name=model_dir,
            num_categories=len(CATEGORIES),
            num_intents=len(INTENTS),
            num_flags=len(FLAGS),
            num_trajectories=len(TRAJECTORIES)
        ).to(self.device)

        state_dict = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        print("✅ DeBERTa Hierarchical Predictor initialized successfully.")

    def predict_dialogue_trajectory(self, conversation_turns: list, max_turn_len: int = DEFAULT_MAX_TURN_LEN) -> dict:
        """
        Generates enterprise telemetry JSON for a multi-turn conversation.

        Args:
            conversation_turns: List of dicts, e.g. [{"speaker": "Customer", "text": "Where is my order?"}]
            max_turn_len: Token cutoff per turn.

        Returns:
            dict: Structured telemetry dictionary for LLM orchestrator.
        """
        self.model.eval()
        turn_input_ids = []
        turn_attention_masks = []
        speaker_ids = []

        # Up to 5 most recent turns for the dialogue context window
        recent_turns = conversation_turns[-5:]
        for t in recent_turns:
            spk_str = str(t.get("speaker", "Customer")).lower()
            spk_id = 0 if ("customer" in spk_str or "user" in spk_str) else 1

            enc = self.tokenizer(
                str(t.get("text", "")),
                padding="max_length",
                truncation=True,
                max_length=max_turn_len,
                return_tensors="pt"
            )
            turn_input_ids.append(enc["input_ids"].squeeze(0))
            turn_attention_masks.append(enc["attention_mask"].squeeze(0))
            speaker_ids.append(spk_id)

        num_turns = len(turn_input_ids)
        turn_positions = torch.arange(num_turns, dtype=torch.long).unsqueeze(0).to(self.device)
        input_ids_tensor = torch.stack(turn_input_ids).unsqueeze(0).to(self.device)
        attention_mask_tensor = torch.stack(turn_attention_masks).unsqueeze(0).to(self.device)
        speaker_ids_tensor = torch.tensor(speaker_ids, dtype=torch.long).unsqueeze(0).to(self.device)

        use_amp = (self.device.type == "cuda")
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
                out = self.model(
                    turn_input_ids=input_ids_tensor,
                    turn_attention_mask=attention_mask_tensor,
                    speaker_ids=speaker_ids_tensor,
                    turn_positions=turn_positions
                )

        cat_probs = torch.softmax(out["cat_logits"], dim=-1).squeeze(0)
        intent_probs = torch.softmax(out["intent_logits"], dim=-1).squeeze(0)
        traj_probs = torch.softmax(out["traj_logits"], dim=-1).squeeze(0)

        top_intent_prob, intent_idx = torch.topk(intent_probs, 1)
        cat_idx = out["cat_logits"].argmax(dim=-1).item()
        traj_idx = out["traj_logits"].argmax(dim=-1).item()

        intent_confidence = float(top_intent_prob.item())
        is_uncertain = intent_confidence < 0.50
        suggested_action = "ASK_CLARIFYING_QUESTION" if is_uncertain else "EXECUTE_INTENT_WORKFLOW"

        esc_score = float(out["escalation_pred"].item())
        effort_score = float(out["effort_pred"].item())

        flag_probs = torch.sigmoid(out["flags_logits"]).squeeze(0).cpu().numpy()
        traj_distribution = {
            id2traj[i]: f"{round(float(p) * 100, 1)}%" for i, p in enumerate(traj_probs.cpu().numpy())
        }

        all_text = " ".join([str(t.get("text", "")) for t in conversation_turns])
        extracted_entities = extract_slots_from_text(all_text)

        return {
            "intent": id2intent.get(intent_idx.item(), "general_inquiry"),
            "category": id2cat.get(cat_idx, "TECH_SUPPORT"),
            "intent_confidence": round(intent_confidence, 3),
            "is_uncertain": is_uncertain,
            "suggested_action": suggested_action,
            "entities": extracted_entities,
            "trajectory_state": id2traj.get(traj_idx, "STABLE_INQUIRY"),
            "dynamic_escalation": round(esc_score, 3),
            "customer_effort_score": round(effort_score, 3),
            "current_emotion_profile": {
                "Anger": f"{round(float(flag_probs[flag2id['W']]) * 100, 1)}%",
                "Frustration": f"{round(float(flag_probs[flag2id['M']]) * 100, 1)}%",
                "Distress": f"{round(float(flag_probs[flag2id['E']]) * 100, 1)}%",
                "Politeness": f"{round(float(flag_probs[flag2id['P']]) * 100, 1)}%"
            },
            "trajectory_distribution": traj_distribution
        }
