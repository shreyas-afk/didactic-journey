"""
User-Friendly Inference CLI & Interactive Telemetry Runner for Trained DeBERTa-v3 Hierarchical NLU.

Supported Execution Modes:
1. Interactive Chat Simulation:
   python run_inference.py --interactive

2. Single Multi-Turn JSON Input:
   python run_inference.py --turns '[{"speaker":"Customer","text":"My order ORD-99212 is late!"},{"speaker":"Agent","text":"Let me check."},{"speaker":"Customer","text":"Hurry up damn it!"}]'

3. Batch File Inference:
   python run_inference.py --input_file data/sample_twitter.csv --output_file outputs/batch_telemetry.json
"""

import os
import sys
import json
import argparse
import torch
import pandas as pd
from transformers import AutoTokenizer

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from src.config import (
    CATEGORIES, INTENTS, FLAGS, TRAJECTORIES,
    id2cat, id2intent, id2traj, flag2id,
    DEFAULT_MAX_TURN_LEN, DEFAULT_PRETRAINED_MODEL
)
from src.model import DeBERTaHierarchicalCustomerSupportTransformer
from src.inference import DeBERTaHierarchicalNLUPredictor, extract_slots_from_text
from src.utils import get_device


def find_default_checkpoint():
    """Searches for available local trained checkpoints."""
    search_paths = [
        "./outputs/deberta_hierarchical_nlu/best_deberta_hierarchical_nlu.pt",
        "./outputs/best_deberta_hierarchical_nlu.pt",
        "../best_deberta_hierarchical_nlu.pt",
        "../best_deberta_hierarchical_nlu1.pt",
        "./best_deberta_hierarchical_nlu.pt"
    ]
    for p in search_paths:
        if os.path.exists(p):
            return os.path.abspath(p)
    return None


def parse_args():
    parser = argparse.ArgumentParser(description="DeBERTa Hierarchical NLU Inference Engine")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to trained PyTorch weights (.pt). If omitted, automatically finds local best checkpoint."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=DEFAULT_PRETRAINED_MODEL,
        help="Base model identifier or tokenizer directory"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Launch interactive conversation simulator in terminal"
    )
    parser.add_argument(
        "--turns",
        type=str,
        default=None,
        help='JSON string representing turns: [{"speaker":"Customer","text":"..."},{"speaker":"Agent","text":"..."}]'
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default=None,
        help="Path to input CSV or JSON file containing conversations for batch inference"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Path to save batch inference results"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device ('cuda' or 'cpu')"
    )
    return parser.parse_args()


class StandaloneNLUInferenceEngine:
    """
    Flexible inference engine capable of loading standalone .pt checkpoint files
    directly alongside the base tokenizer.
    """
    def __init__(self, checkpoint_path: str = None, model_name: str = DEFAULT_PRETRAINED_MODEL, device: str = None):
        if device:
            self.device = torch.device(device)
        else:
            self.device = get_device()

        self.model_name = model_name
        self.checkpoint_path = checkpoint_path or find_default_checkpoint()

        print(f"Loading Tokenizer from: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        print(f"Initializing DeBERTa Hierarchical Model Architecture...")
        self.model = DeBERTaHierarchicalCustomerSupportTransformer(
            model_name=model_name,
            num_categories=len(CATEGORIES),
            num_intents=len(INTENTS),
            num_flags=len(FLAGS),
            num_trajectories=len(TRAJECTORIES)
        ).to(self.device)

        if self.checkpoint_path and os.path.exists(self.checkpoint_path):
            print(f"Loading trained weights from: {self.checkpoint_path}")
            state_dict = torch.load(self.checkpoint_path, map_location=self.device)
            self.model.load_state_dict(state_dict, strict=False)
            print("✅ Trained weights loaded successfully!")
        else:
            print("⚠️ No checkpoint found. Running with base pretrained weights (untuned heads).")

        self.model.eval()

    def predict(self, conversation_turns: list) -> dict:
        """
        Runs multi-task inference and returns structured telemetry.
        """
        turn_input_ids = []
        turn_attention_masks = []
        speaker_ids = []

        recent_turns = conversation_turns[-5:]
        for t in recent_turns:
            spk_str = str(t.get("speaker", "Customer")).lower()
            spk_id = 0 if ("customer" in spk_str or "user" in spk_str) else 1

            enc = self.tokenizer(
                str(t.get("text", "")),
                padding="max_length",
                truncation=True,
                max_length=DEFAULT_MAX_TURN_LEN,
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


def print_telemetry_card(telemetry: dict):
    """Renders a readable terminal card with telemetry indicators."""
    esc = telemetry["dynamic_escalation"]
    if esc > 0.60:
        alert = "🚨 [CRITICAL CHURN RISK / ESCALATION]"
    elif esc > 0.35:
        alert = "⚠️ [MODERATE FRICTION]"
    else:
        alert = "✅ [STABLE / HEALTHY]"

    print("\n" + "=" * 75)
    print(f"📊 REAL-TIME CONVERSATIONAL NLU TELEMETRY  {alert}")
    print("=" * 75)
    print(f"  🎯 Intent               : {telemetry['intent']} (Category: {telemetry['category']})")
    print(f"  📈 Confidence           : {telemetry['intent_confidence'] * 100:.1f}% (Uncertain: {telemetry['is_uncertain']})")
    print(f"  🤖 Suggested Action     : {telemetry['suggested_action']}")
    print(f"  🏷️  Extracted Entities   : {telemetry['entities'] if telemetry['entities'] else 'None'}")
    print(f"  🧭 Trajectory State     : {telemetry['trajectory_state']}")
    print(f"  🔥 Escalation Index     : {telemetry['dynamic_escalation']} / 1.00")
    print(f"  ⚡ Customer Effort (CES): {telemetry['customer_effort_score']} / 1.00")
    print(f"  🎭 Emotion Profile      : {telemetry['current_emotion_profile']}")
    print(f"  🌐 Momentum Distr.      : {telemetry['trajectory_distribution']}")
    print("=" * 75 + "\n")


def run_interactive_mode(engine: StandaloneNLUInferenceEngine):
    """Runs a multi-turn interactive session in the console."""
    print("\n" + "=" * 75)
    print("💬 INTERACTIVE MULTI-TURN TELEMETRY SIMULATOR")
    print("Type your message as Customer or Agent. (Type 'exit' to quit, 'reset' to start new thread)")
    print("Prefix with 'agent:' to simulate an Agent response, or just type for Customer.")
    print("=" * 75 + "\n")

    history = []

    while True:
        try:
            user_input = input("You > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "q"]:
                print("Exiting simulator. Goodbye!")
                break
            if user_input.lower() in ["reset", "clear", "new"]:
                history = []
                print("🔄 Conversation thread reset.\n")
                continue

            if user_input.lower().startswith("agent:"):
                speaker = "Agent"
                text = user_input[6:].strip()
            else:
                speaker = "Customer"
                text = user_input

            history.append({"speaker": speaker, "text": text})

            print(f"\n[{speaker}]: \"{text}\"")
            telemetry = engine.predict(history)
            print_telemetry_card(telemetry)

        except KeyboardInterrupt:
            print("\nExiting simulator.")
            break


def main():
    args = parse_args()
    engine = StandaloneNLUInferenceEngine(
        checkpoint_path=args.checkpoint,
        model_name=args.model_name,
        device=args.device
    )

    # 1. Interactive Mode
    if args.interactive:
        run_interactive_mode(engine)
        return

    # 2. Single Input String
    if args.turns:
        try:
            turns_data = json.loads(args.turns)
            if isinstance(turns_data, str):
                turns_data = [{"speaker": "Customer", "text": turns_data}]
            elif isinstance(turns_data, dict):
                turns_data = [turns_data]
            telemetry = engine.predict(turns_data)
            print_telemetry_card(telemetry)
            print("JSON Payload:")
            print(json.dumps(telemetry, indent=2))
        except Exception as e:
            print(f"Error parsing JSON turns: {e}")
        return

    # 3. Batch File Mode
    if args.input_file:
        print(f"Processing batch file: {args.input_file}...")
        df = pd.read_csv(args.input_file)
        results = []

        text_col = "instruction" if "instruction" in df.columns else ("text" if "text" in df.columns else df.columns[0])
        for idx, row in df.iterrows():
            dialogue = [{"speaker": "Customer", "text": str(row[text_col])}]
            res = engine.predict(dialogue)
            res["input_id"] = idx
            res["input_text"] = str(row[text_col])
            results.append(res)

        output_path = args.output_file or "./outputs/batch_predictions.json"
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        print(f"✅ Batch inference completed for {len(results)} items. Saved to: {output_path}")
        return

    # Default Demo if no flags provided
    print("\nNo mode specified. Running demo on sample multi-turn support conversation:")
    demo_turns = [
        {"speaker": "Customer", "text": "I ordered 2 days ago, order ORD-88192, where is it?"},
        {"speaker": "Agent",    "text": "Please confirm your account email address so I can check."},
        {"speaker": "Customer", "text": "I already gave it twice! Stop asking stupid questions and track my order damn it!"}
    ]
    for t in demo_turns:
        print(f"  [{t['speaker']}]: \"{t['text']}\"")
    telemetry = engine.predict(demo_turns)
    print_telemetry_card(telemetry)
    print("Tip: Run `python run_inference.py --interactive` to test live conversations in real-time!")


if __name__ == "__main__":
    main()
