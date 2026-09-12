"""
Evaluation & Benchmark Suite for DeBERTa-v3 Hierarchical Turn-Transformer NLU.
Runs synthetic and real-world multi-turn dialogues through the inference engine.
"""

import os
import json
import argparse
import pandas as pd

from .inference import DeBERTaHierarchicalNLUPredictor
from .utils import get_device, init_mlflow_run, log_mlflow_metrics


BENCHMARK_CASES = [
    {
        "case_id": "CASE-01-SEVERE-ESCALATION",
        "description": "Order tracking failure leading to anger, profanity, and churn threat",
        "turns": [
            {"speaker": "Customer", "text": "I ordered 2 days ago, order ORD-88192, where is it?"},
            {"speaker": "Agent",    "text": "Please confirm your account email address so I can check."},
            {"speaker": "Customer", "text": "I already gave it twice! Stop asking stupid questions and track my order damn it!"}
        ],
        "expected_intent": "track_package",
        "expected_trajectory": "CRITICAL_CHURN_RISK",
        "min_escalation": 0.50
    },
    {
        "case_id": "CASE-02-ROUTINE-ORDER-STATUS",
        "description": "Routine order tracking with polite clarification and stable continuation",
        "turns": [
            {"speaker": "Customer", "text": "Hello, could you help me check the tracking status of order ORD-55421?"},
            {"speaker": "Agent",    "text": "Sure! Could you verify your delivery city?"},
            {"speaker": "Customer", "text": "Yes, the delivery city is Chicago. Thanks for checking!"}
        ],
        "expected_intent": "track_package",
        "expected_trajectory": "STABLE_INQUIRY",
        "max_escalation": 0.25
    },
    {
        "case_id": "CASE-03-TECH-SUPPORT-FRICTION",
        "description": "Repeated internet disconnection with escalating customer frustration",
        "turns": [
            {"speaker": "Customer", "text": "My internet cuts out every 20 minutes this is ridiculous."},
            {"speaker": "Agent",    "text": "Are the lights blinking on your router?"},
            {"speaker": "Customer", "text": "Yes, red light keeps turning on. Fix my connection."}
        ],
        "expected_intent": "network_outage",
        "expected_trajectory": "ESCALATING_FRICTION"
    },
    {
        "case_id": "CASE-04-REFUND-INQUIRY",
        "description": "Refund amount dispute on invoice INV-99012",
        "turns": [
            {"speaker": "Customer", "text": "I was charged $49.99 on invoice INV-99012 but I requested a cancellation."},
            {"speaker": "Agent",    "text": "Let me review your billing history for invoice INV-99012."},
            {"speaker": "Customer", "text": "Thank you, please make sure the refund of $49.99 is processed."}
        ],
        "expected_intent": "get_refund"
    }
]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate DeBERTa Hierarchical NLU Model")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory containing model checkpoint and tokenizer")
    parser.add_argument("--output_report", type=str, default="./outputs/eval_report.json", help="Path to save evaluation report JSON")
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()

    init_mlflow_run(experiment_name="DeBERTa-Hierarchical-NLU", run_name="Stage3-Model-Evaluation")

    print("=" * 80)
    print("STAGE 3: HIERARCHICAL NLU MODEL EVALUATION")
    print(f"Model Directory : {args.model_dir}")
    print(f"Output Report   : {args.output_report}")
    print("=" * 80)

    predictor = DeBERTaHierarchicalNLUPredictor(model_dir=args.model_dir, device=device)

    results = []
    print("\nRunning Benchmark Test Suite...")

    for case in BENCHMARK_CASES:
        print(f"\n📁 [{case['case_id']}] {case['description']}")
        for t in case["turns"]:
            print(f"   [{t['speaker']}]: \"{t['text']}\"")

        telemetry = predictor.predict_dialogue_trajectory(case["turns"])
        badge = "🚨 [CRITICAL ESCALATION]" if telemetry["dynamic_escalation"] > 0.40 else "✅ [NORMAL]"

        print("\n   📦 Telemetry Output:")
        print(f"      ├─ Intent          : {telemetry['intent']} ({telemetry['category']}) | Conf: {telemetry['intent_confidence']*100:.1f}%")
        print(f"      ├─ Action Guidance : {telemetry['suggested_action']}")
        print(f"      ├─ Extracted Slots : {telemetry['entities']}")
        print(f"      ├─ Trajectory State: 🎯 {telemetry['trajectory_state']}")
        print(f"      ├─ Escalation Index: {telemetry['dynamic_escalation']} {badge}")
        print(f"      ├─ Customer Effort : {telemetry['customer_effort_score']}")
        print(f"      └─ Emotion Nuance  : {telemetry['current_emotion_profile']}")
        print("-" * 80)

        results.append({
            "case_id": case["case_id"],
            "expected_intent": case.get("expected_intent"),
            "predicted_intent": telemetry["intent"],
            "predicted_category": telemetry["category"],
            "confidence": telemetry["intent_confidence"],
            "trajectory_state": telemetry["trajectory_state"],
            "escalation_index": telemetry["dynamic_escalation"],
            "effort_score": telemetry["customer_effort_score"],
            "extracted_slots": telemetry["entities"]
        })

    report = {
        "model_dir": args.model_dir,
        "total_test_cases": len(BENCHMARK_CASES),
        "results": results
    }

    os.makedirs(os.path.dirname(args.output_report) or ".", exist_ok=True)
    with open(args.output_report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n✅ Evaluation report saved to: {args.output_report}")


if __name__ == "__main__":
    main()
