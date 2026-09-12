# 🚀 DeBERTa-v3 Hierarchical Turn-Transformer NLU (Azure Machine Learning)

Enterprise Dual-Level Perception Engine for Customer Support Multi-Turn Dialogue Telemetry, Intent Classification, and Real-Time Escalation Tracking on **Azure ML Compute Clusters**.

---

## 🌟 Architectural Overview

```
Customer Conversation (t=1..N)
             │
             ▼
 ┌─────────────────────────────────────────────────────────┐
 │ Level 1: DeBERTa-v3 + Masked Attention Pooling          │
 │ (Disentangled representations over non-padded tokens)   │
 └───────────────────────────┬─────────────────────────────┘
                             │ Utterance Embeddings u_t
                             ▼
 ┌─────────────────────────────────────────────────────────┐
 │ Level 2: Temporal Turn-Transformer                      │
 │ (+ Speaker Embeddings + Temporal Turn Pos Embeddings)   │
 └───────────┬───────────────────┬────────────────────┬────┘
             │                   │                    │
   [d_root || d_latest]    u_latest (768-D)   [d_global || d_latest]
             │                   │                    │
             ▼                   ▼                    ▼
   ┌───────────────────┐ ┌───────────────┐  ┌───────────────────────┐
   │ Category (12)     │ │ Emotion Flags │  │ Trajectory State (4)  │
   │ Intent (38)       │ │ (14 Nuances)  │  │ Dynamic Escalation    │
   │ Root-Grounded NLU │ └───────────────┘  │ Customer Effort (CES) │
   └───────────────────┘                    └───────────────────────┘
                                 │
                                 ▼
              Kendall Homoscedastic Multi-Task Loss
               (Automatic loss balancing across tasks)
```

---

## 📁 Directory Structure

```
azure_deberta_hierarchical_nlu/
├── data/
│   ├── sample_bitext.csv               # Sample customer support ticket rows
│   ├── sample_twitter.csv              # Sample multi-turn Twitter dialogue chains
│   ├── register_data_assets.py         # Azure ML Data Asset registration script
│   └── README.md                       # Data schema documentation
├── src/
│   ├── __init__.py
│   ├── config.py                       # 38 intents, 12 categories, 14 flags, 4 trajectories
│   ├── dataset.py                      # MLMDataset, HierarchicalDialogueDataset, entity injector
│   ├── model.py                        # Masked Attention Pooler, Turn-Transformer, Kendall Loss
│   ├── train_mlm.py                    # Stage 1: MLM Domain Adaptation with MLflow
│   ├── train_nlu.py                    # Stage 2: Hierarchical Turn-Transformer Fine-tuning
│   ├── evaluate.py                     # Stage 3: Zero-shot evaluation suite
│   ├── inference.py                    # Telemetry engine emitting JSON for LLM Agents
│   └── utils.py                        # LLRD optimizer, checkpointing, MLflow tracking
├── components/
│   ├── mlm_train_component.yaml        # Azure ML Component: Stage 1 MLM
│   ├── nlu_train_component.yaml        # Azure ML Component: Stage 2 NLU
│   └── eval_component.yaml             # Azure ML Component: Stage 3 Eval
├── environment/
│   ├── conda.yaml                      # Conda environment definition for Azure ML
│   ├── requirements.txt                # Pip dependencies
│   └── Dockerfile                      # GPU container Dockerfile
├── pipelines/
│   └── deberta_pipeline.py             # Azure ML SDK v2 Pipeline (Stage 1 -> 2 -> 3)
├── submit_azureml_job.py               # Azure ML job & pipeline submission runner
├── azure_config.json.example           # Workspace configuration template
├── run_local_smoke_test.py             # Local end-to-end smoke test script
└── README.md                           # Documentation and instructions
```

---

## ⚡ Quick Start: Local Smoke Test

You can test the entire pipeline locally without connecting to an Azure cluster:

```bash
cd azure_deberta_hierarchical_nlu
python run_local_smoke_test.py
```

This validates:
1. Multi-turn dialogue synthesis & slot injection.
2. 15% dynamic MLM masking.
3. Masked Attention Pooling over DeBERTa token states.
4. Kendall Multi-Task Uncertainty Loss backward step.
5. Layer-Wise Learning Rate Decay (LLRD) optimizer step.
6. Real-time JSON telemetry emission.

---

## ☁️ Azure Machine Learning Setup

### 1. Azure CLI Authentication
Log into your Azure subscription:
```bash
az login
az account set --subscription "<YOUR_SUBSCRIPTION_ID>"
```

### 2. Configure Workspace Coordinates
Copy `azure_config.json.example` to `azure_config.json` and fill in your values:

```json
{
  "subscription_id": "00000000-0000-0000-0000-000000000000",
  "resource_group": "my-rg",
  "workspace_name": "my-azureml-workspace",
  "compute_name": "gpu-cluster"
}
```

---

## 📊 Register Data Assets on Azure ML

Upload the full Bitext and Twitter datasets to Azure ML:

```bash
python data/register_data_assets.py \
  --subscription_id "<YOUR_SUBSCRIPTION_ID>" \
  --resource_group "<YOUR_RESOURCE_GROUP>" \
  --workspace_name "<YOUR_WORKSPACE_NAME>" \
  --bitext_path "../Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv" \
  --twitter_path "../twitter_augmented_train_df.csv"
```

---

## 🚀 Submitting Jobs to Azure ML Compute Cluster

### Option A: Submit the Full 3-Stage Pipeline (Recommended)
Chains MLM Adaptation $\rightarrow$ Hierarchical NLU Fine-Tuning $\rightarrow$ Evaluation:

```bash
python submit_azureml_job.py \
  --mode pipeline \
  --compute_name "gpu-cluster" \
  --bitext_data "azureml:bitext-customer-support-27k:1" \
  --twitter_data "azureml:twitter-customer-support-dialogues:1" \
  --wait
```

### Option B: Submit Standalone Stage 1 (MLM Domain Adaptation)
```bash
python -m src.train_mlm \
  --train_data "azureml:twitter-customer-support-dialogues:1" \
  --epochs 3 \
  --batch_size 128 \
  --output_dir "./outputs/deberta_twitter_adapted"
```

### Option C: Submit Standalone Stage 2 (Hierarchical NLU Fine-Tuning)
```bash
python -m src.train_nlu \
  --bitext_data "azureml:bitext-customer-support-27k:1" \
  --twitter_data "azureml:twitter-customer-support-dialogues:1" \
  --pretrained_backbone "microsoft/deberta-v3-base" \
  --epochs 5 \
  --batch_size 32 \
  --output_dir "./outputs/deberta_hierarchical_nlu"
```

---

## 🔮 Running Inference with Trained Model

A dedicated runner [`run_inference.py`](file:///c:/Users/Shreyas%20HV/Documents/python/genai-full/azure_deberta_hierarchical_nlu/run_inference.py) is provided:

### 1. Interactive Live Chat Simulator
Simulate customer-agent multi-turn threads in real-time in your terminal:
```bash
python run_inference.py --interactive
```

### 2. Single Multi-Turn JSON Evaluation
```bash
python run_inference.py --turns '[
  {"speaker": "Customer", "text": "I ordered 2 days ago, where is order ORD-88192?"},
  {"speaker": "Agent",    "text": "Please provide your email."},
  {"speaker": "Customer", "text": "I already gave it twice, track my order now!"}
]'
```

### 3. Batch File Inference
```bash
python run_inference.py \
  --input_file data/sample_twitter.csv \
  --output_file outputs/batch_predictions.json
```
```

---

## 📦 Real-Time Structured Telemetry Payload

The inference engine (`src/inference.py` / `run_inference.py`) outputs structured JSON for downstream LLM Agents (e.g. Qwen / Claude / GPT):

```json
{
  "intent": "track_package",
  "category": "DELIVERY",
  "intent_confidence": 0.982,
  "is_uncertain": false,
  "suggested_action": "EXECUTE_INTENT_WORKFLOW",
  "entities": {
    "Order_Number": "ORD-88192"
  },
  "trajectory_state": "CRITICAL_CHURN_RISK",
  "dynamic_escalation": 0.884,
  "customer_effort_score": 0.852,
  "current_emotion_profile": {
    "Anger": "94.2%",
    "Frustration": "88.1%",
    "Distress": "12.0%",
    "Politeness": "2.4%"
  },
  "trajectory_distribution": {
    "STABLE_INQUIRY": "0.8%",
    "ESCALATING_FRICTION": "12.4%",
    "CRITICAL_CHURN_RISK": "86.1%",
    "DE_ESCALATING_RESOLVED": "0.7%"
  }
}
```

