"""
Azure ML Pipeline (SDK v2) for DeBERTa-v3 Hierarchical NLU.
Chains:
  1. Stage 1: Domain Adaptation MLM Pre-training
  2. Stage 2: Hierarchical Turn-Transformer Multi-Task Fine-Tuning
  3. Stage 3: Benchmark Evaluation & Telemetry Export
"""

import os
from azure.ai.ml import dsl, Input, load_component

# Resolve path to component specs
COMPONENTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "components")
mlm_component = load_component(source=os.path.join(COMPONENTS_DIR, "mlm_train_component.yaml"))
nlu_component = load_component(source=os.path.join(COMPONENTS_DIR, "nlu_train_component.yaml"))
eval_component = load_component(source=os.path.join(COMPONENTS_DIR, "eval_component.yaml"))


@dsl.pipeline(
    name="deberta_hierarchical_nlu_pipeline",
    display_name="DeBERTa-v3 Hierarchical NLU End-to-End Pipeline",
    description="Full 3-stage training pipeline: Customer Support MLM Adaptation -> Hierarchical Turn-Transformer NLU -> Evaluation"
)
def build_deberta_nlu_pipeline(
    twitter_mlm_data: Input(type="uri_file"),
    bitext_train_data: Input(type="uri_file"),
    twitter_nlu_data: Input(type="uri_file", optional=True),
    base_model_name: str = "microsoft/deberta-v3-base",
    mlm_epochs: int = 3,
    mlm_batch_size: int = 128,
    nlu_epochs: int = 5,
    nlu_batch_size: int = 32,
    base_lr: float = 0.00002,
    head_lr: float = 0.00006
):
    """
    Assembles the 3-step training workflow on Azure ML Compute Cluster.
    """
    # Step 1: MLM Domain Adaptation
    mlm_step = mlm_component(
        train_data=twitter_mlm_data,
        model_name=base_model_name,
        epochs=mlm_epochs,
        batch_size=mlm_batch_size
    )
    mlm_step.display_name = "1. Customer Support MLM Adaptation"

    # Step 2: Hierarchical Turn-Transformer Fine-Tuning
    nlu_step = nlu_component(
        bitext_data=bitext_train_data,
        twitter_data=twitter_nlu_data,
        pretrained_backbone=mlm_step.outputs.adapted_backbone_dir,
        epochs=nlu_epochs,
        batch_size=nlu_batch_size,
        base_lr=base_lr,
        head_lr=head_lr
    )
    nlu_step.display_name = "2. Hierarchical NLU Fine-Tuning"

    # Step 3: Zero-shot Evaluation & Telemetry Export
    eval_step = eval_component(
        trained_model_dir=nlu_step.outputs.trained_model_dir
    )
    eval_step.display_name = "3. Benchmark Evaluation"

    return {
        "adapted_backbone": mlm_step.outputs.adapted_backbone_dir,
        "trained_nlu_model": nlu_step.outputs.trained_model_dir,
        "evaluation_report": eval_step.outputs.eval_output_dir
    }
