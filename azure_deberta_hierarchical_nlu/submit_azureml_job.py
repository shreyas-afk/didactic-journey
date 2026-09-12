"""
Azure ML Job Submission Runner (Python SDK v2).
Allows submitting single training jobs or the full 3-stage pipeline to an Azure ML GPU Compute Cluster.
"""

import os
import json
import argparse
from azure.ai.ml import MLClient, Input
from azure.identity import DefaultAzureCredential, AzureCliCredential
from pipelines.deberta_pipeline import build_deberta_nlu_pipeline


def load_azure_config(config_path: str = "azure_config.json") -> dict:
    """Loads workspace settings from JSON if present."""
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def parse_args():
    parser = argparse.ArgumentParser(description="Submit DeBERTa NLU Jobs / Pipeline to Azure ML Compute Cluster")
    parser.add_argument("--config", type=str, default="azure_config.json", help="Path to Azure ML configuration JSON")
    parser.add_argument("--subscription_id", type=str, default=None, help="Azure Subscription ID")
    parser.add_argument("--resource_group", type=str, default=None, help="Azure Resource Group Name")
    parser.add_argument("--workspace_name", type=str, default=None, help="Azure ML Workspace Name")
    parser.add_argument("--compute_name", type=str, default=None, help="Target Azure ML GPU Compute Cluster Name")
    parser.add_argument("--mode", type=str, choices=["pipeline", "mlm_only", "nlu_only"], default="pipeline", help="Execution mode")
    parser.add_argument("--bitext_data", type=str, default="./data/sample_bitext.csv", help="Local path or Azure ML Data Asset URI for Bitext dataset")
    parser.add_argument("--twitter_data", type=str, default="./data/sample_twitter.csv", help="Local path or Azure ML Data Asset URI for Twitter dataset")
    parser.add_argument("--experiment_name", type=str, default="DeBERTa-v3-Enterprise-NLU", help="MLflow / Azure ML Experiment name")
    parser.add_argument("--wait", action="store_true", help="Block and stream job logs until completion")
    return parser.parse_args()


def get_ml_client(subscription_id: str, resource_group: str, workspace_name: str) -> MLClient:
    """Authenticates using DefaultAzureCredential with fallback to AzureCliCredential."""
    try:
        credential = DefaultAzureCredential()
        client = MLClient(credential, subscription_id, resource_group, workspace_name)
        # Test auth
        client.workspaces.get(workspace_name)
        return client
    except Exception:
        print("DefaultAzureCredential fallback to AzureCliCredential...")
        credential = AzureCliCredential()
        return MLClient(credential, subscription_id, resource_group, workspace_name)


def main():
    args = parse_args()
    cfg = load_azure_config(args.config)

    subscription_id = args.subscription_id or cfg.get("subscription_id") or os.environ.get("AZURE_SUBSCRIPTION_ID")
    resource_group = args.resource_group or cfg.get("resource_group") or os.environ.get("AZURE_RESOURCE_GROUP")
    workspace_name = args.workspace_name or cfg.get("workspace_name") or os.environ.get("AZURE_WORKSPACE_NAME")
    compute_name = args.compute_name or cfg.get("compute_name") or os.environ.get("AZURE_COMPUTE_NAME", "gpu-cluster")

    if not all([subscription_id, resource_group, workspace_name]):
        raise ValueError(
            "Missing Azure ML workspace coordinates. Please provide --subscription_id, --resource_group, "
            "and --workspace_name, or fill in 'azure_config.json'."
        )

    print("=" * 80)
    print("AZURE MACHINE LEARNING JOB SUBMISSION")
    print(f"Subscription ID : {subscription_id}")
    print(f"Resource Group  : {resource_group}")
    print(f"Workspace       : {workspace_name}")
    print(f"Compute Cluster : {compute_name}")
    print(f"Execution Mode  : {args.mode}")
    print("=" * 80)

    ml_client = get_ml_client(subscription_id, resource_group, workspace_name)
    print(f"✅ Connected to Azure ML Workspace: '{workspace_name}'")

    if args.mode == "pipeline":
        print("\nConstructing End-to-End DeBERTa Pipeline...")
        pipeline_job = build_deberta_nlu_pipeline(
            twitter_mlm_data=Input(type="uri_file", path=args.twitter_data),
            bitext_train_data=Input(type="uri_file", path=args.bitext_data),
            twitter_nlu_data=Input(type="uri_file", path=args.twitter_data),
            base_model_name="microsoft/deberta-v3-base",
            mlm_epochs=3,
            nlu_epochs=5
        )
        pipeline_job.settings.default_compute = compute_name
        pipeline_job.settings.force_rerun = True

        print(f"Submitting pipeline to cluster '{compute_name}' under experiment '{args.experiment_name}'...")
        submitted_job = ml_client.jobs.create_or_update(
            pipeline_job,
            experiment_name=args.experiment_name
        )

        print("\n🚀 Pipeline Job Submitted Successfully!")
        print(f"Job Name    : {submitted_job.name}")
        print(f"Studio URL  : {submitted_job.services['Studio'].endpoint if hasattr(submitted_job, 'services') and 'Studio' in submitted_job.services else submitted_job.studio_url}")

        if args.wait:
            print("\nStreaming job logs from Azure ML...")
            ml_client.jobs.stream(submitted_job.name)


if __name__ == "__main__":
    main()
