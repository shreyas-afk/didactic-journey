"""
Registers local dataset files as Azure ML Data Assets.
"""

import os
import argparse
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential, AzureCliCredential


def parse_args():
    parser = argparse.ArgumentParser(description="Register Data Assets in Azure ML Workspace")
    parser.add_argument("--subscription_id", type=str, required=True, help="Azure Subscription ID")
    parser.add_argument("--resource_group", type=str, required=True, help="Azure Resource Group")
    parser.add_argument("--workspace_name", type=str, required=True, help="Azure ML Workspace")
    parser.add_argument("--bitext_path", type=str, default="../Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv")
    parser.add_argument("--twitter_path", type=str, default="../twitter_augmented_train_df.csv")
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        credential = DefaultAzureCredential()
        ml_client = MLClient(credential, args.subscription_id, args.resource_group, args.workspace_name)
        ml_client.workspaces.get(args.workspace_name)
    except Exception:
        credential = AzureCliCredential()
        ml_client = MLClient(credential, args.subscription_id, args.resource_group, args.workspace_name)

    print(f"Connected to Azure ML Workspace: {args.workspace_name}")

    # 1. Register Bitext Dataset
    if os.path.exists(args.bitext_path):
        print(f"Registering Bitext dataset from: {args.bitext_path}...")
        bitext_asset = Data(
            name="bitext-customer-support-27k",
            version="1.0.0",
            description="Bitext 27K Customer Support dataset with intents, flags, categories, and entities",
            path=args.bitext_path,
            type=AssetTypes.URI_FILE
        )
        ml_client.data.create_or_update(bitext_asset)
        print("✅ Registered Data Asset: 'azureml:bitext-customer-support-27k:1.0.0'")

    # 2. Register Twitter Dataset
    if os.path.exists(args.twitter_path):
        print(f"Registering Twitter dataset from: {args.twitter_path}...")
        twitter_asset = Data(
            name="twitter-customer-support-dialogues",
            version="1.0.0",
            description="Real Twitter Customer Support Multi-Turn Dialogues with trajectory & escalation telemetry",
            path=args.twitter_path,
            type=AssetTypes.URI_FILE
        )
        ml_client.data.create_or_update(twitter_asset)
        print("✅ Registered Data Asset: 'azureml:twitter-customer-support-dialogues:1.0.0'")


if __name__ == "__main__":
    main()
