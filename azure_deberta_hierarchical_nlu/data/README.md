# Datasets & Azure ML Data Assets

This directory contains sample dataset slices and registration utilities for Azure Machine Learning.

## Dataset Schemas

### 1. Bitext Customer Support Dataset (`sample_bitext.csv` / `Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv`)
- **`flags`**: Linguistic nuance tags (e.g. `B`, `Q`, `W` = Anger, `M` = Frustration, `P` = Politeness).
- **`instruction`**: Customer utterance containing optional entity template variables (e.g. `{{Order Number}}`, `{{Refund Amount}}`).
- **`category`**: 12 Macro-Categories (`ACCOUNT`, `BILLING`, `ORDER`, `REFUND`, `TECH_SUPPORT`, etc.).
- **`intent`**: 38 Fine-grained intents (`cancel_order`, `track_package`, `get_refund`, etc.).
- **`response`**: Agent reference response.

### 2. Twitter Customer Support Multi-Turn Dataset (`sample_twitter.csv` / `twitter_augmented_train_df.csv`)
- **`instruction`**: Customer turn 1 tweet.
- **`category`**: Auto-labeled category.
- **`intent`**: Auto-labeled intent.
- **`flags`**: Linguistic flag string.
- **`trajectory`**: 4 Multi-turn conversation momentum states:
  - `STABLE_INQUIRY`
  - `ESCALATING_FRICTION`
  - `CRITICAL_CHURN_RISK`
  - `DE_ESCALATING_RESOLVED`
- **`escalation`**: Continuous escalation index $[0.0, 1.0]$.
- **`customer_effort`**: Continuous Customer Effort Score (CES) $[0.0, 1.0]$.

## Registering Full Datasets in Azure ML

To register the full 27k Bitext dataset and 312k Twitter dialogues as versioned Azure ML Data Assets:

```bash
python data/register_data_assets.py \
  --subscription_id "<SUBSCRIPTION_ID>" \
  --resource_group "<RESOURCE_GROUP>" \
  --workspace_name "<WORKSPACE_NAME>" \
  --bitext_path "../Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv" \
  --twitter_path "../twitter_augmented_train_df.csv"
```
