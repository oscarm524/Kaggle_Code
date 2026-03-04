import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import ydf
from ydf import GradientBoostedTreesLearner

import gc

df = pd.read_csv("train.csv", index_col="id")
df["SeniorCitizen"] = df["SeniorCitizen"].astype("category")
df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

test = pd.read_csv("test.csv", index_col="id")
test["SeniorCitizen"] = test["SeniorCitizen"].astype("category")

# Create a feature for total number of services
service_cols = [
    'PhoneService',
    'MultipleLines',
    'InternetService',
    'OnlineSecurity',
    'OnlineBackup',
    'DeviceProtection',
    # 'TechSupport',
    'StreamingTV',
    'StreamingMovies'
]


# Count services (excluding 'No', 'No internet service', 'No phone service')
def count_services(row):
    count = 0
    for col in service_cols:
        if col in df.columns:
            value = row[col]
            # Count as service if not 'No' or similar
            excluded_values = ['No', 'No internet service', 'No phone service']
            if value not in excluded_values:
                count += 1
    return count


df['TotalServices'] = df.apply(count_services, axis=1)
test['TotalServices'] = test.apply(count_services, axis=1)

X = df.drop(columns=["Churn"])
y = df["Churn"]

ydf_params = dict(
    task=ydf.Task.CLASSIFICATION,
    label="Churn",
    num_threads=10,
    num_trees=1000,
    max_depth=5,
    # early_stopping="LOSS_INCREASE",
    # early_stopping_num_trees_look_ahead=50,
    # validation_ratio=0.2,
)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
auc_scores, test_preds = [], []

for i, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    train_df = X_train.copy()
    train_df["Churn"] = y_train
    
    val_df = X_val.copy()
    val_df["Churn"] = y_val
    
    model = GradientBoostedTreesLearner(**ydf_params).train(train_df)
    
    y_pred = model.predict(val_df)
    auc = roc_auc_score(y_val, y_pred)
    auc_scores.append(auc)
    print(f"Fold {i + 1} AUC: {auc:.4f}")

    test_pred = model.predict(test)
    test_preds.append(test_pred)

    del model, train_df, val_df, y_pred
    gc.collect()

print(f"YDF CV AUC: {np.mean(auc_scores):.5f} ± {np.std(auc_scores):.5f}")

# Train model on 475355 examples
# Model trained in 0:01:25.347949
# Fold 1 AUC: 0.9170
# Train model on 475355 examples
# Model trained in 0:01:11.307647
# Fold 2 AUC: 0.9176
# Train model on 475355 examples
# Model trained in 0:01:46.602154
# Fold 3 AUC: 0.9174
# Train model on 475355 examples
# Model trained in 0:01:30.857511
# Fold 4 AUC: 0.9185
# Train model on 475356 examples
# Model trained in 0:01:44.331568
# Fold 5 AUC: 0.9154
# YDF CV AUC: 0.91717 ± 0.00101