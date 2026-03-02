import gc

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import ydf
from ydf import GradientBoostedTreesLearner


df = pd.read_csv("train.csv", index_col="id")
df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})
test = pd.read_csv("test.csv", index_col="id")

X = df.drop(columns=["Churn"])
y = df["Churn"]

ydf_params = dict(
    task=ydf.Task.CLASSIFICATION,
    label="Churn",
    num_threads=10,
    num_trees=1000,
    max_depth=4,
)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
auc_scores, test_preds = [], []
X = df.drop(columns=["Churn"]).copy()
y = df["Churn"].copy()

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

print(f"YDF CV AUC: {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}")

# Train model on 475355 examples
# Model trained in 0:01:22.754290
# Fold 1 AUC: 0.9167
# Train model on 475355 examples
# Model trained in 0:01:23.160146
# Fold 2 AUC: 0.9178
# Train model on 475355 examples
# Model trained in 0:01:23.085483
# Fold 3 AUC: 0.9170
# Train model on 475355 examples
# Model trained in 0:01:22.692234
# Fold 4 AUC: 0.9182
# Train model on 475356 examples
# Model trained in 0:01:22.564283
# Fold 5 AUC: 0.9153
# YDF CV AUC: 0.9170 ± 0.0010
