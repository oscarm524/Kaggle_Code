import gc

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import ydf
from ydf import GradientBoostedTreesLearner

df = pd.read_csv("train.csv", index_col="id")
df["SeniorCitizen"] = df["SeniorCitizen"].astype("category")
df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

test = pd.read_csv("test.csv", index_col="id")
test["SeniorCitizen"] = test["SeniorCitizen"].astype("category")

X = df.drop(columns=["Churn"])
y = df["Churn"]

ydf_params = {
    'task': ydf.Task.CLASSIFICATION,
    'label': 'Churn',
    'num_threads': 10,
    'shrinkage': 0.1,
    'num_trees': 10000,
    'early_stopping_num_trees_look_ahead': 300,
    'max_depth': 2,
    'growing_strategy': 'BEST_FIRST_GLOBAL',
    'categorical_algorithm': 'RANDOM',
}

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
# Model trained in 0:06:09.693086
# Fold 1 AUC: 0.9176
# Train model on 475355 examples
# Model trained in 0:06:36.738630
# Fold 2 AUC: 0.9183
# Train model on 475355 examples
# Model trained in 0:08:24.918109
# Fold 3 AUC: 0.9179
# Train model on 475355 examples
# Model trained in 0:07:49.827679
# Fold 4 AUC: 0.9189
# Train model on 475356 examples
# Model trained in 0:06:27.181008
# Fold 5 AUC: 0.9162
# YDF CV AUC: 0.91779 ± 0.00092
