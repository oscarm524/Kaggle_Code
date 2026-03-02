import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import lightgbm as lgb

df = pd.read_csv("train.csv", index_col="id")
test = pd.read_csv("test.csv", index_col="id")

# Convert Yes/No columns to binary
yes_no_cols = df.select_dtypes(include='object').columns
for col in yes_no_cols:
    if (df[col].nunique() == 2 and
            set(df[col].unique()).issubset({'Yes', 'No'})):
        if col != "Churn":
            df[col] = (df[col] == 'Yes').astype(int)
            test[col] = (test[col] == 'Yes').astype(int)
        else:
            df[col] = (df[col] == 'Yes').astype(int)

# Convert object columns with 3 or more unique labels to category dtype
object_cols = df.select_dtypes(include='object').columns

for col in object_cols:
    if df[col].nunique() >= 3:
        df[col] = df[col].astype('category')
        test[col] = test[col].astype('category')

X = df.drop(columns=["Churn"])
X['gender'] = X['gender'].map({"Female": 0, "Male": 1})
test['gender'] = test['gender'].map({"Female": 0, "Male": 1})
y = df["Churn"]

lgb_params = {
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.13309456089288524,
    'num_leaves': 22,
    'max_depth': 4,  # 3
    'min_child_samples': 3,
    'min_child_weight': 7.884744663108288,
    'subsample': 0.9254726647847019,
    'colsample_bytree': 0.8160060837693105,
    'reg_alpha': 0.8598760104445989,
    'reg_lambda': 0.9072349527119281,
    'device': 'gpu',
    'random_state': 42,
    'verbose': -1,
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cat_features = X.select_dtypes(include='category').columns.tolist()

# dtest = lgb.Dataset(test, categorical_feature=cat_features)

scores, test_preds = [], []
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    dtrain = lgb.Dataset(X_train, label=y_train,
                         categorical_feature=cat_features)
    dval = lgb.Dataset(X_val, label=y_val,
                       categorical_feature=cat_features)

    model = lgb.train(
        params=lgb_params,
        train_set=dtrain,
        num_boost_round=1000,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )

    y_pred = model.predict(X_val)
    auc_score = roc_auc_score(y_val, y_pred)
    scores.append(auc_score)

    print(f"Fold {fold} AUC: {auc_score:.4f}")

    test_pred = model.predict(test)
    test_preds.append(test_pred)

print(f"Mean AUC: {np.mean(scores):.4f} ± {np.std(scores):.4f}")

# Fold 1 AUC: 0.9152
# Fold 2 AUC: 0.9167
# Fold 3 AUC: 0.9158
# Fold 4 AUC: 0.9168
# Fold 5 AUC: 0.9141
# Mean AUC: 0.9157