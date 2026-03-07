import warnings

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from feature_engineering import (count_services,
                                 add_tenure_features,
                                 add_risk_indicator_features,
                                 add_numeric_features)

warnings.filterwarnings("ignore")

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
    'StreamingTV',
    'StreamingMovies'
]

df['TotalServices'] = df.apply(count_services, axis=1, args=(service_cols,))
df = add_tenure_features(df)
df = add_risk_indicator_features(df)
df = add_numeric_features(df)

test['TotalServices'] = test.apply(
    count_services, axis=1, args=(service_cols,)
)
test = add_tenure_features(test)
test = add_risk_indicator_features(test)
test = add_numeric_features(test)

# Convert object columns with 3 or more unique labels to category dtype
object_cols = df.select_dtypes(include='object').columns.to_list()
# Exclude target variable if it exists in object columns
if "Churn" in object_cols:
    object_cols.remove("Churn")

for col in object_cols:
    df[col] = df[col].astype('category')
    test[col] = test[col].astype('category')

X = df.drop(columns=["Churn"])
y = df["Churn"]

lgb_params = {
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.10618932444696072,
    'num_leaves': 49,
    'max_depth': 4,
    'min_child_samples': 12,
    'min_child_weight': 6.262705455272902,
    'subsample': 0.8739964202686152,
    'colsample_bytree': 0.6072342913974238,
    'reg_alpha': 0.7917445743086134,
    'reg_lambda': 0.8347751490716953,
    'device': 'gpu',
    'random_state': 42,
    'verbose': -1,
    'n_jobs': -1,
}


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cat_features = X.select_dtypes(include='category').columns.tolist()

scores, test_preds = [], []
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    dtrain = lgb.Dataset(
        X_train, label=y_train, categorical_feature=cat_features
    )
    dval = lgb.Dataset(
        X_val, label=y_val, categorical_feature=cat_features
    )

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

# Fold 0 AUC: 0.9178
# Fold 1 AUC: 0.9185
# Fold 2 AUC: 0.9180
# Fold 3 AUC: 0.9191
# Fold 4 AUC: 0.9164
# Mean AUC: 0.9179 ± 0.0009
