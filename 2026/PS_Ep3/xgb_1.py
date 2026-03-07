import gc

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from feature_engineering import (count_services,
                                 add_tenure_features,
                                 add_risk_indicator_features,
                                 add_numeric_features)

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

X = df.drop(columns=["Churn"])
y = df["Churn"]

cat_cols = X.select_dtypes(include=['object']).columns.tolist()

for col in cat_cols:
    if col in X.columns:
        X[col] = X[col].astype('category')
        test[col] = test[col].astype('category')

dtest = xgb.DMatrix(test, enable_categorical=True)

xgb_params = {
    'objective': 'binary:logistic',
    'device': 'cuda',
    'eval_metric': 'auc',
    'max_depth': 7,
    'learning_rate': 0.020875348387155237,
    'subsample': 0.8119481217352481,
    'colsample_bytree': 0.7964084935197875,
    'seed': 42,
    'nthread': -1
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores, oof_preds_df = [], []
test_preds_df = np.zeros(len(test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # Create DMatrix objects with categorical support enabled
    dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
    dval = xgb.DMatrix(X_val, label=y_val, enable_categorical=True)

    # Train the model using xgb.train
    model = xgb.train(
            xgb_params,
            dtrain,
            num_boost_round=1000,
            evals=[(dval, 'validation')],
            early_stopping_rounds=50,
            verbose_eval=False
        )

    # Predict probabilities on the validation set
    y_pred = model.predict(dval)
    auc = roc_auc_score(y_val, y_pred)
    scores.append(auc)
    print(f"Fold {fold + 1} AUC: {auc:.4f}")

    oof_pred = pd.DataFrame()
    oof_pred["id"] = X_val.index
    oof_pred["xgb_4"] = model.predict(dval)
    oof_pred["y"] = y_val.values
    oof_preds_df.append(oof_pred)

    test_preds_df += model.predict(dtest) / skf.n_splits

    # Clear memory
    del X_train, X_val, y_train, y_val, dtrain, dval, model
    gc.collect()

print(f"Mean AUC: {np.mean(scores):.4f} ± {np.std(scores):.4f}")

# Fold 1 AUC: 0.9175
# Fold 2 AUC: 0.9184
# Fold 3 AUC: 0.9178
# Fold 4 AUC: 0.9190
# Fold 5 AUC: 0.9162
# Mean AUC: 0.9178 ± 0.0009
