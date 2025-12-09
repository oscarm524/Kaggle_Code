import pandas as pd
import numpy as np

import gc
import warnings
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import lightgbm as lgb
from lightgbm import Dataset, early_stopping

warnings.filterwarnings("ignore")


def target_encode(
    X_train,
    X_val,
    y_train,
    cat_cols,
    X_test=None,
    smoothing=1.0,
    noise_level=0.01,
):
    """
    Apply target encoding with smoothing and noise to prevent overfitting
    """
    X_train_encoded = X_train.copy()
    X_val_encoded = X_val.copy()
    X_test_encoded = X_test.copy() if X_test is not None else None
    
    # Calculate global mean
    global_mean = y_train.mean()
    
    for col in cat_cols:
        # Create a temporary dataframe for groupby operations
        temp_df = X_train_encoded[[col]].copy()
        temp_df['target'] = y_train.values
        
        # Calculate category means and counts
        category_stats = (
            temp_df
            .groupby(col)['target']
            .agg(['mean', 'count'])
            .reset_index()
        )
        category_stats.columns = [
            col,
            f'{col}_mean',
            f'{col}_count',
        ]
       
        # Apply smoothing with intermediate expressions
        # to keep lines short
        numer = (
            category_stats[f'{col}_count'] * category_stats[f'{col}_mean']
            + smoothing * global_mean
        )
        denom = category_stats[f'{col}_count'] + smoothing
        category_stats[f'{col}_encoded'] = numer / denom
        
        # Add noise to prevent overfitting
        if noise_level > 0:
            noise = np.random.normal(0, noise_level, len(category_stats))
            category_stats[f'{col}_encoded'] += noise
        
        # Create mapping dictionary
        encoding_map = dict(
            zip(
                category_stats[col],
                category_stats[f'{col}_encoded']
            )
        )
        
        # Apply encoding to train and validation sets (kept within 79 chars)
        X_train_encoded[f'{col}_encoded'] = (
            X_train_encoded[col].map(encoding_map).fillna(global_mean)
        )
        X_val_encoded[f'{col}_encoded'] = (
            X_val_encoded[col].map(encoding_map).fillna(global_mean)
        )
        
        # Apply encoding to test set if provided
        if X_test_encoded is not None:
            X_test_encoded[f'{col}_encoded'] = (
                X_test_encoded[col].map(encoding_map).fillna(global_mean)
            )
    
    if X_test_encoded is not None:
        return X_train_encoded, X_val_encoded, X_test_encoded
    else:
        return X_train_encoded, X_val_encoded


df = pd.read_csv("train.csv", index_col="id")
test = pd.read_csv("test.csv", index_col="id")
cat_cols = df.select_dtypes(include="object").columns.tolist()

df["diagnosed_diabetes"] = df["diagnosed_diabetes"].astype(int)

X = df.drop("diagnosed_diabetes", axis=1)
y = df["diagnosed_diabetes"]

del df
gc.collect()

lgb_params = {
    'learning_rate': 0.12152492033540772,
    'max_depth': 4,
    'reg_alpha': 6.926015524352944,
    'reg_lambda': 9.990268389998137,
    'num_leaves': 25,
    'colsample_bytree': 0.6865041358682433,
    'subsample': 0.7054327393072288,
    'min_child_samples': 45,
    'verbose': -1,
    'n_jobs': -1,
    'device': 'gpu',
    'smoothing': 0.27555233813494145,
    'noise_level': 0.006584592103465401
}

# Extract target encoding parameters
smoothing = lgb_params.pop('smoothing')
noise_level = lgb_params.pop('noise_level')

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores, oof_preds_df, test_preds_df = [], [], []
for i, (train_index, test_index) in enumerate(skf.split(X, y)):

    X_train, X_val = X.iloc[train_index], X.iloc[test_index]
    y_train, y_val = y.iloc[train_index], y.iloc[test_index]

    # Apply target encoding within the fold
    X_train_encoded, X_val_encoded, test_encoded = target_encode(
            X_train, X_val, y_train,
            cat_cols, X_test=test, smoothing=smoothing, noise_level=noise_level
    )
            
    # Drop original categorical columns and keep encoded versions
    cols_to_drop = cat_cols
    X_train_final = X_train_encoded.drop(columns=cols_to_drop)
    X_val_final = X_val_encoded.drop(columns=cols_to_drop)

    dtrain = Dataset(X_train_final, label=y_train)
    dtest = Dataset(X_val_final, label=y_val, reference=dtrain)

    md = lgb.train(params=lgb_params,
                   train_set=dtrain,
                   num_boost_round=1000,
                   valid_sets=[dtest],
                   callbacks=[early_stopping(stopping_rounds=100,
                                             verbose=None)])

    md_pred = md.predict(X_val_final)

    oof_preds = pd.DataFrame()
    oof_preds["id"] = X_val.index
    oof_preds["lgb_pred_5"] = md_pred
    oof_preds["y"] = y_val.values
    oof_preds_df.append(oof_preds)

    # Drop original categorical columns from test set and keep encoded versions
    test_final = test_encoded.drop(columns=cols_to_drop)
    
    test_pred = md.predict(test_final)
    test_preds = pd.DataFrame()
    test_preds["lgb_pred_5"] = test_pred
    test_preds["fold"] = i
    test_preds_df.append(test_preds)

    score = roc_auc_score(y_val, md_pred)
    scores.append(score)
    print(f"Fold {i+1} AUC: {score}")

print(f"Mean AUC: {np.mean(scores)}")

# average the per-fold test predictions properly
preds_per_fold = [df["lgb_pred_5"].values for df in test_preds_df]
final_test_pred = np.mean(preds_per_fold, axis=0)
# use averaged fold predictions for submission
# (or retrain on full data if desired)
preds = final_test_pred

submission = pd.read_csv("sample_submission.csv")
submission["diagnosed_diabetes"] = preds
submission.to_csv("lgb_sub_5.csv", index=False)

print("Saving OOF predictions...")
oof_preds_df = pd.concat(oof_preds_df)
oof_preds_df.to_csv("oof_lgb_5.csv", index=False)

print("Saving test predictions...")
test_preds_df = pd.concat(test_preds_df)
test_preds_df.to_csv("test_lgb_5.csv", index=False)
print("Done!")
