import pandas as pd
import numpy as np

import gc
import warnings

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import xgboost as xgb

warnings.filterwarnings("ignore")


def target_encode(
    X_train,
    X_val,
    y_train,
    cat_cols,
    X_test=None,
    smoothing=1.0,
    noise_level=0.01,
    cv_folds=5,
):
    """
    Apply target encoding with cross-validation, smoothing and noise
    to prevent overfitting
    """    

    X_train_encoded = X_train.copy()
    X_val_encoded = X_val.copy()
    X_test_encoded = X_test.copy() if X_test is not None else None
    
    # Calculate global mean
    global_mean = y_train.mean()
    
    # Initialize CV splitter
    kf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=1)
    
    for col in cat_cols:
        # Initialize encoded column with global mean for training data
        X_train_encoded[f'{col}_encoded'] = global_mean
        
        # Cross-validation encoding for training data
        for train_idx, val_idx in kf.split(X_train, y_train):
            # Get CV train and validation subsets
            cv_train_X = X_train.iloc[train_idx]
            cv_train_y = y_train.iloc[train_idx]
            cv_val_X = X_train.iloc[val_idx]
            
            # Create temporary dataframe for groupby operations
            temp_df = cv_train_X[[col]].copy()
            temp_df['target'] = cv_train_y.values
            
            # Calculate category means and counts on CV training data
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
           
            # Apply smoothing
            numer = (
                category_stats[f'{col}_count'] * category_stats[f'{col}_mean']
                + smoothing * global_mean
            )
            denom = category_stats[f'{col}_count'] + smoothing
            category_stats[f'{col}_encoded'] = numer / denom
            
            # Create mapping dictionary
            cv_encoding_map = dict(
                zip(
                    category_stats[col],
                    category_stats[f'{col}_encoded']
                )
            )
            
            # Apply encoding to CV validation subset
            cv_val_indices = cv_val_X.index
            X_train_encoded.loc[cv_val_indices, f'{col}_encoded'] = (
                cv_val_X[col].map(cv_encoding_map).fillna(global_mean)
            )
        
        # Add noise to training encoded values to prevent overfitting
        if noise_level > 0:
            noise = np.random.normal(0, noise_level, len(X_train_encoded))
            X_train_encoded[f'{col}_encoded'] += noise
        
        # For validation and test sets, use encoding from full training data
        temp_df = X_train[[col]].copy()
        temp_df['target'] = y_train.values
        
        # Calculate category means and counts on full training data
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
       
        # Apply smoothing
        numer = (
            category_stats[f'{col}_count'] * category_stats[f'{col}_mean']
            + smoothing * global_mean
        )
        denom = category_stats[f'{col}_count'] + smoothing
        category_stats[f'{col}_encoded'] = numer / denom
        
        # Create mapping dictionary for validation and test
        encoding_map = dict(
            zip(
                category_stats[col],
                category_stats[f'{col}_encoded']
            )
        )
        
        # Apply encoding to validation set
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


# Load and prepare data
df = pd.read_csv("train.csv", index_col="id")
df["diagnosed_diabetes"] = df["diagnosed_diabetes"].astype(int)
test = pd.read_csv("test.csv", index_col="id")

all_cols = df.columns.tolist()
all_cols.remove("diagnosed_diabetes")
cat_cols = df.select_dtypes(include="object").columns.tolist()

original = pd.read_csv("diabetes_dataset.csv")

for col in all_cols:

    mean_col = pd.DataFrame(
        original.groupby(col)["diagnosed_diabetes"]
        .mean()).reset_index()
    mean_col = mean_col.rename(columns={"diagnosed_diabetes": f"{col}_mean"})
    df = df.merge(mean_col, on=col, how="left")
    test = test.merge(mean_col, on=col, how="left")

    count_col = pd.DataFrame(
        original.groupby(col)["diagnosed_diabetes"]
        .size()).reset_index()
    count_col = count_col.rename(columns={"diagnosed_diabetes": f"{col}_count"})
    df = df.merge(count_col, on=col, how="left")
    test = test.merge(count_col, on=col, how="left")

X = df.drop("diagnosed_diabetes", axis=1)
y = df["diagnosed_diabetes"]

del df
gc.collect()

xgb_params = {
    'device': 'cuda',
    'max_depth': 5,
    'learning_rate': 0.04401817650540594,
    'gamma': 0.10305882552604473,
    'min_child_weight': 39,
    'colsample_bytree': 0.8,
    'reg_lambda': 1.4448644594554028,
    'reg_alpha': 9.479310936823271,
    'n_jobs': -1,
    'smoothing': 0.3024290135590247,
    'noise_level': 0.03556370617715808
}

# Extract target encoding parameters
smoothing = xgb_params.pop('smoothing')
noise_level = xgb_params.pop('noise_level')

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
    test_final = test_encoded.drop(columns=cols_to_drop)

    dtrain = xgb.QuantileDMatrix(X_train_final, label=y_train)
    dvalid = xgb.QuantileDMatrix(X_val_final, label=y_val)
    
    xgb_md = xgb.train(xgb_params,
                       dtrain=dtrain,
                       num_boost_round=1000,
                       evals=[(dvalid, 'validation')],
                       early_stopping_rounds=100,
                       verbose_eval=False)
    xgb_pred = xgb_md.predict(dvalid)

    oof_preds = pd.DataFrame()
    oof_preds["id"] = X_val.index
    oof_preds["xgb_pred_5"] = xgb_pred
    oof_preds["y"] = y_val.values
    oof_preds_df.append(oof_preds)

    test_pred = xgb_md.predict(xgb.QuantileDMatrix(test_final))
    test_preds = pd.DataFrame()
    test_preds["xgb_pred_5"] = test_pred
    test_preds["fold"] = i
    test_preds_df.append(test_preds)

    score = roc_auc_score(y_val, xgb_pred)
    scores.append(score)
    print(f"Fold {i+1} AUC: {score}")

print(f"Mean AUC: {np.mean(scores)}")
# average the per-fold test predictions properly
preds_per_fold = [df["xgb_pred_5"].values for df in test_preds_df]
final_test_pred = np.mean(preds_per_fold, axis=0)
# use averaged fold predictions for submission
# (or retrain on full data if desired)
preds = final_test_pred

submission = pd.read_csv("sample_submission.csv")
submission["diagnosed_diabetes"] = preds
submission.to_csv("xgb_sub_5.csv", index=False)

print("Saving OOF predictions...")
oof_preds_df = pd.concat(oof_preds_df)
oof_preds_df.to_csv("oof_xgb_5.csv", index=False)

print("Saving test predictions...")
test_preds_df = pd.concat(test_preds_df)
test_preds_df.to_csv("test_xgb_5.csv", index=False)
print("Done!")
