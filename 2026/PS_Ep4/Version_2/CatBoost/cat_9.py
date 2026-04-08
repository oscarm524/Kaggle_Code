"""CatBoost experiment 9 -- All combined features + LR stacking.

Combines interactions, ratios, log/poly, frequency encoding,
second decimal digit, clustering, AND logistic regression stacking
feature. Kitchen-sink approach.
"""

import gc
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, OneHotEncoder, StandardScaler
from sklearn.cluster import MiniBatchKMeans
from catboost import CatBoostClassifier, Pool

BASE_DIR = "/media/oscarm524/Expansion/Kaggle/2026/PS_Ep4/Version_1"
TRAIN_PATH = f"{BASE_DIR}/train.csv"
TEST_PATH = f"{BASE_DIR}/test.csv"
SUBMISSION_PATH = f"{BASE_DIR}/sample_submission.csv"
SAVE_DIR = "/media/oscarm524/Expansion/Kaggle/2026/PS_Ep4/Version_2/CatBoost"
OUTPUT_PATH = f"{SAVE_DIR}/cat_9.csv"
OOF_PATH = f"{SAVE_DIR}/cat_9_oof.csv"
TEST_PROBS_PATH = f"{SAVE_DIR}/cat_9_test.csv"

NUMERIC_COLS = [
    "Soil_Moisture", "Temperature_C", "Rainfall_mm",
    "Wind_Speed_kmh", "Humidity", "Soil_pH",
    "Organic_Carbon", "Electrical_Conductivity",
    "Sunlight_Hours", "Field_Area_hectare",
    "Previous_Irrigation_mm",
]

label_map = {0: "Low", 1: "Medium", 2: "High"}


def add_decimal_digits(df):
    for col in NUMERIC_COLS:
        v = df[col].values
        df[f"{col}_dec1"] = np.floor((v - np.floor(v)) * 10).astype(int)
        df[f"{col}_dec2"] = np.floor((v * 100) % 10).astype(int)
    return df


def add_binary_flags(df):
    df["soil_lt_25"] = (df["Soil_Moisture"] < 25).astype(int)
    df["rain_lt_300"] = (df["Rainfall_mm"] < 300).astype(int)
    df["temp_gt_30"] = (df["Temperature_C"] > 30).astype(int)
    df["wind_gt_10"] = (df["Wind_Speed_kmh"] > 10).astype(int)
    df["is_harvest"] = (df["Crop_Growth_Stage"] == "Harvest").astype(int)
    df["is_sowing"] = (df["Crop_Growth_Stage"] == "Sowing").astype(int)
    df["mulching_yes"] = (df["Mulching_Used"] == "Yes").astype(int)
    return df


def add_interactions(df):
    df["moisture_x_temp"] = df["Soil_Moisture"] * df["Temperature_C"]
    df["moisture_x_humidity"] = df["Soil_Moisture"] * df["Humidity"]
    df["rain_x_humidity"] = df["Rainfall_mm"] * df["Humidity"]
    df["temp_x_humidity"] = df["Temperature_C"] * df["Humidity"]
    df["rain_x_area"] = df["Rainfall_mm"] * df["Field_Area_hectare"]
    df["sun_x_temp"] = df["Sunlight_Hours"] * df["Temperature_C"]
    df["wind_x_humidity"] = df["Wind_Speed_kmh"] * df["Humidity"]
    df["prev_irr_x_moisture"] = df["Previous_Irrigation_mm"] * df["Soil_Moisture"]
    df["pH_x_carbon"] = df["Soil_pH"] * df["Organic_Carbon"]
    df["EC_x_moisture"] = df["Electrical_Conductivity"] * df["Soil_Moisture"]
    return df


def add_ratios(df):
    eps = 1e-6
    df["rain_per_hectare"] = df["Rainfall_mm"] / (df["Field_Area_hectare"] + eps)
    df["moisture_over_humidity"] = df["Soil_Moisture"] / (df["Humidity"] + eps)
    df["temp_over_wind"] = df["Temperature_C"] / (df["Wind_Speed_kmh"] + eps)
    df["rain_over_sun"] = df["Rainfall_mm"] / (df["Sunlight_Hours"] + eps)
    df["moisture_over_temp"] = df["Soil_Moisture"] / (df["Temperature_C"] + eps)
    return df


def add_log_poly(df):
    for col in ["Rainfall_mm", "Field_Area_hectare", "Previous_Irrigation_mm"]:
        df[f"{col}_log"] = np.log1p(df[col])
    for col in ["Soil_Moisture", "Temperature_C", "Humidity"]:
        df[f"{col}_sq"] = df[col] ** 2
    df["Rainfall_mm_cbrt"] = np.cbrt(df["Rainfall_mm"])
    return df


def add_freq_encoding(train_df, test_df, cat_columns):
    for col in cat_columns:
        freq = train_df[col].value_counts(normalize=True)
        train_df[f"{col}_freq"] = train_df[col].map(freq).astype(float)
        test_df[f"{col}_freq"] = test_df[col].map(freq).fillna(0).astype(float)
    return train_df, test_df


def add_cluster_features(X_tr, X_te, n_clusters=15):
    scaler = StandardScaler()
    tr_sc = scaler.fit_transform(X_tr[NUMERIC_COLS])
    te_sc = scaler.transform(X_te[NUMERIC_COLS])
    km = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=10000)
    km.fit(tr_sc)
    X_tr["cluster_id"] = km.predict(tr_sc)
    X_te["cluster_id"] = km.predict(te_sc)
    tr_d = km.transform(tr_sc)
    te_d = km.transform(te_sc)
    X_tr["cluster_dist"] = tr_d[np.arange(len(tr_d)), X_tr["cluster_id"].values]
    X_te["cluster_dist"] = te_d[np.arange(len(te_d)), X_te["cluster_id"].values]
    X_tr["cluster_min_dist"] = tr_d.min(axis=1)
    X_te["cluster_min_dist"] = te_d.min(axis=1)
    return X_tr, X_te


# -- Load Data -----------------------------------------------------
train = pd.read_csv(TRAIN_PATH, index_col="id")
test = pd.read_csv(TEST_PATH, index_col="id")

X_raw = train.drop(columns=["Irrigation_Need"])
y = train["Irrigation_Need"].map({"Low": 0, "Medium": 1, "High": 2})
X_test_raw = test.copy()

# -- LR stacking feature -------------------------------------------
cat_cols_lr = X_raw.select_dtypes(include="object").columns
ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
X_cat = ohe.fit_transform(X_raw[cat_cols_lr])
X_num = X_raw.drop(columns=cat_cols_lr).values
X_lr = np.hstack([X_num, X_cat])

base_logit = make_pipeline(
    RobustScaler(),
    LogisticRegression(C=0.1, class_weight="balanced",
                       multi_class="multinomial", max_iter=1000,
                       random_state=42),
)
base_logit.fit(X_lr, y)
tr_lr_pred = base_logit.predict(X_lr)
print("Base LR BA:", balanced_accuracy_score(y, tr_lr_pred))

X_te_cat = ohe.transform(X_test_raw[cat_cols_lr])
X_te_num = X_test_raw.drop(columns=cat_cols_lr).values
X_te_lr = np.hstack([X_te_num, X_te_cat])
te_lr_pred = base_logit.predict(X_te_lr)

del X_cat, X_num, X_lr, X_te_cat, X_te_num, X_te_lr
gc.collect()

# -- Build features ------------------------------------------------
X = train.drop(columns=["Irrigation_Need"])
X["original_pred"] = [label_map[i] for i in tr_lr_pred]
X_test = test.copy()
X_test["original_pred"] = [label_map[i] for i in te_lr_pred]

orig_cat_cols = X.select_dtypes(include="object").columns.tolist()

for func in [add_decimal_digits, add_binary_flags, add_interactions,
             add_ratios, add_log_poly]:
    X = func(X)
    X_test = func(X_test)

X, X_test = add_freq_encoding(X, X_test, orig_cat_cols)
X, X_test = add_cluster_features(X, X_test, n_clusters=15)

cat_cols = X.select_dtypes(include="object").columns.tolist()

param = {
    "loss_function": "MultiClass",
    "classes_count": 3,
    "depth": 4,
    "learning_rate": 0.1,
    "l2_leaf_reg": 1.0,
    "iterations": 5000,
    "border_count": 2000,
    "early_stopping_rounds": 300,
    "random_seed": 42,
    "verbose": 100,
    "task_type": "GPU",
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X))
oof_probs = np.zeros((len(X), 3))
test_preds = np.zeros((len(X_test), 3))
fold_scores = []

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
    print(f"Fold {fold+1} / 5")
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

    cw = compute_class_weight("balanced", classes=np.sort(y_tr.unique()), y=y_tr)
    fp = {**param, "class_weights": list(cw)}

    tp = Pool(X_tr, label=y_tr, cat_features=cat_cols)
    vp = Pool(X_va, label=y_va, cat_features=cat_cols)
    ep = Pool(X_test, cat_features=cat_cols)

    model = CatBoostClassifier(**fp)
    model.fit(tp, eval_set=vp)

    oof_preds[va_idx] = model.predict(vp).flatten().astype(int)
    sc = balanced_accuracy_score(y_va, oof_preds[va_idx])
    fold_scores.append(sc)
    print(f"  Fold {fold+1} Balanced Accuracy: {sc:.6f}\n")

    # OOF likelihoods
    oof_probs[va_idx] = model.predict_proba(vp)

    test_preds += model.predict_proba(ep) / 5

print(f"Overall OOF Balanced Accuracy: {np.mean(fold_scores):.6f}")

# -- Save OOF Predictions ------------------------------------------
oof_df = pd.DataFrame({
    "id": X.index,
    "prob_Low": oof_probs[:, 0],
    "prob_Medium": oof_probs[:, 1],
    "prob_High": oof_probs[:, 2],
})
oof_df.to_csv(OOF_PATH, index=False)
print(f"OOF predictions saved to {OOF_PATH}")

# -- Save Test Predicted Likelihoods --------------------------------
test_probs_df = pd.DataFrame({
    "id": X_test.index,
    "prob_Low": test_preds[:, 0],
    "prob_Medium": test_preds[:, 1],
    "prob_High": test_preds[:, 2],
})
test_probs_df.to_csv(TEST_PROBS_PATH, index=False)
print(f"Test predicted likelihoods saved to {TEST_PROBS_PATH}")

# -- Generate Submission --------------------------------------------
test_classes = np.argmax(test_preds, axis=1)

submission = pd.read_csv(SUBMISSION_PATH)
submission["Irrigation_Need"] = [label_map[c] for c in test_classes]
submission.to_csv(OUTPUT_PATH, index=False)
print(f"\nSubmission saved to {OUTPUT_PATH}")
