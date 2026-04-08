"""CatBoost experiment 10 -- Lossguide grow policy + all features.

Uses grow_policy='Lossguide' (like LightGBM leaf-wise growth)
with increased num_leaves and the full feature set including
interactions, ratios, and decimal digits.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from catboost import CatBoostClassifier, Pool

BASE_DIR = "/media/oscarm524/Expansion/Kaggle/2026/PS_Ep4/Version_1"
TRAIN_PATH = f"{BASE_DIR}/train.csv"
TEST_PATH = f"{BASE_DIR}/test.csv"
SUBMISSION_PATH = f"{BASE_DIR}/sample_submission.csv"
SAVE_DIR = "/media/oscarm524/Expansion/Kaggle/2026/PS_Ep4/Version_2/CatBoost"
OUTPUT_PATH = f"{SAVE_DIR}/cat_10.csv"
OOF_PATH = f"{SAVE_DIR}/cat_10_oof.csv"
TEST_PROBS_PATH = f"{SAVE_DIR}/cat_10_test.csv"

NUMERIC_COLS = [
    "Soil_Moisture", "Temperature_C", "Rainfall_mm",
    "Wind_Speed_kmh", "Humidity", "Soil_pH",
    "Organic_Carbon", "Electrical_Conductivity",
    "Sunlight_Hours", "Field_Area_hectare",
    "Previous_Irrigation_mm",
]


def add_decimal_digits(df):
    for col in NUMERIC_COLS:
        v = df[col].values
        df[f"{col}_dec"] = np.floor((v - np.floor(v)) * 10).astype(int)
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
    return df


def add_ratios(df):
    eps = 1e-6
    df["rain_per_hectare"] = df["Rainfall_mm"] / (df["Field_Area_hectare"] + eps)
    df["moisture_over_humidity"] = df["Soil_Moisture"] / (df["Humidity"] + eps)
    df["temp_over_wind"] = df["Temperature_C"] / (df["Wind_Speed_kmh"] + eps)
    df["moisture_over_temp"] = df["Soil_Moisture"] / (df["Temperature_C"] + eps)
    return df


train = pd.read_csv(TRAIN_PATH, index_col="id")
test = pd.read_csv(TEST_PATH, index_col="id")

X = train.drop(columns=["Irrigation_Need"])
y = train["Irrigation_Need"].map({"Low": 0, "Medium": 1, "High": 2})
X_test = test.copy()

for func in [add_decimal_digits, add_binary_flags, add_interactions, add_ratios]:
    X = func(X)
    X_test = func(X_test)

cat_cols = X.select_dtypes(include="object").columns.tolist()
label_map = {0: "Low", 1: "Medium", 2: "High"}

param = {
    "loss_function": "MultiClass",
    "classes_count": 3,
    "grow_policy": "Lossguide",
    "max_leaves": 64,
    "depth": 8,
    "learning_rate": 0.08,
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
