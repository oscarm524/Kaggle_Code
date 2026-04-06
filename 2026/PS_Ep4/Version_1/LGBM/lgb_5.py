"""LightGBM model with best hyperparameters from fifth Optuna HPO.

Trains a 5-fold stratified cross-validation LightGBM model using the
best hyperparameter combination found during the fifth tuning run
(with balanced sample weights, a logistic regression meta-feature,
max_bin=2000, and extended boosting rounds).  Saves out-of-fold
predicted probabilities (likelihoods) and a test submission to CSV.
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
from lightgbm import Dataset
import lightgbm as lgb

# -- Paths -------------------------------------------------------------------
BASE_DIR = "/media/oscarm524/Expansion/Kaggle/2026/PS_Ep4/Version_1"
TRAIN_PATH = f"{BASE_DIR}/train.csv"
TEST_PATH = f"{BASE_DIR}/test.csv"
SUBMISSION_PATH = f"{BASE_DIR}/sample_submission.csv"
OUTPUT_PATH = f"{BASE_DIR}/Models/LGBM/lgb_5.csv"
OOF_PATH = f"{BASE_DIR}/Models/LGBM/lgb_5_oof.csv"

# -- Load Data ---------------------------------------------------------------
train = pd.read_csv(TRAIN_PATH, index_col="id")
test = pd.read_csv(TEST_PATH, index_col="id")

# -- Prepare features --------------------------------------------------------
X = train.drop(columns=["Irrigation_Need"])
y = train["Irrigation_Need"].map({"Low": 0, "Medium": 1, "High": 2})

X_test = test.copy()

# Convert object columns to category
cat_cols = X.select_dtypes(include="object").columns.tolist()
for col in cat_cols:
    X[col] = X[col].astype("category")
    X_test[col] = X_test[col].astype("category")

# -- Best Hyper-parameters from Optuna ---------------------------------------
param = {
    # "device": "gpu",
    "objective": "multiclass",
    "num_class": 3,
    "metric": "multi_logloss",
    "verbosity": -1,
    "max_depth": 2,
    "num_leaves": 24,
    "learning_rate": 0.07569094658099462,
    "subsample": 0.808689869335482,
    "colsample_bytree": 0.9288709861273847,
    "min_child_samples": 31,
    "reg_alpha": 3.920993953631824e-08,
    "reg_lambda": 8.112880182665321e-07,
    "max_bin": 1000,
    "random_state": 42,
}

# -- 5-Fold CV Training & Test Prediction ------------------------------------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_preds = np.zeros(len(X))
oof_probs = np.zeros((len(X), 3))
test_preds = np.zeros((len(X_test), 3))
fold_scores = []

for fold, (train_index, valid_index) in enumerate(skf.split(X, y)):
    print(f"Fold {fold + 1} / 5")

    X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
    y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

    w_train = compute_sample_weight("balanced", y_train)
    w_valid = compute_sample_weight("balanced", y_valid)

    dtrain = Dataset(
        X_train, label=y_train, weight=w_train,
        categorical_feature=cat_cols,
    )
    dvalid = Dataset(
        X_valid, label=y_valid, weight=w_valid,
        reference=dtrain, categorical_feature=cat_cols,
    )

    model = lgb.train(
        param,
        dtrain,
        num_boost_round=5000,
        valid_sets=[dvalid],
        callbacks=[
            lgb.early_stopping(stopping_rounds=200, verbose=True),
            lgb.log_evaluation(period=100),
        ],
    )

    # OOF predictions
    oof_probs[valid_index] = model.predict(X_valid)
    oof_preds[valid_index] = oof_probs[valid_index].argmax(axis=1)
    fold_score = balanced_accuracy_score(
        y_valid, oof_preds[valid_index]
    )
    fold_scores.append(fold_score)
    print(f"  Fold {fold + 1} Balanced Accuracy: {fold_score:.6f}\n")

    # Test predictions (average probabilities across folds)
    test_preds += model.predict(X_test) / 5

print(f"Overall OOF Balanced Accuracy: {np.mean(fold_scores):.6f}")

# -- Save OOF Predictions ---------------------------------------------------
oof_df = pd.DataFrame({
    "id": X.index,
    "prob_Low": oof_probs[:, 0],
    "prob_Medium": oof_probs[:, 1],
    "prob_High": oof_probs[:, 2],
})
oof_df.to_csv(OOF_PATH, index=False)
print(f"OOF predictions saved to {OOF_PATH}")

# -- Generate Submission -----------------------------------------------------
label_map = {0: "Low", 1: "Medium", 2: "High"}
test_classes = np.argmax(test_preds, axis=1)

submission = pd.read_csv(SUBMISSION_PATH)
submission["Irrigation_Need"] = [label_map[c] for c in test_classes]
submission.to_csv(OUTPUT_PATH, index=False)
print(f"\nSubmission saved to {OUTPUT_PATH}")

# Fold 1 / 5
# Training until validation scores don't improve for 200 rounds
# [100]   valid_0's multi_logloss: 0.219986
# [200]   valid_0's multi_logloss: 0.148143
# [300]   valid_0's multi_logloss: 0.123221
# [400]   valid_0's multi_logloss: 0.110509
# [500]   valid_0's multi_logloss: 0.102937
# [600]   valid_0's multi_logloss: 0.0979018
# [700]   valid_0's multi_logloss: 0.094526
# [800]   valid_0's multi_logloss: 0.0921823
# [900]   valid_0's multi_logloss: 0.0904281
# [1000]  valid_0's multi_logloss: 0.089079
# [1100]  valid_0's multi_logloss: 0.0879632
# [1200]  valid_0's multi_logloss: 0.0869695
# [1300]  valid_0's multi_logloss: 0.0862244
# [1400]  valid_0's multi_logloss: 0.0855059
# [1500]  valid_0's multi_logloss: 0.0848616
# [1600]  valid_0's multi_logloss: 0.0842469
# [1700]  valid_0's multi_logloss: 0.0837037
# [1800]  valid_0's multi_logloss: 0.0833128
# [1900]  valid_0's multi_logloss: 0.0829645
# [2000]  valid_0's multi_logloss: 0.0824062
# [2100]  valid_0's multi_logloss: 0.0820644
# [2200]  valid_0's multi_logloss: 0.0817916
# [2300]  valid_0's multi_logloss: 0.0815674
# [2400]  valid_0's multi_logloss: 0.0813943
# [2500]  valid_0's multi_logloss: 0.081172
# [2600]  valid_0's multi_logloss: 0.0810533
# [2700]  valid_0's multi_logloss: 0.0808675
# [2800]  valid_0's multi_logloss: 0.0807754
# [2900]  valid_0's multi_logloss: 0.0806804
# [3000]  valid_0's multi_logloss: 0.0806562
# [3100]  valid_0's multi_logloss: 0.080523
# [3200]  valid_0's multi_logloss: 0.0804612
# [3300]  valid_0's multi_logloss: 0.0804089
# [3400]  valid_0's multi_logloss: 0.0803198
# [3500]  valid_0's multi_logloss: 0.0801785
# [3600]  valid_0's multi_logloss: 0.0801395
# [3700]  valid_0's multi_logloss: 0.0801084
# [3800]  valid_0's multi_logloss: 0.0800493
# [3900]  valid_0's multi_logloss: 0.0800004
# [4000]  valid_0's multi_logloss: 0.0798514
# [4100]  valid_0's multi_logloss: 0.0797814
# [4200]  valid_0's multi_logloss: 0.0798011
# [4300]  valid_0's multi_logloss: 0.0798032
# [4400]  valid_0's multi_logloss: 0.0797918
# Early stopping, best iteration is:
# [4239]  valid_0's multi_logloss: 0.0797585
#   Fold 1 Balanced Accuracy: 0.973091

# Fold 2 / 5
# Training until validation scores don't improve for 200 rounds
# [100]   valid_0's multi_logloss: 0.217748
# [200]   valid_0's multi_logloss: 0.146466
# [300]   valid_0's multi_logloss: 0.121709
# [400]   valid_0's multi_logloss: 0.108618
# [500]   valid_0's multi_logloss: 0.101062
# [600]   valid_0's multi_logloss: 0.0960879
# [700]   valid_0's multi_logloss: 0.092633
# [800]   valid_0's multi_logloss: 0.0900723
# [900]   valid_0's multi_logloss: 0.0882661
# [1000]  valid_0's multi_logloss: 0.0868427
# [1100]  valid_0's multi_logloss: 0.0858829
# [1200]  valid_0's multi_logloss: 0.0850029
# [1300]  valid_0's multi_logloss: 0.0843052
# [1400]  valid_0's multi_logloss: 0.0837704
# [1500]  valid_0's multi_logloss: 0.0832079
# [1600]  valid_0's multi_logloss: 0.0827128
# [1700]  valid_0's multi_logloss: 0.0823051
# [1800]  valid_0's multi_logloss: 0.0820525
# [1900]  valid_0's multi_logloss: 0.081805
# [2000]  valid_0's multi_logloss: 0.0816083
# [2100]  valid_0's multi_logloss: 0.0812216
# [2200]  valid_0's multi_logloss: 0.0809908
# [2300]  valid_0's multi_logloss: 0.0808856
# [2400]  valid_0's multi_logloss: 0.0807451
# [2500]  valid_0's multi_logloss: 0.0804379
# [2600]  valid_0's multi_logloss: 0.0802097
# [2700]  valid_0's multi_logloss: 0.0801423
# [2800]  valid_0's multi_logloss: 0.0799963
# [2900]  valid_0's multi_logloss: 0.0798117
# [3000]  valid_0's multi_logloss: 0.0796983
# [3100]  valid_0's multi_logloss: 0.0797099
# [3200]  valid_0's multi_logloss: 0.0796583
# [3300]  valid_0's multi_logloss: 0.0796876
# [3400]  valid_0's multi_logloss: 0.0796831
# [3500]  valid_0's multi_logloss: 0.0797035
# Early stopping, best iteration is:
# [3335]  valid_0's multi_logloss: 0.0796522
#   Fold 2 Balanced Accuracy: 0.974602

# Fold 3 / 5
# Training until validation scores don't improve for 200 rounds
# [100]   valid_0's multi_logloss: 0.219744
# [200]   valid_0's multi_logloss: 0.14643
# [300]   valid_0's multi_logloss: 0.121809
# [400]   valid_0's multi_logloss: 0.108977
# [500]   valid_0's multi_logloss: 0.101514
# [600]   valid_0's multi_logloss: 0.0964793
# [700]   valid_0's multi_logloss: 0.0930639
# [800]   valid_0's multi_logloss: 0.0904848
# [900]   valid_0's multi_logloss: 0.0885376
# [1000]  valid_0's multi_logloss: 0.0871581
# [1100]  valid_0's multi_logloss: 0.0859971
# [1200]  valid_0's multi_logloss: 0.0851208
# [1300]  valid_0's multi_logloss: 0.0842099
# [1400]  valid_0's multi_logloss: 0.0835217
# [1500]  valid_0's multi_logloss: 0.0828817
# [1600]  valid_0's multi_logloss: 0.0823422
# [1700]  valid_0's multi_logloss: 0.0818915
# [1800]  valid_0's multi_logloss: 0.0815228
# [1900]  valid_0's multi_logloss: 0.0812785
# [2000]  valid_0's multi_logloss: 0.0809631
# [2100]  valid_0's multi_logloss: 0.0806518
# [2200]  valid_0's multi_logloss: 0.0803808
# [2300]  valid_0's multi_logloss: 0.0801319
# [2400]  valid_0's multi_logloss: 0.0798816
# [2500]  valid_0's multi_logloss: 0.0796429
# [2600]  valid_0's multi_logloss: 0.079499
# [2700]  valid_0's multi_logloss: 0.0793249
# [2800]  valid_0's multi_logloss: 0.0791543
# [2900]  valid_0's multi_logloss: 0.0790458
# [3000]  valid_0's multi_logloss: 0.0789731
# [3100]  valid_0's multi_logloss: 0.0788089
# [3200]  valid_0's multi_logloss: 0.0786336
# [3300]  valid_0's multi_logloss: 0.0785062
# [3400]  valid_0's multi_logloss: 0.078484
# [3500]  valid_0's multi_logloss: 0.0783453
# [3600]  valid_0's multi_logloss: 0.0783764
# [3700]  valid_0's multi_logloss: 0.0782726
# [3800]  valid_0's multi_logloss: 0.0782813
# [3900]  valid_0's multi_logloss: 0.078152
# [4000]  valid_0's multi_logloss: 0.0780352
# [4100]  valid_0's multi_logloss: 0.0779738
# [4200]  valid_0's multi_logloss: 0.0780016
# Early stopping, best iteration is:
# [4081]  valid_0's multi_logloss: 0.0779569
#   Fold 3 Balanced Accuracy: 0.974577

# Fold 4 / 5
# Training until validation scores don't improve for 200 rounds
# [100]   valid_0's multi_logloss: 0.217277
# [200]   valid_0's multi_logloss: 0.14684
# [300]   valid_0's multi_logloss: 0.121372
# [400]   valid_0's multi_logloss: 0.108135
# [500]   valid_0's multi_logloss: 0.100308
# [600]   valid_0's multi_logloss: 0.0952865
# [700]   valid_0's multi_logloss: 0.0918137
# [800]   valid_0's multi_logloss: 0.0893088
# [900]   valid_0's multi_logloss: 0.0873638
# [1000]  valid_0's multi_logloss: 0.0857644
# [1100]  valid_0's multi_logloss: 0.084676
# [1200]  valid_0's multi_logloss: 0.0837763
# [1300]  valid_0's multi_logloss: 0.0829566
# [1400]  valid_0's multi_logloss: 0.0823532
# [1500]  valid_0's multi_logloss: 0.0815187
# [1600]  valid_0's multi_logloss: 0.0811195
# [1700]  valid_0's multi_logloss: 0.0807285
# [1800]  valid_0's multi_logloss: 0.0803753
# [1900]  valid_0's multi_logloss: 0.0800508
# [2000]  valid_0's multi_logloss: 0.0797915
# [2100]  valid_0's multi_logloss: 0.0795657
# [2200]  valid_0's multi_logloss: 0.0793035
# [2300]  valid_0's multi_logloss: 0.0790802
# [2400]  valid_0's multi_logloss: 0.0788712
# [2500]  valid_0's multi_logloss: 0.0786777
# [2600]  valid_0's multi_logloss: 0.0784979
# [2700]  valid_0's multi_logloss: 0.0783347
# [2800]  valid_0's multi_logloss: 0.0782354
# [2900]  valid_0's multi_logloss: 0.0781407
# [3000]  valid_0's multi_logloss: 0.0780311
# [3100]  valid_0's multi_logloss: 0.0779588
# [3200]  valid_0's multi_logloss: 0.0778709
# [3300]  valid_0's multi_logloss: 0.0778236
# [3400]  valid_0's multi_logloss: 0.0777403
# [3500]  valid_0's multi_logloss: 0.0776718
# [3600]  valid_0's multi_logloss: 0.0775724
# [3700]  valid_0's multi_logloss: 0.0775449
# [3800]  valid_0's multi_logloss: 0.0775271
# [3900]  valid_0's multi_logloss: 0.07742
# [4000]  valid_0's multi_logloss: 0.0774196
# [4100]  valid_0's multi_logloss: 0.0774322
# Early stopping, best iteration is:
# [3913]  valid_0's multi_logloss: 0.0774029
#   Fold 4 Balanced Accuracy: 0.974135

# Fold 5 / 5
# Training until validation scores don't improve for 200 rounds
# [100]   valid_0's multi_logloss: 0.218074
# [200]   valid_0's multi_logloss: 0.145008
# [300]   valid_0's multi_logloss: 0.120608
# [400]   valid_0's multi_logloss: 0.107692
# [500]   valid_0's multi_logloss: 0.100125
# [600]   valid_0's multi_logloss: 0.0950559
# [700]   valid_0's multi_logloss: 0.0915692
# [800]   valid_0's multi_logloss: 0.089231
# [900]   valid_0's multi_logloss: 0.0873266
# [1000]  valid_0's multi_logloss: 0.085943
# [1100]  valid_0's multi_logloss: 0.0846986
# [1200]  valid_0's multi_logloss: 0.0837952
# [1300]  valid_0's multi_logloss: 0.0829864
# [1400]  valid_0's multi_logloss: 0.0821934
# [1500]  valid_0's multi_logloss: 0.081472
# [1600]  valid_0's multi_logloss: 0.0809911
# [1700]  valid_0's multi_logloss: 0.0803934
# [1800]  valid_0's multi_logloss: 0.0799182
# [1900]  valid_0's multi_logloss: 0.0796158
# [2000]  valid_0's multi_logloss: 0.0792892
# [2100]  valid_0's multi_logloss: 0.0789933
# [2200]  valid_0's multi_logloss: 0.078782
# [2300]  valid_0's multi_logloss: 0.0784964
# [2400]  valid_0's multi_logloss: 0.0782775
# [2500]  valid_0's multi_logloss: 0.0781242
# [2600]  valid_0's multi_logloss: 0.0778888
# [2700]  valid_0's multi_logloss: 0.0776303
# [2800]  valid_0's multi_logloss: 0.0774085
# [2900]  valid_0's multi_logloss: 0.0772126
# [3000]  valid_0's multi_logloss: 0.0769884
# [3100]  valid_0's multi_logloss: 0.0768764
# [3200]  valid_0's multi_logloss: 0.0765993
# [3300]  valid_0's multi_logloss: 0.0765392
# [3400]  valid_0's multi_logloss: 0.0764004
# [3500]  valid_0's multi_logloss: 0.0762836
# [3600]  valid_0's multi_logloss: 0.0761938
# [3700]  valid_0's multi_logloss: 0.0761146
# [3800]  valid_0's multi_logloss: 0.0760058
# [3900]  valid_0's multi_logloss: 0.075913
# [4000]  valid_0's multi_logloss: 0.0758401
# [4100]  valid_0's multi_logloss: 0.0758196
# [4200]  valid_0's multi_logloss: 0.0757897
# [4300]  valid_0's multi_logloss: 0.0757551
# [4400]  valid_0's multi_logloss: 0.0757523
# [4500]  valid_0's multi_logloss: 0.0757364
# [4600]  valid_0's multi_logloss: 0.0756739
# [4700]  valid_0's multi_logloss: 0.0755606
# [4800]  valid_0's multi_logloss: 0.0755028
# [4900]  valid_0's multi_logloss: 0.0755556
# Early stopping, best iteration is:
# [4769]  valid_0's multi_logloss: 0.0754808
#   Fold 5 Balanced Accuracy: 0.974612

# Overall OOF Balanced Accuracy: 0.974203