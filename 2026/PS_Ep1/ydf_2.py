import pandas as pd
import numpy as np

from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error

from ydf import GradientBoostedTreesLearner
import ydf

import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("train.csv", index_col="id")
test = pd.read_csv("test.csv", index_col="id")
original = pd.read_csv("Exam_Score_Prediction.csv", index_col="student_id")

df["generated"] = 1
test["generated"] = 1
original["generated"] = 0

skf = KFold(n_splits=5, shuffle=True, random_state=42)
scores, oof_preds_df, test_preds_df = [], [], []
for i, (train_index, test_index) in enumerate(
    skf.split(df, df["exam_score"])
):

    X_train, X_val = df.iloc[train_index], df.iloc[test_index]
    X_train = pd.concat([X_train, original], axis=0).reset_index(drop=True)

    ydf_md = GradientBoostedTreesLearner(
        label="exam_score",
        task=ydf.Task.REGRESSION,
        num_threads=10,
        num_trees=1000
    ).train(X_train)
    ydf_pred = ydf_md.predict(X_val)

    oof_preds = pd.DataFrame()
    oof_preds["id"] = X_val.index
    oof_preds["ydf_pred_2"] = ydf_pred
    oof_preds["y"] = X_val["exam_score"].values
    oof_preds_df.append(oof_preds)

    test_pred = ydf_md.predict(test)
    test_preds = pd.DataFrame()
    test_preds["ydf_pred_2"] = test_pred
    test_preds["fold"] = i
    test_preds_df.append(test_preds)

    score = root_mean_squared_error(X_val['exam_score'], ydf_pred)
    print('Fold:', i, 'RMSE:', score)
    scores.append(score)

print(f"Mean RMSE: {np.mean(scores)}")

# average the per-fold test predictions properly
preds_per_fold = [df["ydf_pred_2"].values for df in test_preds_df]
final_test_pred = np.mean(preds_per_fold, axis=0)
# use averaged fold predictions for submission
# (or retrain on full data if desired)
preds = final_test_pred

submission = pd.read_csv("sample_submission.csv")
submission["exam_score"] = preds
submission.to_csv("ydf_sub_2.csv", index=False)

print("Saving OOF predictions...")
oof_preds_df = pd.concat(oof_preds_df)
oof_preds_df.to_csv("oof_ydf_2.csv", index=False)

print("Saving test predictions...")
test_preds_df = pd.concat(test_preds_df)
test_preds_df.to_csv("test_ydf_2.csv", index=False)
print("Done!")

# Fold: 0 RMSE: 8.688687450030669
# Fold: 1 RMSE: 8.693370769942865
# Fold: 2 RMSE: 8.690121569901875
# Fold: 3 RMSE: 8.700055049023506
# Fold: 4 RMSE: 8.721888191956445
# Mean RMSE: 8.69882460617107
