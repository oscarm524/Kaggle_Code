import pandas as pd
import gc

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.linear_model import Ridge
# from sklearn.ensemble import (
#     HistGradientBoostingClassifier,
#     GradientBoostingClassifier,
# )
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier


def main():
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    df = pd.read_csv("train.csv", index_col="id")
    X = df.drop(columns=["Heart Disease"], axis=1)
    y = df["Heart Disease"].map({"Absence": 0, "Presence": 1})

    test = pd.read_csv("test.csv", index_col="id")

    del df
    gc.collect()

    lgbm_params = {
        'n_estimators': 998,
        'learning_rate': 0.11239530686383664,
        'num_leaves': 35,
        'max_depth': 3,
        'min_child_samples': 50,
        'subsample': 0.7094235914815041,
        'colsample_bytree': 0.9939521128432284,
        'reg_alpha': 0.9844372637384456,
        'reg_lambda': 0.0022962114949080514,
        'verbose': -1,
        'random_state': 42
    }
    LGBM_md = LGBMClassifier(**lgbm_params)

    xgb_params = {
        'n_estimators': 943,
        'learning_rate': 0.09717789596330055,
        'max_depth': 3,
        'subsample': 0.8312966962670887,
        'colsample_bytree': 0.6067405874013712,
        'gamma': 0.4597939772867661,
        'min_child_weight': 6,
        'random_state': 42
    }
    XGB_md = XGBClassifier(**xgb_params)

    cat_params = {
        'max_depth': 4,
        'verbose': 0
    }
    Cat_md = CatBoostClassifier(**cat_params)

    oof_preds = pd.DataFrame()
    oof_preds["md_1"] = cross_val_predict(
        LGBM_md, X, y, cv=skf, method="predict_proba"
    )[:, 1]
    oof_preds["md_2"] = cross_val_predict(
        XGB_md, X, y, cv=skf, method="predict_proba"
    )[:, 1]
    oof_preds["md_3"] = cross_val_predict(
        Cat_md, X, y, cv=skf, method="predict_proba"
    )[:, 1]

    oof_preds_test = pd.DataFrame()
    oof_preds_test["md_1"] = LGBM_md.fit(X, y).predict_proba(test)[:, 1]
    oof_preds_test["md_2"] = XGB_md.fit(X, y).predict_proba(test)[:, 1]
    oof_preds_test["md_3"] = Cat_md.fit(X, y).predict_proba(test)[:, 1]

    original = pd.read_csv("Heart_Disease_Prediction.csv")
    original["Heart Disease"] = original["Heart Disease"].map({
        "Absence": 0, "Presence": 1
    })

    potential_cat = [
        "Chest pain type", "EKG results", "Slope of ST",
        "Number of vessels fluro", "Thallium"
    ]

    for col in potential_cat:
        mean_col = pd.DataFrame(
            original.groupby(col)["Heart Disease"].mean()
        ).reset_index()
        mean_col = mean_col.rename(columns={"Heart Disease": f"{col}_mean"})
        X = X.merge(mean_col, on=col, how="left")
        test = test.merge(mean_col, on=col, how="left")

    lgbm_params = {
        'n_estimators': 783,
        'learning_rate': 0.11232301086863652,
        'num_leaves': 40,
        'max_depth': 3,
        'min_child_samples': 64,
        'subsample': 0.9346504089353533,
        'colsample_bytree': 0.6144043725950896,
        'reg_alpha': 0.2693234207150637,
        'reg_lambda': 0.3303738446455776,
        'device': 'gpu',
        'random_state': 42,
        'verbose': -1,
        'n_jobs': -1,
    }
    LGBM_md = LGBMClassifier(**lgbm_params)

    xgb_params = {
        "n_estimators": 832,
        "max_depth": 7,
        "learning_rate": 0.013432781360505153,
        "subsample": 0.7711052834966381,
        "random_state": 42,
    }
    XGB_md = XGBClassifier(**xgb_params)

    cat_params = {
        'max_depth': 2,
        'verbose': 0
    }
    Cat_md = CatBoostClassifier(**cat_params)

    oof_preds["md_4"] = cross_val_predict(
        LGBM_md, X, y, cv=skf, method="predict_proba"
    )[:, 1]
    oof_preds["md_5"] = cross_val_predict(
        XGB_md, X, y, cv=skf, method="predict_proba"
    )[:, 1]
    oof_preds["md_6"] = cross_val_predict(
        Cat_md, X, y, cv=skf, method="predict_proba"
    )[:, 1]
    oof_preds["y"] = y.values

    oof_preds_test["md_4"] = LGBM_md.fit(X, y).predict_proba(test)[:, 1]
    oof_preds_test["md_5"] = XGB_md.fit(X, y).predict_proba(test)[:, 1]
    oof_preds_test["md_6"] = Cat_md.fit(X, y).predict_proba(test)[:, 1]

    X_stack = oof_preds.drop(columns=["y"], axis=1)
    y_stack = oof_preds["y"]
    ridge_preds = cross_val_predict(
        Ridge(), X_stack, y_stack, cv=skf, method="predict"
    )

    print("Ridge ROC-AUC:", roc_auc_score(y_stack, ridge_preds))
    # Ridge ROC-AUC: 0.9555781303481308

    # Predicting on the test set
    ridge_test_preds = Ridge().fit(X_stack, y_stack).predict(oof_preds_test)

    submission = pd.read_csv("sample_submission.csv", index_col="id")
    submission["Heart Disease"] = ridge_test_preds
    submission.to_csv("stacking_sub_12.csv", index=True)


if __name__ == "__main__":
    main()