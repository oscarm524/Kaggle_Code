import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.base import is_regressor


class StackingClassifierEvaluator:
    def __init__(self, models, X, y, cv, stacker=None):
        """
        models: list of sklearn-like classifiers
        X: features (numpy array or pandas DataFrame)
        y: target (numpy array or pandas Series)
        cv: cross-validation strategy (e.g., StratifiedKFold)
        stacker: meta-classifier (default: LogisticRegression)
        """
        self.models = models
        self.X = X
        self.y = y
        self.cv = cv
        self.stacker = stacker if stacker is not None else LogisticRegression(
            solver='lbfgs', max_iter=1000
        )
        self.base_scores = {}
        self.stacker_score = None
        self.stacker_model = None

    def fit(self):
        # Out-of-fold predictions for each base model
        oof_preds = []
        for i, model in enumerate(self.models):
            preds = cross_val_predict(
                model, self.X, self.y, cv=self.cv, method='predict_proba'
            )[:, 1]
            score = roc_auc_score(self.y, preds)
            self.base_scores[f'model_{i}'] = score
            oof_preds.append(preds.reshape(-1, 1))
        # Stack out-of-fold predictions as features for the stacker
        stack_X = np.hstack(oof_preds)
        # Train stacker on out-of-fold predictions
        self.stacker_model = self.stacker.fit(stack_X, self.y)
        # Check if stacker is a regressor
        if is_regressor(self.stacker):
            stacker_preds = self.stacker_model.predict(stack_X)
        else:
            stacker_preds = self.stacker_model.predict_proba(stack_X)[:, 1]
        self.stacker_score = roc_auc_score(self.y, stacker_preds)

    def report(self):
        print("Base model ROC-AUC scores:")
        for name, score in self.base_scores.items():
            print(f"  {name}: {score:.4f}")
        print(f"Stacker ROC-AUC score: {self.stacker_score:.4f}")

    def predict(self, X):
        # Predict with each base model
        base_preds = [
            model.predict_proba(X)[:, 1].reshape(-1, 1)
            for model in self.models
        ]
        stack_X = np.hstack(base_preds)
        # Check if stacker is a regressor
        if is_regressor(self.stacker):
            return self.stacker_model.predict(stack_X)
        else:
            return self.stacker_model.predict_proba(stack_X)[:, 1]
