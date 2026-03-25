import numpy as np
from sklearn.metrics import roc_auc_score


def _hill_climbing_weights(X_subset, y, max_iter=1000, step=0.01):
    """Optimize weights for a given subset of models using hill climbing.

    Args:
        X_subset (np.ndarray): 2D array of model predictions with shape
            (n_samples, n_models).
        y (np.ndarray): True binary labels.
        max_iter (int): Maximum number of hill climbing iterations.
        step (float): Step size for weight adjustments.

    Returns:
        tuple: (weights, best_score) where weights is a 1D np.ndarray of
            optimized model weights and best_score is the corresponding
            ROC-AUC score.
    """
    n_models = X_subset.shape[1]
    weights = np.ones(n_models) / n_models
    best_score = roc_auc_score(
        y, np.average(X_subset, axis=1, weights=weights)
    )
    improved = True
    iter_count = 0

    while improved and iter_count < max_iter:
        improved = False
        for i in range(n_models):
            for delta in [-step, step]:
                new_weights = weights.copy()
                new_weights[i] += delta
                if np.any(new_weights < 0):
                    continue
                new_weights /= new_weights.sum()
                score = roc_auc_score(
                    y, np.average(X_subset, axis=1, weights=new_weights)
                )
                if score > best_score:
                    weights = new_weights
                    best_score = score
                    improved = True
        iter_count += 1

    return weights, best_score


def hill_climbing_ensemble(X_oof, y, X_test, max_iter=1000, step=0.01):
    """Build an ensemble via greedy forward model selection with hill climbing.

    Ranks individual models by ROC-AUC, starts with the top 2, then
    iteratively adds the next best model as long as it improves the ensemble
    score. Weights are optimised at each step using hill climbing.

    Args:
        X_oof (pd.DataFrame): Out-of-fold predictions, one column per model.
        y (pd.Series or np.ndarray): True binary labels aligned with X_oof.
        X_test (pd.DataFrame): Test predictions with the same columns as
            X_oof. Used to generate the final ensemble prediction.
        max_iter (int): Maximum hill climbing iterations per candidate set.
        step (float): Step size for weight adjustments during hill climbing.

    Returns:
        tuple: (ensemble_preds, best_global_score, best_global_models,
            best_global_weights) where ensemble_preds is a 1D np.ndarray of
            final test predictions, best_global_score is the best ROC-AUC
            achieved on OOF data, best_global_models is the list of selected
            model names, and best_global_weights is a 1D np.ndarray of their
            corresponding weights.
    """
    y_arr = np.asarray(y)

    # Step 1: Compute individual ROC-AUC for each model and rank them
    individual_scores = {
        col: roc_auc_score(y_arr, X_oof[col]) for col in X_oof.columns
    }

    ranked_models = sorted(
        individual_scores, key=individual_scores.get, reverse=True
    )

    print("Individual ROC-AUC scores (ranked best to worst):")
    for i, model_name in enumerate(ranked_models):
        print(f"  {i + 1}. {model_name}: {individual_scores[model_name]:.6f}")

    # Step 2: Start with the top 2 models
    print("\n--- Hill Climbing Ensemble (greedy forward addition) ---")
    best_global_score = 0
    best_global_models = []
    best_global_weights = None

    selected_models = [ranked_models[0], ranked_models[1]]
    weights, score = _hill_climbing_weights(
        X_oof[selected_models].values, y_arr, max_iter=max_iter, step=step
    )
    best_global_score = score
    best_global_models = selected_models.copy()
    best_global_weights = weights.copy()

    print(f"\nModels: {selected_models}")
    weights_map = dict(zip(selected_models, weights))
    print(f"  Score: {score:.6f} | Weights: {weights_map}")

    # Step 3: Iteratively add remaining models
    for k in range(2, len(ranked_models)):
        candidate = ranked_models[k]
        candidate_models = selected_models + [candidate]
        weights, score = _hill_climbing_weights(
            X_oof[candidate_models].values, y_arr, max_iter=max_iter, step=step
        )

        print(f"\nAdding '{candidate}' -> Models: {candidate_models}")
        weights_map = dict(zip(candidate_models, weights))
        print(f"  Score: {score:.6f} | Weights: {weights_map}")

        if score > best_global_score:
            best_global_score = score
            best_global_models = candidate_models.copy()
            best_global_weights = weights.copy()
            selected_models = candidate_models
        else:
            print("  No improvement. Stopping early.")
            break

    print("\n=== Best Ensemble ===")
    print(f"Score: {best_global_score:.6f}")
    print(f"Models: {best_global_models}")
    print(f"Weights: {dict(zip(best_global_models, best_global_weights))}")

    # Generate final test predictions using the best ensemble
    ensemble_preds = np.average(
        X_test[best_global_models].values,
        axis=1,
        weights=best_global_weights,
    )

    return (
        ensemble_preds,
        best_global_score,
        best_global_models,
        best_global_weights,
    )


def hill_climbing_ensemble_v2(X_oof, y, X_test, max_iter=1000, step=0.01):
    """Build an ensemble via exhaustive forward model selection with hill
    climbing.

    Like ``hill_climbing_ensemble`` but, instead of stopping at the first
    model that does not improve the score, it *skips* that model and
    continues evaluating all remaining candidates. Every model that yields a
    net improvement when added to the current ensemble is included.

    Args:
        X_oof (pd.DataFrame): Out-of-fold predictions, one column per model.
        y (pd.Series or np.ndarray): True binary labels aligned with X_oof.
        X_test (pd.DataFrame): Test predictions with the same columns as
            X_oof. Used to generate the final ensemble prediction.
        max_iter (int): Maximum hill climbing iterations per candidate set.
        step (float): Step size for weight adjustments during hill climbing.

    Returns:
        tuple: (ensemble_preds, best_global_score, best_global_models,
            best_global_weights) where ensemble_preds is a 1D np.ndarray of
            final test predictions, best_global_score is the best ROC-AUC
            achieved on OOF data, best_global_models is the list of selected
            model names, and best_global_weights is a 1D np.ndarray of their
            corresponding weights.
    """
    y_arr = np.asarray(y)

    # Step 1: Compute individual ROC-AUC for each model and rank them
    individual_scores = {
        col: roc_auc_score(y_arr, X_oof[col]) for col in X_oof.columns
    }

    ranked_models = sorted(
        individual_scores, key=individual_scores.get, reverse=True
    )

    print("Individual ROC-AUC scores (ranked best to worst):")
    for i, model_name in enumerate(ranked_models):
        print(f"  {i + 1}. {model_name}: {individual_scores[model_name]:.6f}")

    # Step 2: Start with the top 2 models
    print("\n--- Hill Climbing Ensemble V2 (skip non-improving models) ---")
    best_global_score = 0
    best_global_models = []
    best_global_weights = None

    selected_models = [ranked_models[0], ranked_models[1]]
    weights, score = _hill_climbing_weights(
        X_oof[selected_models].values, y_arr, max_iter=max_iter, step=step
    )
    best_global_score = score
    best_global_models = selected_models.copy()
    best_global_weights = weights.copy()

    print(f"\nModels: {selected_models}")
    weights_map = dict(zip(selected_models, weights))
    print(f"  Score: {score:.6f} | Weights: {weights_map}")

    # Step 3: Iterate over all remaining models, skipping non-improving ones
    for k in range(2, len(ranked_models)):
        candidate = ranked_models[k]
        candidate_models = selected_models + [candidate]
        weights, score = _hill_climbing_weights(
            X_oof[candidate_models].values,
            y_arr,
            max_iter=max_iter,
            step=step,
        )

        print(f"\nAdding '{candidate}' -> Models: {candidate_models}")
        weights_map = dict(zip(candidate_models, weights))
        print(f"  Score: {score:.6f} | Weights: {weights_map}")

        if score > best_global_score:
            best_global_score = score
            best_global_models = candidate_models.copy()
            best_global_weights = weights.copy()
            selected_models = candidate_models
        else:
            print("  No improvement. Skipping model.")

    print("\n=== Best Ensemble ===")
    print(f"Score: {best_global_score:.6f}")
    print(f"Models: {best_global_models}")
    print(f"Weights: {dict(zip(best_global_models, best_global_weights))}")

    # Generate final test predictions using the best ensemble
    ensemble_preds = np.average(
        X_test[best_global_models].values,
        axis=1,
        weights=best_global_weights,
    )

    return (
        ensemble_preds,
        best_global_score,
        best_global_models,
        best_global_weights,
    )

