"""Target encoding module with cross-validation, smoothing and noise."""

import numpy as np
from sklearn.model_selection import StratifiedKFold


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
    to prevent overfitting.
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    X_val : pd.DataFrame
        Validation features.
    y_train : pd.Series
        Training target variable.
    cat_cols : list
        List of categorical column names to encode.
    X_test : pd.DataFrame, optional
        Test features to encode. Default is None.
    smoothing : float, optional
        Smoothing parameter for regularization.
        Default is 1.0.
    noise_level : float, optional
        Standard deviation of Gaussian noise added to training
        encodings. Default is 0.01.
    cv_folds : int, optional
        Number of cross-validation folds for encoding training data.
        Default is 5.
    
    Returns
    -------
    tuple
        If X_test is None: (X_train_encoded, X_val_encoded)
        If X_test is provided: (X_train_encoded, X_val_encoded, X_test_encoded)
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


