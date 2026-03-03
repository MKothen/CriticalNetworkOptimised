"""
Data loading and preprocessing utilities
Handles MNIST dataset loading and preparation
"""

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import os

# Cache for MNIST data to avoid reloading
_MNIST_DATA_CACHE = {}


def load_and_preprocess_mnist(num_train_max, num_test_max, seed=42):
    """
    Load and preprocess MNIST dataset for reservoir computing tasks.

    Parameters
    ----------
    num_train_max : int
        Maximum number of training samples to load
    num_test_max : int
        Maximum number of test samples to load
    seed : int
        Random seed for reproducibility

    Returns
    -------
    tuple
        (X_train_sample, y_train_onehot, y_train_sample,
         X_test_sample, y_test_onehot, y_test_sample)
    """
    cache_key = (num_train_max, num_test_max, seed)
    if cache_key in _MNIST_DATA_CACHE:
        print("Using cached MNIST data.")
        return _MNIST_DATA_CACHE[cache_key]

    print(f"Loading MNIST data (max {num_train_max} train, {num_test_max} test)...")

    try:
        mnist_data = fetch_openml(
            'mnist_784', version=1, cache=True,
            data_home=os.path.expanduser('~/sklearn_datasets'),
            parser='auto'
        )
    except Exception:
        mnist_data = fetch_openml(
            'mnist_784', version=1, cache=True,
            data_home=os.path.expanduser('~/sklearn_datasets')
        )

    X_pd, y_pd = mnist_data["data"], mnist_data["target"]
    X = X_pd.to_numpy() if isinstance(X_pd, pd.DataFrame) else np.asarray(X_pd)
    y = y_pd.to_numpy().astype(int) if isinstance(y_pd, pd.Series) else np.asarray(y_pd, dtype=int)

    # Normalize and binarize in-place to save memory
    X = (X > 127.5).astype(np.float32)  # Combined normalize + binarize in one step

    # Split into train and test
    X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y
    )

    # Free full dataset memory
    del X, y

    # Sample the requested number of examples (uses global RNG to preserve sequence)
    num_train_actual = min(num_train_max, len(X_train_full))
    num_test_actual = min(num_test_max, len(X_test_full))

    train_indices = np.random.choice(len(X_train_full), num_train_actual, replace=False)
    test_indices = np.random.choice(len(X_test_full), num_test_actual, replace=False)

    X_train_sample = X_train_full[train_indices]
    y_train_sample = y_train_full[train_indices]
    X_test_sample = X_test_full[test_indices]
    y_test_sample = y_test_full[test_indices]

    # Free split data memory
    del X_train_full, X_test_full

    # One-hot encode labels
    onehot_encoder = OneHotEncoder(sparse_output=False, categories='auto')
    y_train_onehot = onehot_encoder.fit_transform(y_train_sample.reshape(-1, 1))
    y_test_onehot = onehot_encoder.transform(y_test_sample.reshape(-1, 1))

    print(f"Loaded {len(X_train_sample)} training, {len(X_test_sample)} test samples.")

    result = (
        X_train_sample, y_train_onehot, y_train_sample,
        X_test_sample, y_test_onehot, y_test_sample.ravel()
    )

    _MNIST_DATA_CACHE[cache_key] = result
    return result


def calculate_samples_to_reach_threshold(learning_curve_dict, target_accuracy, training_subsets_sorted):
    """
    Calculate how many training samples are needed to reach a target accuracy.

    Parameters
    ----------
    learning_curve_dict : dict
        Dictionary mapping sample sizes to accuracies
    target_accuracy : float
        Target accuracy threshold
    training_subsets_sorted : list
        Sorted list of training subset sizes

    Returns
    -------
    float
        Minimum number of samples needed, or np.nan if threshold not reached
    """
    if not learning_curve_dict:
        return np.nan

    for n_samples in training_subsets_sorted:
        if n_samples in learning_curve_dict and learning_curve_dict[n_samples] >= target_accuracy:
            return n_samples

    return np.nan


def get_accuracy_at_fixed_samples(learning_curve_dict, fixed_sample_size):
    """
    Get accuracy at a specific sample size.

    Parameters
    ----------
    learning_curve_dict : dict
        Dictionary mapping sample sizes to accuracies
    fixed_sample_size : int
        Sample size to query

    Returns
    -------
    float
        Accuracy at the specified sample size, or np.nan if not available
    """
    if not learning_curve_dict:
        return np.nan
    return learning_curve_dict.get(fixed_sample_size, np.nan)
