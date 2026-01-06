import pandas as pd
import numpy as np
import json

def load_data_with_folds(file_path, target_col, drop_cols, n_folds):
    df = pd.read_csv(file_path)

    X = df.drop(columns=[target_col] + drop_cols)
    y = df[target_col].copy()

    patient_ids = df['patientId'].copy()
    unique_patient_ids = patient_ids.unique()
    shuffled_patient_ids = np.random.permutation(unique_patient_ids)
    patient_to_fold = {patient: i % n_folds for i, patient in enumerate(shuffled_patient_ids)}

    fold_indices = [[] for _ in range(n_folds)]
    for patient_id in patient_ids:
        fold_idx = patient_to_fold[patient_id]
        fold_indices[fold_idx].append(patient_id)

    fold_arrays = [np.array(fold_indices) for fold_indices in fold_indices]

    return X, y, patient_ids, fold_arrays

def get_fold_data(X, y, patient_ids, fold_indices, fold_idx):
    test_patient_ids = set(fold_indices[fold_idx])
    train_patient_ids = set(np.concatenate([fold_indices[i] for i in range(len(fold_indices)) if i != fold_idx]))
    
    # Convert patient IDs to row indices
    test_mask = patient_ids.isin(test_patient_ids)
    train_mask = patient_ids.isin(train_patient_ids)
    
    test_indices = np.where(test_mask)[0]
    train_indices = np.where(train_mask)[0]
    
    X_train = X.iloc[train_indices].copy()
    y_train = y.iloc[train_indices].copy()
    X_test = X.iloc[test_indices].copy()
    y_test = y.iloc[test_indices].copy()

    # Verify no patient leakage
    train_patients = set(patient_ids.iloc[train_indices].unique())
    test_patients = set(patient_ids.iloc[test_indices].unique())
    overlap = train_patients & test_patients
    
    if overlap:
        raise ValueError(f"Patient leakage detected! Overlapping patients: {overlap}")

    return X_train, y_train, X_test, y_test

def load_param_grid(config_path: str):
    with open(config_path, "r") as f:
        return json.load(f)