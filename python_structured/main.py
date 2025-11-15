import numpy as np
from load_data import load_data_with_folds
from experiment import run_single_fold

def run_experiment(fold_idx: int = 0, n_folds: int = 10, random_state: int = 42):
    X, y, patient_ids, fold_indices = load_data_with_folds(
        file_path="data/slice_localization_data.csv",
        target_col="reference",
        drop_cols=["patientId"],
        n_folds=n_folds,
        random_state=random_state
    )

    params = {
        'objective': 'regression',
        'boosting_type': 'dart',
        'num_iterations': 25,  # ensemble_size
        'num_leaves': 100,
        'learning_rate': 0.2,
        'feature_fraction_bynode': 0.2,
        'drop_rate': 0.2,
        'metric': 'mse',
        'verbose': -1
    }

    fold_results = []
    for fold_idx in range(n_folds):
        result = run_single_fold(X, y, patient_ids, fold_indices, fold_idx, params, 25)
        fold_results.append(result)

    print_cv_summary(fold_results)    

def print_cv_summary(fold_results):
    print("\n" + "=" * 60)
    print("CROSS-VALIDATION SUMMARY")
    print("=" * 60)
    for result in fold_results:
        print(f"Fold {result['fold_idx']:2d}: Train MSE={result['train_mse']:.4f}, Test MSE={result['test_mse']:.4f} (Train: {result['train_samples']:5d}, Test: {result['test_samples']:5d})")
    avg_test_mse = np.mean([r['test_mse'] for r in fold_results])
    std_test_mse = np.std([r['test_mse'] for r in fold_results])
    print(f"\nAverage Test MSE: {avg_test_mse:.4f} ± {std_test_mse:.4f}")

if __name__ == "__main__":
    run_experiment()

# Average Test MSE: 30.8327 ± 13.7149