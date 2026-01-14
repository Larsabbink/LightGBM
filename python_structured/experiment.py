import lightgbm as lgb
from load_data import get_fold_data
from callbacks import create_shap_drop_callback
from sklearn.metrics import mean_squared_error

def run_param_search(X, y, patient_ids, fold_indices, param_grid, n_folds):
    X_train, y_train, X_test, y_test = get_fold_data(
        X, y, patient_ids, fold_indices, fold_idx
    )

    booster = create_booster_with_callback(
        X_train, y_train, X_test, y_test, params, use_lightgbm_default
    )

    train_fold_model(booster, num_iterations)

    results = evaluate_fold(booster, X_train, y_train, X_test, y_test, fold_idx)

    return results

def create_booster_with_callback(X_train, y_train, X_test, y_test, params, use_lightgbm_default):
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    booster = lgb.Booster(params, train_set=train_data)
    booster.add_valid(test_data, "valid")

    if not use_lightgbm_default:
        callback = create_shap_drop_callback(
            X_train=X_train,
            y_train=y_train,
            drop_rate=params.get('drop_rate', 0.1),
            skip_drop=params.get('skip_drop', 0.5),
            max_drop=params.get('max_drop', 0),
            learning_rate=params.get('learning_rate', 0.1),
            last_shap_iteration=params.get('num_iterations', 25) - 1,
        )
        booster.set_dart_callback(callback, user_data=booster)

    return booster

def train_fold_model(booster, num_iterations, print_interval=10):
    for i in range(num_iterations):
        booster.update()
        if (i + 1) % print_interval == 0:
            train_result = booster.eval_train()[0]
            test_result = booster.eval_valid()[0]
            train_mse = float(train_result[2])
            test_mse = float(test_result[2])
            print(f"Iteration {i+1}: Train MSE={train_mse:.4f}, Test MSE={test_mse:.4f}")

def evaluate_fold(booster, X_train, y_train, X_test, y_test, fold_idx):
    y_pred_train = booster.predict(X_train)
    y_pred_test = booster.predict(X_test)
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    return {
        'fold_idx': fold_idx,
        'train_mse': train_mse,
        'test_mse': test_mse,
        'train_samples': len(X_train),
        'test_samples': len(X_test)
    }

