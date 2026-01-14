import numpy as np

def calculate_shap_values(booster, X_test):
    n_trees = booster.num_trees()
    shap_prev = 0
    per_tree_shap = []

    for k in range(1, n_trees + 1):
        shap_k = booster.predict(X_test, num_iteration=k, pred_contrib=True)
        per_tree_shap.append(shap_k - shap_prev)
        shap_prev = shap_k

    return np.stack(per_tree_shap, axis=0)