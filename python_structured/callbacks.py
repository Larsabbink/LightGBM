import random
from extract_data_from_booster import get_tree_estimators_from_booster

def create_shap_drop_callback(X_train, y_train):
    _y_train = y_train

    def shap_drop_callback(iteration, userdata):
        booster = userdata
        y_train = _y_train

        params = booster.params
        current_drop_rate = drop_rate if drop_rate is not None else params.get('drop_rate', 0.1)
        current_skip_drop = skip_drop if skip_drop is not None else params.get('skip_drop', 0.5)
        current_max_drop = max_drop if max_drop is not None else params.get('max_drop', 0)
        learning_rate = params.get('learning_rate', 0.1)
        is_shap_computation_phase = (iteration > last_shap_iteration)

        if not is_shap_computation_phase:
            determine_drop_mask(
                booster,
                drop_rate=current_drop_rate,
                skip_drop=current_skip_drop,
                max_drop=current_max_drop
            )

        return 0


def determine_drop_mask(booster, drop_rate=None, skip_drop=None, max_drop=None, uniform_drop=True, drop_seed=42):
    random.seed(drop_seed)
    
    if random.random() < skip_drop:
        booster.set_dart_drop_indices([])
        return []
    
    num_trees = booster.num_trees()
    num_trees_per_iter = booster.num_trees_per_iteration()
    current_iter = num_trees // num_trees_per_iter
    
    if current_iter <= 0:
        booster.set_dart_drop_indices([])
        return []
    
    drop_indices = []
    
    if current_uniform_drop:
        effective_drop_rate = current_drop_rate
        if current_max_drop > 0:
            effective_drop_rate = min(effective_drop_rate, current_max_drop / current_iter)
        
        for i in range(current_iter):
            if random.random() < effective_drop_rate:
                drop_indices.append(i)
                if len(drop_indices) >= current_max_drop and current_max_drop > 0:
                    break
    else:
        learning_rate = params.get('learning_rate', 0.1)
        tree_weights = [learning_rate] * current_iter
        sum_weight = sum(tree_weights)
        
        if sum_weight > 0:
            inv_average_weight = len(tree_weights) / sum_weight
            effective_drop_rate = current_drop_rate
            if current_max_drop > 0:
                effective_drop_rate = min(effective_drop_rate, current_max_drop * inv_average_weight / sum_weight)
            
            for i in range(current_iter):
                drop_prob = effective_drop_rate * tree_weights[i] * inv_average_weight
                if random.random() < drop_prob:
                    drop_indices.append(i)
                    if len(drop_indices) >= current_max_drop and current_max_drop > 0:
                        break
    
    booster.set_dart_drop_indices(drop_indices)
    return drop_indices