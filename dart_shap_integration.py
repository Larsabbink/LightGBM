#!/usr/bin/env python3
"""
SHAP-based DART tree dropping integration

This module provides helper functions to integrate SHAP-based tree dropping
with LightGBM's DART booster.
"""

import lightgbm as lgb
import numpy as np
from typing import List, Dict, Optional, Any, Tuple


def get_tree_estimators_from_booster(booster) -> List[Dict[str, Any]]:
    """
    Extract tree estimator information from LightGBM booster.
    
    Returns a list where each element represents a tree.
    Each tree has information about which features it uses for splitting.
    Format compatible with determine_tree_scores from shap_utils.
    
    Args:
        booster: LightGBM Booster instance
        
    Returns:
        List of tree dictionaries. Each tree dict contains:
        - 'feature_usage': Dict[feature_idx: count] - how many times each feature is used
        - 'num_splits': int - total number of splits in the tree
    """
    model_dict = booster.dump_model()
    tree_infos = model_dict["tree_info"]
    
    estimators = []
    
    def count_feature_usage(tree_dict: Dict[str, Any]) -> Dict[int, int]:
        """Recursively count how many times each feature is used for splitting."""
        feature_usage = {}
        
        if "split_feature" in tree_dict:
            feature_idx = tree_dict["split_feature"]
            feature_usage[feature_idx] = feature_usage.get(feature_idx, 0) + 1
            
            # Recursively process children
            if "left_child" in tree_dict:
                child_usage = count_feature_usage(tree_dict["left_child"])
                for feat, count in child_usage.items():
                    feature_usage[feat] = feature_usage.get(feat, 0) + count
            
            if "right_child" in tree_dict:
                child_usage = count_feature_usage(tree_dict["right_child"])
                for feat, count in child_usage.items():
                    feature_usage[feat] = feature_usage.get(feat, 0) + count
        
        return feature_usage
    
    for tree_info in tree_infos:
        tree_structure = tree_info.get("tree_structure", {})
        feature_usage = count_feature_usage(tree_structure)
        
        # Calculate total number of splits
        num_splits = sum(feature_usage.values())
        
        estimator = {
            'feature_usage': feature_usage,
            'num_splits': num_splits
        }
        estimators.append(estimator)
    
    return estimators


def get_feature_usage_per_tree(booster) -> Dict[int, Dict[int, int]]:
    """
    Get feature usage counts per tree.
    
    Returns a dictionary: {tree_index: {feature_index: count}}
    where count is how many times the feature is used for splitting.
    
    Args:
        booster: LightGBM Booster instance
        
    Returns:
        Dictionary mapping tree index to feature usage counts
    """
    model_dict = booster.dump_model()
    tree_infos = model_dict["tree_info"]
    
    feature_usage = {}
    
    def count_features_in_tree(tree_dict: Dict[str, Any], tree_idx: int):
        """Recursively count feature usage in a tree."""
        if "split_feature" in tree_dict:
            feature_idx = tree_dict["split_feature"]
            
            if tree_idx not in feature_usage:
                feature_usage[tree_idx] = {}
            if feature_idx not in feature_usage[tree_idx]:
                feature_usage[tree_idx][feature_idx] = 0
            feature_usage[tree_idx][feature_idx] += 1
            
            if "left_child" in tree_dict:
                count_features_in_tree(tree_dict["left_child"], tree_idx)
            if "right_child" in tree_dict:
                count_features_in_tree(tree_dict["right_child"], tree_idx)
    
    for tree_idx, tree_info in enumerate(tree_infos):
        tree_structure = tree_info.get("tree_structure", {})
        count_features_in_tree(tree_structure, tree_idx)
    
    return feature_usage


def determine_drop_mask_xai(
    estimators: List[Dict[str, Any]], 
    drop_rate: float, 
    skip_drop: float, 
    total_shap_values: Optional[np.ndarray], 
    max_drop: Optional[int] = None, 
    current_round: Optional[int] = None, 
    learning_rate: Optional[float] = None
) -> Tuple[np.ndarray, bool]:
    """
    Determine which trees to drop based on SHAP values.
    
    This is your original function - adapt it as needed for LightGBM.
    
    Args:
        estimators: List of tree estimators (from get_tree_estimators_from_booster)
        drop_rate: Dropout rate
        skip_drop: Probability of skipping dropout
        total_shap_values: Accumulated SHAP values
        max_drop: Maximum number of trees to drop per round
        current_round: Current boosting round (for GBDT mode detection)
        learning_rate: Learning rate (for GBDT mode detection)
        
    Returns:
        tuple: (drop_mask, dropout_applied)
    """
    drop_mask = np.full(len(estimators), False)
    
    # Check if we're still in GBDT mode (no dropout)
    if current_round is not None and learning_rate is not None:
        gbdt_rounds = max(1, int(1.0 / learning_rate))
        if current_round < gbdt_rounds:
            return drop_mask, False  # GBDT mode: no dropout
    
    if np.random.rand() < skip_drop or total_shap_values is None:
        return drop_mask, False  # No dropout
    else:
        # Import your actual determine_tree_scores function
        # If it's in a module, uncomment and modify the import path as needed:
        try:
            # Option 1: If shap_utils is in the same directory
            from shap_utils import determine_tree_scores
            drop_probs = determine_tree_scores(total_shap_values, estimators)
        except ImportError:
            try:
                # Option 2: If it's a relative import
                from .shap_utils import determine_tree_scores
                drop_probs = determine_tree_scores(total_shap_values, estimators)
            except ImportError:
                # Fallback: uniform probabilities if determine_tree_scores not found
                print("  Warning: determine_tree_scores not found, using uniform probabilities")
                drop_probs = np.ones(len(estimators)) / len(estimators)
        
        indexes = np.arange(len(estimators))
        amount_of_trees_to_drop = int(np.floor(len(estimators) * drop_rate))
        
        # Apply max_drop constraint
        if max_drop is not None:
            amount_of_trees_to_drop = min(amount_of_trees_to_drop, max_drop)
        
        indexes_to_drop = np.random.choice(
            indexes, 
            amount_of_trees_to_drop, 
            replace=False, 
            p=drop_probs
        )
        
        for idx in indexes_to_drop:
            drop_mask[idx] = True
            
        return drop_mask, True  # Dropout applied


def create_shap_drop_callback(
    X_train: np.ndarray,
    y_train: Optional[np.ndarray] = None,
    shap_explainer: Optional[Any] = None,
    drop_rate: Optional[float] = None,
    skip_drop: Optional[float] = None,
    max_drop: Optional[int] = None,
    accumulate_shap: bool = True
):
    """
    Create a callback function that uses SHAP values to determine tree drops.
    
    Args:
        X_train: Training features (for SHAP computation)
        y_train: Training labels (optional, for SHAP computation)
        shap_explainer: Pre-initialized SHAP explainer (if None, will create TreeExplainer)
        drop_rate: Override drop_rate from params (if None, uses booster.params)
        skip_drop: Override skip_drop from params (if None, uses booster.params)
        max_drop: Override max_drop from params (if None, uses booster.params)
        accumulate_shap: If True, accumulates SHAP values across iterations
        
    Returns:
        Callback function that can be passed to booster.set_simple_callback()
    """
    # Accumulated SHAP values across iterations
    accumulated_shap = None
    
    def shap_drop_callback(iteration: int, userdata):
        nonlocal accumulated_shap
        
        booster = userdata
        
        # Get parameters
        params = booster.params
        current_drop_rate = drop_rate if drop_rate is not None else params.get('drop_rate', 0.1)
        current_skip_drop = skip_drop if skip_drop is not None else params.get('skip_drop', 0.5)
        current_max_drop = max_drop if max_drop is not None else params.get('max_drop', 0)
        learning_rate = params.get('learning_rate', 0.1)
        
        print(f"\n🔥 SHAP drop callback at iteration {iteration}")
        print(f"  Total trees: {booster.num_trees()}")
        print(f"  Params: drop_rate={current_drop_rate}, skip_drop={current_skip_drop}, lr={learning_rate}")
        
        # Get tree estimators
        estimators = get_tree_estimators_from_booster(booster)
        
        if len(estimators) == 0:
            print("  No trees available yet, skipping drop")
            booster.set_dart_drop_indices([])
            return
        
        # Compute SHAP values if explainer is available
        current_shap_values = None
        if shap_explainer is not None:
            try:
                import shap
                # Update explainer with current model state (important!)
                # The model has changed, so we need to update the explainer
                shap_explainer.model = booster
                
                # Compute SHAP values for current model state
                shap_values = shap_explainer.shap_values(X_train)
                
                # Handle multi-dimensional SHAP values (e.g., for multi-class)
                if isinstance(shap_values, list):
                    # Multi-class: use sum of absolute values or mean
                    shap_values = np.array(shap_values)
                    shap_values = np.sum(np.abs(shap_values), axis=0)  # Sum across classes
                elif shap_values.ndim > 2:
                    # Reshape if needed
                    shap_values = np.sum(np.abs(shap_values), axis=0)
                
                # Accumulate if requested
                if accumulate_shap:
                    if accumulated_shap is None:
                        accumulated_shap = shap_values.copy()
                    else:
                        # Ensure shapes match (in case model grew)
                        min_shape = tuple(min(s1, s2) for s1, s2 in zip(accumulated_shap.shape, shap_values.shape))
                        accumulated_shap = accumulated_shap[:min_shape[0], :min_shape[1]] if len(min_shape) == 2 else accumulated_shap[:min_shape[0]]
                        shap_values = shap_values[:min_shape[0], :min_shape[1]] if len(min_shape) == 2 else shap_values[:min_shape[0]]
                        accumulated_shap += shap_values
                    current_shap_values = accumulated_shap
                else:
                    current_shap_values = shap_values
                    
                print(f"  Computed SHAP values: shape {current_shap_values.shape}")
            except Exception as e:
                print(f"  Error computing SHAP values: {e}")
                import traceback
                traceback.print_exc()
                current_shap_values = None
        else:
            print("  No SHAP explainer provided, using default logic (random drop)")
            current_shap_values = None
        
        # Determine drop mask using your function
        drop_mask, dropout_applied = determine_drop_mask_xai(
            estimators=estimators,
            drop_rate=current_drop_rate,
            skip_drop=current_skip_drop,
            total_shap_values=current_shap_values,
            max_drop=current_max_drop if current_max_drop > 0 else None,
            current_round=iteration,
            learning_rate=learning_rate
        )
        
        if dropout_applied:
            # Extract indices where drop_mask is True
            drop_indices = [int(idx) for idx in np.where(drop_mask)[0]]
            
            # IMPORTANT: Map estimator indices to actual tree model indices
            # The estimators list corresponds to all trees in the model
            # DART expects tree indices that correspond to iteration numbers
            # We need to map estimator index to the actual tree index in the model
            # 
            # For DART, tree indices are typically: num_init_iteration + iteration_index
            # But since we're working with all trees in the model, we use direct indices
            
            num_trees = booster.num_trees()
            valid_drop_indices = [idx for idx in drop_indices if 0 <= idx < num_trees]
            
            if valid_drop_indices:
                print(f"  Dropping {len(valid_drop_indices)} trees at indices: {valid_drop_indices}")
                booster.set_dart_drop_indices(valid_drop_indices)
            else:
                print(f"  No valid trees to drop (all indices out of range)")
                booster.set_dart_drop_indices([])
        else:
            print(f"  Dropout skipped (GBDT mode or random skip)")
            booster.set_dart_drop_indices([])
    
    return shap_drop_callback


# Example usage
if __name__ == "__main__":
    import lightgbm as lgb
    
    # Create sample data
    np.random.seed(42)
    X_train = np.random.rand(100, 10)
    y_train = np.random.rand(100)
    
    # Create dataset
    train_data = lgb.Dataset(X_train, label=y_train)
    
    # Set parameters with DART
    params = {
        'objective': 'regression',
        'boosting_type': 'dart',
        'drop_rate': 0.1,
        'skip_drop': 0.0,  # Set to 0 to ensure dropping happens
        'learning_rate': 0.1,
        'num_iterations': 10,
        'verbose': -1
    }
    
    # Create booster
    booster = lgb.Booster(params, train_set=train_data)
    
    # Option 1: Create SHAP explainer if you have shap installed
    try:
        import shap
        shap_explainer = shap.TreeExplainer(booster)
        print("SHAP explainer created")
    except ImportError:
        print("SHAP not available, using default drop logic")
        shap_explainer = None
    
    # Create callback with SHAP integration
    callback = create_shap_drop_callback(
        X_train=X_train,
        y_train=y_train,
        shap_explainer=shap_explainer,
        accumulate_shap=True
    )
    
    # Set callback
    booster.set_simple_callback(callback, user_data=booster)
    
    # Train for a few iterations
    print("Training with SHAP-based drop callback...\n")
    for i in range(5):
        booster.update()
        # Update SHAP explainer with new model state
        if shap_explainer is not None:
            try:
                import shap
                shap_explainer = shap.TreeExplainer(booster)
            except:
                pass
    
    print("\n--- Done! ---")

