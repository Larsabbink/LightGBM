#!/usr/bin/env python3
"""
Example: Accessing tree information from DART callback

This demonstrates how to access tree structure information
during training to determine feature usage per tree.
"""

import lightgbm as lgb
import numpy as np

def get_feature_usage_per_tree(booster):
    """
    Get a dictionary mapping tree index to feature usage counts.
    
    Returns:
        dict: {tree_index: {feature_index: count}}
    """
    # Get tree structure from booster
    model_dict = booster.dump_model()
    tree_infos = model_dict["tree_info"]
    
    feature_usage = {}
    
    def count_features_in_tree(tree_dict, tree_idx):
        """Recursively count feature usage in a tree."""
        if "split_feature" in tree_dict:
            # This is a split node
            feature_idx = tree_dict["split_feature"]
            
            if tree_idx not in feature_usage:
                feature_usage[tree_idx] = {}
            if feature_idx not in feature_usage[tree_idx]:
                feature_usage[tree_idx][feature_idx] = 0
            feature_usage[tree_idx][feature_idx] += 1
            
            # Recursively process children
            if "left_child" in tree_dict:
                count_features_in_tree(tree_dict["left_child"], tree_idx)
            if "right_child" in tree_dict:
                count_features_in_tree(tree_dict["right_child"], tree_idx)
    
    # Process each tree
    for tree_idx, tree_info in enumerate(tree_infos):
        tree_structure = tree_info.get("tree_structure", {})
        count_features_in_tree(tree_structure, tree_idx)
    
    return feature_usage


def training_callback_with_tree_info(iteration, userdata):
    """
    Callback that accesses tree information from the booster.
    """
    # Get the booster object from userdata
    booster = userdata  # We'll pass the booster as userdata
    
    print(f"\n🔥 Callback at iteration {iteration}")
    
    # Get tree information
    num_trees = booster.num_trees()
    print(f"  Total trees: {num_trees}")
    
    # Get feature usage per tree
    feature_usage = get_feature_usage_per_tree(booster)
    
    # Print summary
    print(f"  Trees with splits: {len(feature_usage)}")
    if len(feature_usage) > 0:
        # Example: show feature usage for first tree
        first_tree = list(feature_usage.keys())[num_trees - 1]
        print(f"  Tree {first_tree} uses features: {list(feature_usage[first_tree].keys())[:5]}...")
    
    # Get parameters
    params = booster.params
    drop_rate = params.get('drop_rate', 0.1)
    skip_drop = params.get('skip_drop', 0.5)
    learning_rate = params.get('learning_rate', 0.1)
    
    print(f"  DART params: drop_rate={drop_rate}, skip_drop={skip_drop}, lr={learning_rate}")


if __name__ == "__main__":
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
        'skip_drop': 0.5,
        'learning_rate': 0.1,
        'num_iterations': 10,
        'verbose': -1
    }
    
    # Create booster
    booster = lgb.Booster(params, train_set=train_data)
    
    # Set callback - pass booster as userdata
    booster.set_simple_callback(training_callback_with_tree_info, user_data=booster)
    
    # Train for a few iterations
    print("Training with tree info callback...\n")
    for i in range(5):
        booster.update()
    
    print("\n--- Done! ---")

