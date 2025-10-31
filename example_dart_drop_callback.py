#!/usr/bin/env python3
"""
Example: Setting DART drop indices from Python callback

This demonstrates how to override the random tree dropping in DART
by providing specific drop indices from a Python callback.
"""

import lightgbm as lgb
import numpy as np

def custom_drop_callback(iteration, userdata):
    """
    Callback that determines which trees to drop based on custom logic.
    In this example, we drop specific trees (first 2 trees for demonstration).
    In practice, you would use SHAP values or other criteria.
    """
    booster = userdata
    
    print(f"\n🔥 Drop callback at iteration {iteration}")
    print(f"  Total trees: {booster.num_trees()}")
    
    # Example: Drop first 2 trees (for demonstration)
    # In your actual implementation, you would:
    # 1. Get tree structure information
    # 2. Calculate SHAP values
    # 3. Use your determine_drop_mask_xai function
    # 4. Extract indices from the drop mask
    
    if iteration > 1:  # Only drop if we have at least 2 trees
        # Example: drop trees at indices 0 and 1 (first two trees)
        # Note: These are tree indices in the model, not iteration indices
        drop_indices = [0, 1]
        
        # Limit to available trees
        num_trees = booster.num_trees()
        drop_indices = [idx for idx in drop_indices if idx < num_trees]
        
        if drop_indices:
            print(f"  Setting drop indices: {drop_indices}")
            booster.set_dart_drop_indices(drop_indices)
        else:
            print(f"  No trees to drop (clearing drop indices)")
            booster.set_dart_drop_indices([])
    else:
        print(f"  Too few trees, using default random dropping")


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
        'skip_drop': 0.0,  # Set to 0 to ensure dropping happens
        'learning_rate': 0.1,
        'num_iterations': 10,
        'verbose': -1
    }
    
    # Create booster
    booster = lgb.Booster(params, train_set=train_data)
    
    # Set callback - pass booster as userdata
    booster.set_simple_callback(custom_drop_callback, user_data=booster)
    
    # Train for a few iterations
    print("Training with custom drop callback...\n")
    for i in range(5):
        booster.update()
    
    print("\n--- Done! ---")

