#!/usr/bin/env python3
"""
Example: Using simple callback with LightGBM DART booster

This demonstrates how to define a Python function that gets called
during each training iteration when using DART boosting.
"""

import lightgbm as lgb
import numpy as np

# Define your Python callback function
def my_training_callback(iteration, userdata):
    """This function will be called during each DART training iteration."""
    print(f"🔥 Python callback triggered at iteration {iteration}")
    # You can add any Python code here:
    # - Log metrics
    # - Save checkpoints
    # - Modify training parameters
    # - etc.

# Example usage:
if __name__ == "__main__":
    # Create some sample data
    np.random.seed(42)
    X_train = np.random.rand(100, 10)
    y_train = np.random.rand(100)
    
    # Create dataset
    train_data = lgb.Dataset(X_train, label=y_train)
    
    # Set parameters - use 'dart' as boosting type
    params = {
        'objective': 'regression',
        'boosting_type': 'dart',  # Important: use 'dart' to trigger callback in DART::TrainOneIter
        'num_iterations': 10,
        'verbose': -1
    }
    
    # Create booster first
    booster = lgb.Booster(params, train_set=train_data)
    
    print("\n--- Setting callback and training ---\n")
    
    # Set the callback BEFORE training
    booster.set_simple_callback(my_training_callback)
    
    # Train with callback - callback will be triggered during each iteration
    print("Training for 5 iterations...")
    for i in range(5):
        print(f"  Calling update() iteration {i+1}...")
        booster.update()
    
    print("\n--- Clearing callback ---\n")
    
    # Clear the callback
    booster.set_simple_callback(None)
    
    # Continue training - no callback will be triggered
    print("Updating once more without callback...")
    booster.update()
    
    print("\n--- Done! ---")

