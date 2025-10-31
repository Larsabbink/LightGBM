#!/usr/bin/env python3
"""
Complete example: SHAP-based DART tree dropping

This demonstrates the full integration of SHAP-based tree dropping
with LightGBM's DART booster.
"""

import lightgbm as lgb
import numpy as np
from dart_shap_integration import create_shap_drop_callback

if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    n_samples = 500
    n_features = 20
    X_train = np.random.rand(n_samples, n_features)
    y_train = np.random.rand(n_samples)
    
    # Create dataset
    train_data = lgb.Dataset(X_train, label=y_train)
    
    # Set parameters with DART
    params = {
        'objective': 'regression',
        'boosting_type': 'dart',
        'drop_rate': 0.1,
        'skip_drop': 0.0,  # Set to 0 to ensure dropping happens (or use your desired value)
        'max_drop': 5,  # Maximum trees to drop per iteration
        'learning_rate': 0.1,
        'num_iterations': 20,
        'verbose': -1
    }
    
    # Create booster
    booster = lgb.Booster(params, train_set=train_data)
    
    # Option 1: Use SHAP explainer (if available)
    shap_explainer = None
    try:
        import shap
        # Create SHAP explainer - update it as model grows
        # Note: For efficiency, you might want to create it once and update it
        shap_explainer = shap.TreeExplainer(booster)
        print("✓ SHAP explainer created\n")
    except ImportError:
        print("⚠ SHAP not available - install with: pip install shap\n")
    except Exception as e:
        print(f"⚠ Could not create SHAP explainer: {e}\n")
    
    # Create callback with SHAP integration
    callback = create_shap_drop_callback(
        X_train=X_train,
        y_train=y_train,
        shap_explainer=shap_explainer,
        drop_rate=params.get('drop_rate'),  # Use params or override
        skip_drop=params.get('skip_drop'),   # Use params or override  
        max_drop=params.get('max_drop'),     # Use params or override
        accumulate_shap=True  # Accumulate SHAP values across iterations
    )
    
    # Set callback
    booster.set_simple_callback(callback, user_data=booster)
    
    # Train with SHAP-based dropping
    print("Training with SHAP-based drop callback...\n")
    for i in range(params.get('num_iterations', 10)):
        booster.update()
        
        # Update SHAP explainer periodically (optional - can be expensive)
        if shap_explainer is not None and i % 5 == 0:
            try:
                import shap
                shap_explainer = shap.TreeExplainer(booster)
            except:
                pass
    
    print("\n--- Training complete! ---")
    print(f"Final number of trees: {booster.num_trees()}")

