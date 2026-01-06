def get_tree_estimators_from_booster(booster):
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
    
    def count_feature_usage(tree_dict):
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