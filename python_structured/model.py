from dataclasses import dataclass

from sklearn.base import BaseEstimator, RegressorMixin

@dataclass
class ShapDropLGBMRegressor(BaseEstimator, RegressorMixin):
    """
    Sklearn wrapper around the custom LightGBM DART + SHAP-drop logic.

    Needed to use GridSearch
    """
    # Parameters tune via param_grid
    num_boost