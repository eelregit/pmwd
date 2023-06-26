"""Some util functions related to the Symbolic Regression."""
import numpy as np
from pysr import PySRRegressor
from sklearn.ensemble import RandomForestRegressor


def gen_data_sr():
    """Generate the input data & output value for SR fitting."""
    # sample k and the input features of so nn
    X, y = None, None
    X_names = None
    return X, y, X_names


def get_feature_importance(X, y, n_estimators=100, max_depth=3, random_state=None):
    """Get the feature importance and indices, from high to low."""
    clf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                random_state=random_state)
    clf.fit(X, y)
    idx = np.argsort(clf.feature_importances_)[::-1]  # high to low

    return clf.feature_importances_[idx], idx
