from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


# Log transformer class to tranform the numerical variables
class LogTransformer(BaseEstimator, TransformerMixin):
    """Logarithmic transformer for numerical variables."""
    
    def __init__(self):
        """Initialize the LogTransformer."""
        pass
    
    def fit(self, X, y=None):
        """Fit the LogTransformer."""
        return self
    
    def transform(self, X):
        """Transform the input data using logarithmic transformation."""
        if np.any(X <= 0):
            raise ValueError("Input array must contain only positive values for log transformation.")
        return np.log1p(X)
