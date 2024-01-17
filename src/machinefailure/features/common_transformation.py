from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


# Log transformer class to tranform the numerical variables
class LogTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return np.log1p(X) 