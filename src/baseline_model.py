from typing import Optional

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import validation


class BaselineModel(BaseEstimator, RegressorMixin):
    """Baseline model using simple map and groupby operations"""
    def __init__(self, groupby: Optional[str] = None, aggregate: str = "median"):
        """Set the aggregate function and the groupby column"""
        self.aggregate = aggregate
        self.groupby = groupby

    def fit(self, X: pd.DataFrame, label: str):
        """Compute global aggregate value and mapping based on groupby"""
        self.label_ = label
        X = X.dropna(subset=[self.label_])

        self.global_agg_ = X[self.label_].agg(self.aggregate)

        self.mapping_ = {}
        if self.groupby:
            self.mapping_ = X.groupby(self.groupby)[self.label_].agg(self.aggregate).to_dict()

        return self
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        validation.check_is_fitted(self)

        X = X.dropna(subset=[self.label_])

        if self.groupby:
            map_operator = lambda x: self.mapping_.get(x, self.global_agg_) if x is not np.nan else np.nan
            vector_map = np.vectorize(map_operator)
            return vector_map(X[self.groupby])
        else:
            return np.full((X.shape[0], ), self.global_agg_)   
