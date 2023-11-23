# ref: https://www.kaggle.com/code/rapela/tpss3e5-1st-place-solution-rapids-xgboost
from functools import partial

import numpy as np
import scipy.optimize as opt
from sklearn.metrics import cohen_kappa_score


class OptimizedRounder:
    def __init__(self):
        self.coef_ = 0

    def _kappa_loss(self, coef, X, y):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            else:
                X_p[i] = 3

        log_loss = cohen_kappa_score(y, X_p, weights="quadratic")
        return -log_loss

    def fit(self, X, y):
        loss_partial = partial(self._kappa_loss, X=X, y=y)
        initial_coef = [0.5, 1.5, 2.5]
        self.coef_ = opt.minimize(loss_partial, initial_coef, method="nelder-mead")

    def predict(self, X, coef):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            else:
                X_p[i] = 3

        return X_p.astype(int)

    @property
    def coefficients(self):
        return self.coef_["x"]


class OrdinalMultiClassifier:
    def __init__(self, predictors, num_classes):
        assert len(predictors) == (num_classes - 1)
        self.num_classes = num_classes
        self.predictors = predictors

    def fit(self, X, y, **kwargs):
        for k, p in enumerate(self.predictors):
            y_binary = np.where(y > k, 1, 0)
            p.fit(X, y_binary, **kwargs)

    def predict_proba(self, X):
        probabilities = np.zeros((X.shape[0], self.num_classes))

        proba_k = np.ones((X.shape[0], ))
        for k, p in enumerate(self.predictors):
            greater_than_k = p.predict_proba(X)[:, 1]
            proba_k -= greater_than_k
            probabilities[:, k] = proba_k

            proba_k = greater_than_k
        
        probabilities[:, -1] = greater_than_k

        return probabilities
        
    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


class OrdinalClassCutoff:
    def __init__(self, model):
        self.model = model
        self.coefficients = None

    def fit(self, X, y, **kwargs):
        self.model.fit(X, y, **kwargs)

        rounder = OptimizedRounder()
        preds = self.model.predict(X)
        rounder.fit(preds, y)
        self.coefficients = rounder.coefficients

    def predict(self, X):
        preds = self.model.predict(X)

        rounder = OptimizedRounder()
        rounded = rounder.predict(preds, self.coefficients)

        return rounded
    