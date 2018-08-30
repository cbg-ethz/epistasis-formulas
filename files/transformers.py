import numpy as np
from sklearn import base, utils


class ColumnDropper(base.BaseEstimator, base.TransformerMixin):
    """Transformer to drop columns from dataset.

    Parameters
    ----------
    columns: (0-based) indices of columns to drop
    """

    def __init__(self, columns=None):
        self.columns = tuple() if columns is None else columns

    def fit(self, X, y=None):
        """Do nothing and return the estimator unchanged

        This method is just there to implement the usual API and hence
        work in pipelines.
        """
        return self

    def transform(self, X, y=None):
        """Drop specified columns of X

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The dataset to process.

        Returns
        -------
        X : np.ndarray shape [n_samples, NF]
            The matrix of features, where NF is the number of features
            remaining after `columns` where dropped.
        """
        return np.delete(X, self.columns, axis=1)
