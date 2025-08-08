
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
pd.set_option('future.no_silent_downcasting', True)

class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X.drop(columns=self.columns, errors='ignore')

class ConvertToNumeric(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        return X

class DropHighNullColumns(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.4):
        self.threshold = threshold
        self.columns_to_drop_ = []

    def fit(self, X, y=None):
        self.columns_to_drop_ = X.columns[X.isnull().mean() > self.threshold].tolist()
        return self

    def transform(self, X):
        return X.drop(columns=self.columns_to_drop_, errors='ignore')

class InvalidValueToNaN(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        
        # values outside bounds physiologically invalid
        if 'chol' in X.columns:
            X.loc[X['chol'] < 40, 'chol'] = np.nan
        if 'oldpeak' in X.columns:
            X.loc[X['oldpeak'] < 0, 'oldpeak'] = np.nan
        
        return X

class RandomNormalImputer(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, random_state=None):
        self.columns = columns
        self.random_state = random_state
        self.stats_ = {}

    def fit(self, X, y=None):
        if self.columns is None:
            self.columns = X.select_dtypes(include=np.number).columns.tolist()
        
        self.stats_ = {}
        for col in self.columns:
            if X[col].isnull().any():
                mean = X[col].mean()
                std = X[col].std()

                # values outside bounds physiologically invalid
                if col == 'chol':
                    lower_bound = 40
                elif col == 'oldpeak':
                    lower_bound = 0
                else:
                    lower_bound = -np.inf

                self.stats_[col] = (mean, std, lower_bound)
        return self

    def transform(self, X):
        X = X.copy()
        rng = np.random.default_rng(self.random_state)

        for col, (mean, std, lower_bound) in self.stats_.items():
            n_missing = X[col].isna().sum()
            if n_missing > 0:
                # generate values until all are valid
                valid_values = []
                while len(valid_values) < n_missing:
                    sampled = rng.normal(loc=mean, scale=std, size=n_missing)
                    valid_sampled = sampled[sampled >= lower_bound]
                    valid_values.extend(valid_sampled.tolist())
                valid_values = valid_values[:n_missing]

                # rounding rules by domain
                if col in ['trestbps', 'chol', 'thalch']:
                    valid_values = [int(round(v)) for v in valid_values]
                elif col == 'oldpeak':
                    valid_values = [round(v, 1) for v in valid_values]
                # else: leave values as is for continuous features

                X.loc[X[col].isna(), col] = valid_values
        return X

    
class CategoricalImputer(BaseEstimator, TransformerMixin):
    def __init__(self, strategy='mode', fill_value='Missing'):
        self.strategy = strategy
        self.fill_value = fill_value
        self.fill_values_ = {}
        self.columns_ = []

    def fit(self, X, y=None):
        X = X.copy()
        self.columns_ = X.select_dtypes(exclude=np.number).columns.tolist()

        for col in self.columns_:
            if self.strategy == 'mode':
                mode_series = X[col].mode()
                self.fill_values_[col] = mode_series[0] if not mode_series.empty else self.fill_value
            elif self.strategy == 'constant':
                self.fill_values_[col] = self.fill_value
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.columns_:
            if col in X.columns:
                X[col] = X[col].fillna(self.fill_values_[col])
        return X
        
class CategoricalMissingCategoryImputer(BaseEstimator, TransformerMixin):
    def __init__(self, fill_value='Missing'):
        self.fill_value = fill_value
        self.columns_ = []

    def fit(self, X, y=None):
        self.columns_ = X.select_dtypes(exclude=np.number).columns.tolist()
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.columns_:
            if col in X.columns:
                X[col] = X[col].fillna(self.fill_value)
        return X
    
class OutlierCapper(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, method='iqr', factor=1.5):
        self.columns = columns
        self.method = method
        self.factor = factor
        self.caps_ = {}

    def fit(self, X, y=None):
        X = X.copy()
        if self.columns is None:
            self.columns = X.select_dtypes(include=np.number).columns.tolist()

        for col in self.columns:
            if self.method == 'iqr':
                Q1 = X[col].quantile(0.25)
                Q3 = X[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - self.factor * IQR
                upper = Q3 + self.factor * IQR

                self.caps_[col] = (lower, upper)
        return self

    def transform(self, X):
        X = X.copy()
        for col, (lower, upper) in self.caps_.items():
            X[col] = X[col].clip(lower=lower, upper=upper)
        return X