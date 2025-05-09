import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

def full_frequency_analysis(df):
    results = {}
    cat_columns = df.select_dtypes(include=['object', 'category']).columns

    for col in cat_columns:
        freq = df[col].value_counts()
        prop = df[col].value_counts(normalize=True)
        cum_prop = prop.cumsum()
        summary = pd.DataFrame({
            'Frequency': freq,
            'Proportion': prop,
            'Cumulative Proportion': cum_prop
        })
        results[col] = summary

    return results

class GetDummiesTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        self.dummy_columns_ = None
    
    def fit(self, X, y=None):
        # Ensure X is a DataFrame
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        # Validate columns
        missing_cols = [col for col in self.columns if col not in X.columns]
        if missing_cols:
            raise ValueError(f"Columns {missing_cols} not found in input data")
        # Perform get_dummies
        X_dummy = pd.get_dummies(X, columns=self.columns, prefix_sep='_')
        self.dummy_columns_ = X_dummy.columns
        print(X_dummy.shape)
        return self
    
    def transform(self, X):
        # Ensure X is a DataFrame
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        # Apply get_dummies
        X_dummy = pd.get_dummies(X, columns=self.columns, prefix_sep='_')
        # Add missing columns with zeros
        missing_cols = set(self.dummy_columns_) - set(X_dummy.columns)
        for col in missing_cols:
            X_dummy[col] = 0
        # Reorder columns to match training set
        return X_dummy[self.dummy_columns_]

class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, features_to_keep):
        self.features_to_keep = features_to_keep
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Ensure X is a DataFrame
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        # Select available features
        available_features = [col for col in self.features_to_keep if col in X.columns]
        missing_features = [col for col in self.features_to_keep if col not in X.columns]
        X_selected = X[available_features].copy()
        # Add missing features with zeros
        for col in missing_features:
            X_selected[col] = 0
        # Reorder columns to match features_to_keep
        return X_selected[self.features_to_keep]