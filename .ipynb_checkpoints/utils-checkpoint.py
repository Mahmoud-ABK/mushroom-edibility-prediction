import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

def full_frequency_analysis(df):
    import pandas as pd  
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
        # Perform get_dummies on training data to capture all possible columns
        X_dummy = pd.get_dummies(X, columns=self.columns, prefix_sep='_')
        self.dummy_columns_ = X_dummy.columns
        return self
    
    def transform(self, X):
        # Apply get_dummies and ensure consistent columns
        X_dummy = pd.get_dummies(X, columns=self.columns, prefix_sep='_')
        # Add missing columns with zeros
        missing_cols = set(self.dummy_columns_) - set(X_dummy.columns)
        for col in missing_cols:
            X_dummy[col] = 0
        # Reorder columns to match training set
        X_dummy = X_dummy[self.dummy_columns_]
        return X_dummy


class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, features_to_keep):
        self.features_to_keep = features_to_keep
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Select only the specified features, fill missing ones with 0
        available_features = [col for col in self.features_to_keep if col in X.columns]
        missing_features = [col for col in self.features_to_keep if col not in X.columns]
        X_selected = X[available_features]
        if missing_features:
            for col in missing_features:
                X_selected[col] = 0
        return X_selected[self.features_to_keep]