"""
Data Preprocessing Module
Handles all data preprocessing methods including encoding, scaling, and transformation
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import (
    LabelEncoder, 
    OneHotEncoder, 
    StandardScaler, 
    MinMaxScaler,
    RobustScaler,
    Normalizer,
    OrdinalEncoder
)
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import streamlit as st


class DataPreprocessor:
    """Comprehensive data preprocessing class with all major techniques"""
    
    def __init__(self, df):
        self.df = df.copy()
        self.processed_df = df.copy()
        self.encoders = {}
        self.scalers = {}
        
    def get_column_types(self):
        """Identify numeric and categorical columns"""
        numeric_cols = self.processed_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = self.processed_df.select_dtypes(include=['object', 'category']).columns.tolist()
        return numeric_cols, categorical_cols
    
    # ============ ENCODING METHODS ============
    
    def apply_label_encoding(self, columns):
        """Apply Label Encoding to specified columns"""
        for col in columns:
            if col in self.processed_df.columns:
                le = LabelEncoder()
                self.processed_df[f'{col}_label_encoded'] = le.fit_transform(
                    self.processed_df[col].astype(str)
                )
                self.encoders[f'{col}_label'] = le
        return self.processed_df
    
    def apply_onehot_encoding(self, columns):
        """Apply One-Hot Encoding to specified columns"""
        for col in columns:
            if col in self.processed_df.columns:
                # Get dummies
                dummies = pd.get_dummies(
                    self.processed_df[col], 
                    prefix=f'{col}_onehot',
                    drop_first=False
                )
                self.processed_df = pd.concat([self.processed_df, dummies], axis=1)
        return self.processed_df
    
    def apply_ordinal_encoding(self, columns, categories_dict=None):
        """Apply Ordinal Encoding to specified columns"""
        for col in columns:
            if col in self.processed_df.columns:
                if categories_dict and col in categories_dict:
                    # Use provided order
                    cat_order = categories_dict[col]
                    self.processed_df[f'{col}_ordinal_encoded'] = self.processed_df[col].map(
                        {cat: idx for idx, cat in enumerate(cat_order)}
                    )
                else:
                    # Auto-detect order
                    unique_vals = sorted(self.processed_df[col].unique())
                    self.processed_df[f'{col}_ordinal_encoded'] = self.processed_df[col].map(
                        {val: idx for idx, val in enumerate(unique_vals)}
                    )
        return self.processed_df
    
    def apply_frequency_encoding(self, columns):
        """Apply Frequency Encoding to specified columns"""
        for col in columns:
            if col in self.processed_df.columns:
                freq_map = self.processed_df[col].value_counts(normalize=True).to_dict()
                self.processed_df[f'{col}_freq_encoded'] = self.processed_df[col].map(freq_map)
        return self.processed_df
    
    def apply_target_encoding(self, columns, target_col):
        """Apply Target Encoding (mean encoding) to specified columns"""
        if target_col in self.processed_df.columns:
            for col in columns:
                if col in self.processed_df.columns:
                    target_mean = self.processed_df.groupby(col)[target_col].mean()
                    self.processed_df[f'{col}_target_encoded'] = self.processed_df[col].map(target_mean)
        return self.processed_df
    
    # ============ SCALING METHODS ============
    
    def apply_standard_scaling(self, columns):
        """Apply Standard Scaling (Z-score normalization) to numeric columns"""
        scaler = StandardScaler()
        for col in columns:
            if col in self.processed_df.columns:
                self.processed_df[f'{col}_standard_scaled'] = scaler.fit_transform(
                    self.processed_df[[col]]
                )
                self.scalers[f'{col}_standard'] = scaler
        return self.processed_df
    
    def apply_minmax_scaling(self, columns):
        """Apply MinMax Scaling to numeric columns"""
        scaler = MinMaxScaler()
        for col in columns:
            if col in self.processed_df.columns:
                self.processed_df[f'{col}_minmax_scaled'] = scaler.fit_transform(
                    self.processed_df[[col]]
                )
                self.scalers[f'{col}_minmax'] = scaler
        return self.processed_df
    
    def apply_robust_scaling(self, columns):
        """Apply Robust Scaling (using median and IQR) to numeric columns"""
        scaler = RobustScaler()
        for col in columns:
            if col in self.processed_df.columns:
                self.processed_df[f'{col}_robust_scaled'] = scaler.fit_transform(
                    self.processed_df[[col]]
                )
                self.scalers[f'{col}_robust'] = scaler
        return self.processed_df
    
    def apply_normalization(self, columns):
        """Apply L2 Normalization to numeric columns"""
        normalizer = Normalizer()
        for col in columns:
            if col in self.processed_df.columns:
                self.processed_df[f'{col}_normalized'] = normalizer.fit_transform(
                    self.processed_df[[col]]
                )
                self.scalers[f'{col}_normalizer'] = normalizer
        return self.processed_df
    
    def apply_log_transformation(self, columns):
        """Apply Log Transformation to numeric columns"""
        for col in columns:
            if col in self.processed_df.columns:
                # Add small constant to avoid log(0)
                self.processed_df[f'{col}_log'] = np.log1p(self.processed_df[col].abs())
        return self.processed_df
    
    def apply_sqrt_transformation(self, columns):
        """Apply Square Root Transformation to numeric columns"""
        for col in columns:
            if col in self.processed_df.columns:
                self.processed_df[f'{col}_sqrt'] = np.sqrt(self.processed_df[col].abs())
        return self.processed_df
    
    # ============ TEXT VECTORIZATION ============
    
    def apply_tfidf(self, text_column, max_features=100):
        """Apply TF-IDF to text column"""
        if text_column in self.processed_df.columns:
            vectorizer = TfidfVectorizer(max_features=max_features)
            tfidf_matrix = vectorizer.fit_transform(
                self.processed_df[text_column].astype(str)
            )
            
            # Create DataFrame from TF-IDF matrix
            tfidf_df = pd.DataFrame(
                tfidf_matrix.toarray(),
                columns=[f'{text_column}_tfidf_{i}' for i in range(tfidf_matrix.shape[1])]
            )
            
            self.processed_df = pd.concat([self.processed_df, tfidf_df], axis=1)
            self.encoders[f'{text_column}_tfidf'] = vectorizer
        
        return self.processed_df
    
    def apply_count_vectorization(self, text_column, max_features=100):
        """Apply Count Vectorization to text column"""
        if text_column in self.processed_df.columns:
            vectorizer = CountVectorizer(max_features=max_features)
            count_matrix = vectorizer.fit_transform(
                self.processed_df[text_column].astype(str)
            )
            
            # Create DataFrame from count matrix
            count_df = pd.DataFrame(
                count_matrix.toarray(),
                columns=[f'{text_column}_count_{i}' for i in range(count_matrix.shape[1])]
            )
            
            self.processed_df = pd.concat([self.processed_df, count_df], axis=1)
            self.encoders[f'{text_column}_count'] = vectorizer
        
        return self.processed_df
    
    # ============ MISSING VALUE HANDLING ============
    
    def handle_missing_values(self, strategy='mean'):
        """Handle missing values with various strategies"""
        numeric_cols, categorical_cols = self.get_column_types()
        
        if strategy == 'mean':
            for col in numeric_cols:
                if self.processed_df[col].isnull().any():
                    self.processed_df[col].fillna(
                        self.processed_df[col].mean(), 
                        inplace=True
                    )
        elif strategy == 'median':
            for col in numeric_cols:
                if self.processed_df[col].isnull().any():
                    self.processed_df[col].fillna(
                        self.processed_df[col].median(), 
                        inplace=True
                    )
        elif strategy == 'mode':
            for col in categorical_cols:
                if self.processed_df[col].isnull().any():
                    mode_val = self.processed_df[col].mode()
                    if len(mode_val) > 0:
                        self.processed_df[col].fillna(mode_val[0], inplace=True)
        elif strategy == 'ffill':
            self.processed_df.fillna(method='ffill', inplace=True)
        elif strategy == 'bfill':
            self.processed_df.fillna(method='bfill', inplace=True)
        elif strategy == 'drop':
            self.processed_df.dropna(inplace=True)
        elif strategy == 'constant':
            self.processed_df.fillna(0, inplace=True)
        
        return self.processed_df
    
    # ============ OUTLIER HANDLING ============
    
    def remove_outliers_iqr(self, columns, multiplier=1.5):
        """Remove outliers using IQR method"""
        for col in columns:
            if col in self.processed_df.columns:
                Q1 = self.processed_df[col].quantile(0.25)
                Q3 = self.processed_df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - multiplier * IQR
                upper_bound = Q3 + multiplier * IQR
                
                # Filter outliers
                self.processed_df = self.processed_df[
                    (self.processed_df[col] >= lower_bound) & 
                    (self.processed_df[col] <= upper_bound)
                ]
        
        return self.processed_df
    
    def cap_outliers_iqr(self, columns, multiplier=1.5):
        """Cap outliers using IQR method (winsorization)"""
        for col in columns:
            if col in self.processed_df.columns:
                Q1 = self.processed_df[col].quantile(0.25)
                Q3 = self.processed_df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - multiplier * IQR
                upper_bound = Q3 + multiplier * IQR
                
                # Cap outliers
                self.processed_df[col] = self.processed_df[col].clip(lower_bound, upper_bound)
        
        return self.processed_df
    
    # ============ FEATURE ENGINEERING ============
    
    def create_binning(self, column, bins=5, labels=None):
        """Create bins for numeric column"""
        if column in self.processed_df.columns:
            if labels is None:
                labels = [f'bin_{i}' for i in range(bins)]
            
            self.processed_df[f'{column}_binned'] = pd.cut(
                self.processed_df[column], 
                bins=bins, 
                labels=labels
            )
        
        return self.processed_df
    
    def create_polynomial_features(self, columns, degree=2):
        """Create polynomial features"""
        from sklearn.preprocessing import PolynomialFeatures
        
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        
        if len(columns) > 0:
            poly_features = poly.fit_transform(self.processed_df[columns])
            feature_names = [f'poly_{i}' for i in range(poly_features.shape[1])]
            
            poly_df = pd.DataFrame(poly_features, columns=feature_names)
            self.processed_df = pd.concat([self.processed_df, poly_df], axis=1)
        
        return self.processed_df
    
    def create_interaction_features(self, col1, col2):
        """Create interaction features between two columns"""
        if col1 in self.processed_df.columns and col2 in self.processed_df.columns:
            self.processed_df[f'{col1}_x_{col2}'] = self.processed_df[col1] * self.processed_df[col2]
        
        return self.processed_df
    
    # ============ UTILITIES ============
    
    def remove_duplicates(self):
        """Remove duplicate rows"""
        before = len(self.processed_df)
        self.processed_df.drop_duplicates(inplace=True)
        after = len(self.processed_df)
        return before - after
    
    def reset_index(self):
        """Reset index of processed dataframe"""
        self.processed_df.reset_index(drop=True, inplace=True)
        return self.processed_df
    
    def get_statistics(self):
        """Get comprehensive statistics about the data"""
        stats = {
            'shape': self.df.shape,
            'columns': len(self.df.columns),
            'rows': len(self.df),
            'missing_values': self.df.isnull().sum().sum(),
            'numeric_columns': len(self.df.select_dtypes(include=['int64', 'float64']).columns),
            'categorical_columns': len(self.df.select_dtypes(include=['object', 'category']).columns),
            'memory_usage': self.df.memory_usage(deep=True).sum() / 1024**2,  # MB
            'duplicates': self.df.duplicated().sum()
        }
        return stats
