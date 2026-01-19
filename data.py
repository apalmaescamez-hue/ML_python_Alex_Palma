import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def get_preprocessing_pipeline(numerical_cols, categorical_cols):
    """Creates a preprocessing pipeline for numerical and categorical data."""
    
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, numerical_cols),
            ('cat', cat_transformer, categorical_cols)
        ]
    )
    
    return preprocessor

def load_and_split_data(file_path='marketing_data.csv', target='converted'):
    """Loads CSV data and splits it into training and testing sets."""
    df = pd.read_csv(file_path)
    
    # Drop IDs or non-relevant time columns if needed
    X = df.drop(columns=[target, 'lead_id', 'created_at'])
    y = df[target]
    
    # Identify numerical and categorical columns automatically if not specified
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = X.select_dtypes(include=['object']).columns
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, numerical_cols, categorical_cols
