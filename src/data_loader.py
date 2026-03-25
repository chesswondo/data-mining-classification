import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer

def load_and_preprocess_data(
    csv_path: str, 
    target_col: str, 
    drop_cols: Optional[List[str]] = None,
    test_size: float = 0.2, 
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, LabelEncoder]:
    
    df = pd.read_csv(csv_path)
    if drop_cols:
        existing_drop_cols = [col for col in drop_cols if col in df.columns]
        df = df.drop(columns=existing_drop_cols)
        
    df = df.dropna()

    cat_cols_to_convert = ['Gender', 'Ethnicity', 'ParentalEducation', 'Tutoring', 
                           'ParentalSupport', 'Extracurricular', 'Sports', 'Music', 'Volunteering']
    
    for col in cat_cols_to_convert:
        if col in df.columns:
            df[col] = df[col].astype(str)

    X = df.drop(columns=[target_col])
    y = df[target_col].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='passthrough'
    )
    
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)
    
    return X_train_processed, X_test_processed, y_train_encoded, y_test_encoded, le