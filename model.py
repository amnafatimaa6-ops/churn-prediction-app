# model.py
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report

def get_trained_model(path="telco.csv"):
    # Load data
    df = pd.read_csv(path)

    # Convert TotalCharges to numeric
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # Drop customerID
    df = df.drop('customerID', axis=1)

    # Target
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    X = df.drop('Churn', axis=1)
    y = df['Churn']

    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    # Preprocessing pipelines
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_pipeline, numeric_features),
        ('cat', categorical_pipeline, categorical_features)
    ])

    # Models
    pipeline_lr = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000))
    ])

    pipeline_rf = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier())
    ])

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # GridSearch for LR
    param_grid_lr = {
        'classifier__C': [0.01, 0.1, 1, 10],
        'classifier__penalty': ['l2']
    }

    grid_lr = GridSearchCV(
        pipeline_lr,
        param_grid_lr,
        cv=5,
        scoring='f1',
        n_jobs=-1
    )
    grid_lr.fit(X_train, y_train)

    # GridSearch for RF
    param_grid_rf = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [None, 10, 20],
        'classifier__min_samples_split': [2, 5]
    }

    grid_rf = GridSearchCV(
        pipeline_rf,
        param_grid_rf,
        cv=5,
        scoring='f1',
        n_jobs=-1
    )
    grid_rf.fit(X_train, y_train)

    best_lr = grid_lr.best_estimator_
    best_rf = grid_rf.best_estimator_

    # Print metrics (optional)
    print("Logistic Regression:\n", classification_report(y_test, best_lr.predict(X_test)))
    print("\nRandom Forest:\n", classification_report(y_test, best_rf.predict(X_test)))

    # Return the best model (Random Forest for stability) and feature columns
    return best_rf, X.columns.tolist()
