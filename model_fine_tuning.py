import os

import joblib
import lightgbm as lgb
import pandas as pd
import xgboost as xgb
from bedrock_mistral_prediction import classify_using_llm
from preprocessing import pre_processing_data
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier


def get_data_structure_to_predict(df):
    # we dont use the following columns:
    # Sex
    # Prompt Length
    # Prediction Mistral
    # Loan Category
    # Monthly Payment
    df.drop(
        columns=[
            "Sex",
            "Prompt Length",
            "Prediction Mistral",
            "Loan Category",
            "Monthly_Payment",
        ],
        inplace=True,
    )
    X = df.drop(columns=["Prediction Mistral Small"])
    y = df["Prediction Mistral Small"]
    string_columns = X.select_dtypes(include=["object"]).columns
    X = pd.get_dummies(X, columns=string_columns)
    X = X.sort_index(axis=1)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=1
    )

    return X_train, X_test, y_train, y_test


def print_results_of_modeling(y_test, y_pred):
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"Test Accuracy: {accuracy}")
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    print("Classification Report:")
    print(report)


def train_dtc(X_train, X_test, y_train, y_test):
    # Initialize the classifiers
    clf1 = DecisionTreeClassifier(random_state=1)
    # Define the parameter grid for DecisionTreeClassifier
    param_grid = {
        "max_depth": [3, 5, 7, 10],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }

    # Initialize GridSearchCV
    grid_search = GridSearchCV(
        estimator=clf1, param_grid=param_grid, cv=5, n_jobs=-1, verbose=0
    )

    # Fit GridSearchCV on the training data
    grid_search.fit(X_train, y_train)

    # Get the best estimator
    clf1 = grid_search.best_estimator_

    # Predict on the test data
    y_pred = clf1.predict(X_test)

    print_results_of_modeling(y_test, y_pred)

    return clf1


def train_lgbm(X_train, X_test, y_train, y_test):
    clf2 = lgb.LGBMClassifier(random_state=1, verbosity=-1)
    # Define the parameter grid for LGBMClassifier
    param_grid_lgb = {
        "num_leaves": [31, 50, 70],
        "learning_rate": [0.01, 0.05, 0.1],
        "n_estimators": [100, 200, 500],
    }

    # Initialize GridSearchCV for LGBMClassifier
    grid_search_lgb = GridSearchCV(
        estimator=clf2, param_grid=param_grid_lgb, cv=5, n_jobs=-1, verbose=1
    )

    # Fit GridSearchCV on the training data
    grid_search_lgb.fit(X_train, y_train)

    # Get the best estimator
    clf2 = grid_search_lgb.best_estimator_

    # Predict on the test data
    y_pred = clf2.predict(X_test)

    print_results_of_modeling(y_test, y_pred)

    return clf2


def train_xgboost(X_train, X_test, y_train, y_test):
    # Encode the target variable
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)

    clf3 = xgb.XGBClassifier(random_state=1)

    # Define the parameter grid for XGBClassifier
    param_grid_xgb = {
        "max_depth": [3, 5, 7, 10],
        "learning_rate": [0.01, 0.05, 0.1],
        "n_estimators": [100, 200, 500],
    }

    # Initialize GridSearchCV for XGBClassifier
    grid_search_xgb = GridSearchCV(
        estimator=clf3, param_grid=param_grid_xgb, cv=5, n_jobs=-1, verbose=1
    )

    # Fit GridSearchCV on the training data
    grid_search_xgb.fit(X_train, y_train_encoded)

    # Get the best estimator
    clf3 = grid_search_xgb.best_estimator_

    # Predict on the test data
    y_pred_encoded = clf3.predict(X_test)

    # Decode the predictions back to original labels
    y_pred = label_encoder.inverse_transform(y_pred_encoded)

    print_results_of_modeling(y_test, y_pred)

    return clf3


def train_voting(X_train, X_test, y_train, y_test, clf1, clf2, clf3):
    # Create a VotingClassifier
    voting_clf = VotingClassifier(
        estimators=[("dt", clf1), ("lgb", clf2), ("xgb", clf3)], voting="soft"
    )

    # Fit the VotingClassifier on the training data
    voting_clf.fit(X_train, y_train)

    # Predict on the test data
    y_pred = voting_clf.predict(X_test)
    print_results_of_modeling(y_test, y_pred)

    return voting_clf


def model_main():
    file_path = "credit_risk_reto_processed.csv"

    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
    else:
        df_processed = pre_processing_data("credit_risk_reto.csv")
        df = classify_using_llm(df_processed)
        df.to_csv(file_path, index=False)

    X_train, X_test, y_train, y_test = get_data_structure_to_predict(df)
    print("Training DecisionTreeClassifier...")
    clf1 = train_dtc(X_train, X_test, y_train, y_test)
    print("Training LGBMClassifier...")
    clf2 = train_lgbm(X_train, X_test, y_train, y_test)
    print("Training XGBClassifier...")
    clf3 = train_xgboost(X_train, X_test, y_train, y_test)
    print("Training VotingClassifier...")
    voting_clf = train_voting(X_train, X_test, y_train, y_test, clf1, clf2, clf3)
    joblib.dump(voting_clf, "voting_classifier.pkl")
