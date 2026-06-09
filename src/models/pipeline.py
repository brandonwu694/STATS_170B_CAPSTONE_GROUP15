from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight


def build_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    """Build preprocessing fit only on training data inside each model pipeline."""
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_cols),
            ("categorical", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor


def build_classifier(numeric_cols: list[str], categorical_cols: list[str]) -> Pipeline:
    """Build the selected preprocessing + gradient boosting classifier pipeline."""
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    classifier = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=250,
        max_leaf_nodes=31,
        l2_regularization=0.05,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", classifier),
        ]
    )


def build_logistic_regression_baseline(numeric_cols: list[str], categorical_cols: list[str]) -> Pipeline:
    """Build a regularized multinomial logistic regression baseline."""
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    classifier = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        n_jobs=None,
        random_state=42,
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", classifier),
        ]
    )


def build_dummy_baseline() -> DummyClassifier:
    """Build a majority-class baseline that ignores predictors."""
    return DummyClassifier(strategy="most_frequent")


def fit_with_balanced_weights(pipeline: Pipeline, X_train, y_train) -> Pipeline:
    sample_weight = compute_sample_weight(class_weight="balanced", y=np.asarray(y_train))
    pipeline.fit(X_train, y_train, model__sample_weight=sample_weight)
    return pipeline
