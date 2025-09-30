import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.cluster import FeatureAgglomeration
from scipy.stats import spearmanr
import xgboost as xgb
import openml

# Download the dataset with ID 43403 from OpenML
dataset = openml.datasets.get_dataset(43403)
X, y, categorical_indicator, attribute_names = dataset.get_data(
    target=dataset.default_target_attribute
)

# Drop 'kfold' column if it exists
if 'kfold' in X.columns:
    X = X.drop('kfold', axis=1)

# Save the combined dataset
data = pd.concat([X, y.rename('target')], axis=1)
data.to_csv('data.csv', index=False)

# Separate features and target
X = data.drop('target', axis=1)
y = data['target']

# Function to evaluate a set of features using cross-validation with different models
def evaluate_features(X_selected, y, model_name='rf'):
    if model_name == 'rf':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_name == 'xgb':
        model = xgb.XGBClassifier(n_estimators=100, random_state=42)
    elif model_name == 'lr':
        model = LogisticRegression(max_iter=1000, random_state=42)
    else:
        raise ValueError("model_name must be one of: 'rf', 'xgb', 'lr'")
    scores = cross_val_score(model, X_selected, y, cv=5, scoring='accuracy')
    return scores.mean()

# ---- Random Forest Feature Importance ----
def rf_feature_selection(X, y, k=5):
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    selected_features = X.columns[indices[:k]].tolist()
    highest_feature = X.columns[indices[0]]
    return selected_features, highest_feature

# ---- XGBoost Feature Importance ----
def xgb_feature_selection(X, y, k=5):
    model = xgb.XGBClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    selected_features = X.columns[indices[:k]].tolist()
    highest_feature = X.columns[indices[0]]
    return selected_features, highest_feature

# ---- Logistic Regression Feature Importance ----
def lr_feature_selection(X, y, k=5):
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X, y)
    importances = np.abs(model.coef_[0])
    indices = np.argsort(importances)[::-1]
    selected_features = X.columns[indices[:k]].tolist()
    highest_feature = X.columns[indices[0]]
    return selected_features, highest_feature

# ---- Feature Agglomeration (fit on raw data; select top across clusters) ----
def feature_agglomeration(X, y=None, k=5):
    n_clusters = 5
    fa = FeatureAgglomeration(n_clusters=n_clusters)
    fa.fit(X.values)
    clusters = fa.labels_

    # Calculate variance for each feature (raw scale)
    variances = X.var().to_dict()

    # Create list of (feature, variance, cluster) tuples
    feature_info = [(col, variances[col], clusters[i]) for i, col in enumerate(X.columns)]

    # Sort by variance descending
    feature_info.sort(key=lambda x: x[1], reverse=True)

    # Track selected clusters to avoid duplicates initially
    selected_clusters = set()
    selected_features = []
    feature_rankings = {}

    # Select top features across all clusters (one per cluster first)
    for feature, variance, cluster in feature_info:
        if len(selected_features) >= k:
            break
        if cluster not in selected_clusters:
            selected_features.append(feature)
            selected_clusters.add(cluster)
            feature_rankings[feature] = variance

    # If we still need more features, allow multiple from same cluster
    if len(selected_features) < k:
        remaining = [f for f, v, c in feature_info if f not in selected_features]
        for feature in remaining:
            if len(selected_features) >= k:
                break
            # Retrieve variance for ranking dict for completeness
            feature_rankings[feature] = variances[feature]
            selected_features.append(feature)

    highest_feature = selected_features[0] if selected_features else None
    return selected_features[:k], highest_feature

# ---- Highly Variable Gene Selection (variance-based) ----
def hvgs_selection(X, y, k=5):
    variances = X.var()
    sorted_features = variances.sort_values(ascending=False)
    selected_features = sorted_features.index[:k].tolist()
    highest_feature = sorted_features.index[0] if not sorted_features.empty else None
    return selected_features, highest_feature

# ---- Spearman Correlation ----
def spearman_selection(X, y, k=5):
    correlations = []
    for col in X.columns:
        corr, _ = spearmanr(X[col], y)
        correlations.append((col, abs(corr)))
    correlations.sort(key=lambda x: x[1], reverse=True)
    selected_features = [corr[0] for corr in correlations[:k]]
    highest_feature = correlations[0][0] if correlations else None
    return selected_features, highest_feature

# Apply all feature selection methods
print("Applying feature selection methods...")

# First round of feature selection (set1)
rf_features, rf_top = rf_feature_selection(X, y)
xgb_features, xgb_top = xgb_feature_selection(X, y)
lr_features, lr_top = lr_feature_selection(X, y)
fa_features, fa_top = feature_agglomeration(X, y)
hvgs_features, hvgs_top = hvgs_selection(X, y)
spearman_features, spearman_top = spearman_selection(X, y)

# Display first round results
print("\nTop 5 features per method (set1):")
print(f"Random Forest: {rf_features}")
print(f"XGBoost: {xgb_features}")
print(f"Logistic Regression: {lr_features}")
print(f"Feature Agglomeration: {fa_features}")
print(f"Highly Variable Gene Selection: {hvgs_features}")
print(f"Spearman Correlation: {spearman_features}")

# Cross-validate the selected features
rf_score = evaluate_features(X[rf_features], y, 'rf')
xgb_score = evaluate_features(X[xgb_features], y, 'xgb')
lr_score = evaluate_features(X[lr_features], y, 'lr')
fa_score = evaluate_features(X[fa_features], y, 'rf')
hvgs_score = evaluate_features(X[hvgs_features], y, 'rf')
spearman_score = evaluate_features(X[spearman_features], y, 'rf')

print("\nCross-validation scores for set1:")
print(f"Random Forest features: {rf_score:.4f}")
print(f"XGBoost features: {xgb_score:.4f}")
print(f"Logistic Regression features: {lr_score:.4f}")
print(f"Feature Agglomeration features: {fa_score:.4f}")
print(f"Highly Variable Gene Selection features: {hvgs_score:.4f}")
print(f"Spearman Correlation features: {spearman_score:.4f}")

# Create reduced datasets by removing the highest feature for each algorithm
X_rf_reduced = X.drop(columns=[rf_top])
X_xgb_reduced = X.drop(columns=[xgb_top])
X_lr_reduced = X.drop(columns=[lr_top])
X_fa_reduced = X.drop(columns=[fa_top]) if fa_top is not None else X.copy()
X_hvgs_reduced = X.drop(columns=[hvgs_top])
X_spearman_reduced = X.drop(columns=[spearman_top])

print(f"\nHighest features removed to create reduced datasets:")
print(f"Random Forest highest feature removed: {rf_top}")
print(f"XGBoost highest feature removed: {xgb_top}")
print(f"Logistic Regression highest feature removed: {lr_top}")
print(f"Feature Agglomeration highest feature removed: {fa_top}")
print(f"Highly Variable Gene Selection highest feature removed: {hvgs_top}")
print(f"Spearman Correlation highest feature removed: {spearman_top}")

# Second round of feature selection on reduced datasets (set2)
rf_features_r2, _ = rf_feature_selection(X_rf_reduced, y, k=4)
xgb_features_r2, _ = xgb_feature_selection(X_xgb_reduced, y, k=4)
lr_features_r2, _ = lr_feature_selection(X_lr_reduced, y, k=4)
fa_features_r2, _ = feature_agglomeration(X_fa_reduced, y, k=4)
hvgs_features_r2, _ = hvgs_selection(X_hvgs_reduced, y, k=4)
spearman_features_r2, _ = spearman_selection(X_spearman_reduced, y, k=4)

# Display second round results
print(f"\nTop 4 features from reduced datasets per method (set2):")
print(f"Random Forest: {rf_features_r2}")
print(f"XGBoost: {xgb_features_r2}")
print(f"Logistic Regression: {lr_features_r2}")
print(f"Feature Agglomeration: {fa_features_r2}")
print(f"Highly Variable Gene Selection: {hvgs_features_r2}")
print(f"Spearman Correlation: {spearman_features_r2}")
