import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.cluster import FeatureAgglomeration
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from scipy.stats import spearmanr
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
    
    scores = cross_val_score(model, X_selected, y, cv=5, scoring='accuracy')
    return scores.mean()

# ---- Random Forest Feature Importance ----
def rf_feature_selection(X, y, k=5):
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    # Get feature importances
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Select top k features
    selected_features = X.columns[indices[:k]].tolist()
    highest_feature = X.columns[indices[0]]
    return selected_features, highest_feature

# ---- XGBoost Feature Importance ----
def xgb_feature_selection(X, y, k=5):
    model = xgb.XGBClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Get feature importances
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Select top k features
    selected_features = X.columns[indices[:k]].tolist()
    highest_feature = X.columns[indices[0]]
    return selected_features, highest_feature

# ---- Logistic Regression Feature Importance ----
def lr_feature_selection(X, y, k=5):
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X, y)
    
    # Get feature importances (absolute coefficients)
    importances = np.abs(model.coef_[0])
    indices = np.argsort(importances)[::-1]
    
    # Select top k features
    selected_features = X.columns[indices[:k]].tolist()
    highest_feature = X.columns[indices[0]]
    return selected_features, highest_feature

# ---- Feature Agglomeration ----
def feature_agglomeration(X, y, k=5):
    # Create Feature Agglomeration
    n_clusters = int(X.shape[1] * 0.2)  # Select 20% of features
    n_clusters = max(n_clusters, k)  # Ensure we have at least k clusters
    
    # Calculate correlation distance matrix
    X_values = X.values  # Convert DataFrame to NumPy array
    corr = np.abs(np.corrcoef(X_values.T))
    corr_dist = 1 - corr
    
    # Calculate variance for each feature
    variances = np.var(X_values, axis=0)
    var_matrix = np.zeros((len(variances), len(variances)))
    
    for i in range(len(variances)):
        for j in range(len(variances)):
            var_matrix[i, j] = abs(variances[i] - variances[j])
    
    # Normalize the variance matrix
    if np.max(var_matrix) > 0:
        var_matrix = var_matrix / np.max(var_matrix)
    
    # Combine correlation distance and variance with given ratio
    combined_dist = 0.2 * corr_dist + 0.8 * var_matrix
    
    # Using FeatureAgglomeration with the combined distance
    agglo = FeatureAgglomeration(n_clusters=n_clusters)
    agglo.fit(combined_dist)
    
    # Extract top features from each cluster
    top_features = []
    feature_names = X.columns.tolist()
    
    # For each cluster, find the feature with highest variance
    for cluster_idx in range(n_clusters):
        cluster_features_idx = np.where(agglo.labels_ == cluster_idx)[0]
        if len(cluster_features_idx) > 0:
            # Get features in this cluster
            cluster_features = [feature_names[i] for i in cluster_features_idx]
            cluster_X = X[cluster_features]
            cluster_variances = cluster_X.var().values
            
            # Select the feature with highest variance in this cluster
            best_feature_idx = np.argmax(cluster_variances)
            top_features.append(cluster_features[best_feature_idx])
            
            if len(top_features) >= k:
                break
    
    # Ensure we have k features
    top_features = top_features[:k]
    highest_feature = top_features[0] if top_features else None
    return top_features, highest_feature

# ---- Highly Variable Gene Selection (adapting for general feature selection) ----
def hvgs_selection(X, y, k=5):
    # Calculate variance for each feature
    variances = X.var()
    
    # Sort features by variance
    sorted_features = variances.sort_values(ascending=False)
    
    # Select top k features with highest variance
    selected_features = sorted_features.index[:k].tolist()
    highest_feature = sorted_features.index[0] if not sorted_features.empty else None
    return selected_features, highest_feature

# ---- Spearman Correlation ----
def spearman_selection(X, y, k=5):
    correlations = []
    
    # Calculate Spearman correlation for each feature with the target
    for col in X.columns:
        corr, _ = spearmanr(X[col], y)
        correlations.append((col, abs(corr)))
    
    # Sort by absolute correlation
    correlations.sort(key=lambda x: x[1], reverse=True)
    
    # Select top k features
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
X_fa_reduced = X.drop(columns=[fa_top])
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
