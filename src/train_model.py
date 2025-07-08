from utils import load_data, save_model  
from evaluate import print_metrics, plot_confusion_matrix
from sklearn.model_selection import train_test_split
from evaluate import cross_validate_model
from pipelines import logistic_regression, random_forest, xg_boost
from sklearn.model_selection import GridSearchCV
import pandas as pd
from evaluate import plot_roc_curve
from sklearn.metrics import f1_score, precision_score, recall_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
import shap
import os
from sklearn.utils import resample

# Create necessary directories before any file saving
os.makedirs("plots", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Load and split
df = load_data()
X = df.drop(columns='HeartDisease')
y = df['HeartDisease']


# Train-test split, 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
 
# Create a copy of the training set and add target as well as 'Sex' column
# This is because it will be used for gender-specific resampling
train_df = X_train.copy()
train_df['HeartDisease'] = y_train
train_df['Sex'] = X_train['Sex']

# Split training data by sex for upsampling minority group (females)
train_males = train_df[train_df['Sex'] == 'M']
train_females = train_df[train_df['Sex'] == 'F']

# Upsample females to match males
train_females_upsampled = resample(train_females, replace=True, n_samples=len(train_males), random_state=42)

# Combine the resampled females with original males
train_balanced = pd.concat([train_males, train_females_upsampled])

# Shuffle Dataset
train_balanced = train_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

# Separate features and target from the newly balanced training dataset
X_train_balanced = train_balanced.drop(columns='HeartDisease')
y_train_balanced = train_balanced['HeartDisease']


# Define models 
models = {
    "Logistic Regression": logistic_regression,
    "Random Forest": random_forest,
    "XGBoost": xg_boost
}

# Define hyperparameter grids for each model to be used with GridSearchCV
# These grids are used to search for the best model parameters via cross-validation
param_grids = {
    "Logistic Regression": {
        'classifier__C': [0.01, 0.1, 1, 10],
        'classifier__penalty': ['l2'],
        'classifier__solver': ['lbfgs']
    },
    "Random Forest": {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [5, 8, 12],
        'classifier__min_samples_split': [5, 10],
        'classifier__min_samples_leaf': [3, 5, 8]
    },
    "XGBoost": {
        'classifier__n_estimators': [50, 100, 150],
        'classifier__max_depth': [3, 4, 5],
        'classifier__learning_rate': [0.01, 0.05, 0.1],
        'classifier__reg_alpha': [0.5, 1, 5],
        'classifier__reg_lambda': [5, 10, 15]
    }
}

# Initialize a list to store evaluation results and placeholder for XGBoost's grid
# Note: Only chosed XGBoost because it is the best model according to result
results = [] # Store evaluation metrics for each model
xgb_grid = None  # Will hold GridSearchCV result for XGBoost

# Loop through each modell and its pipeline
for name, model in models.items():
    print(f"\nRunning GridSearchCV for: {name}")

    # Perform hyperparameter tuning (using 5-fold cross-validation)
    grid = GridSearchCV(model, param_grids[name], cv=5, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)

    # Show the best hyperparameters found
    print(f"Best parameterse for {name}: {grid.best_params_}")

    # Evaluate model on test set
    print(f"\n{name} Test Performance")
    y_pred = grid.predict(X_test)
    test_metrics = print_metrics(y_test, y_pred)

    
    # Evaluate model on training set to check for overfitting
    print(f"\n{name} Train Performance")
    y_train_pred = grid.predict(X_train)
    train_metrics = print_metrics(y_train, y_train_pred)

    # Show side-by-side comparison of train vs. test performance to check overfitting
    print(f"\n{name} Performance Comparison:")
    print(f"{'Metric':<10}{'Train':>10}{'Test':>10}")
    print(f"{'Accuracy':<10}{train_metrics['accuracy']:>10.4f}{test_metrics['accuracy']:>10.4f}")
    print(f"{'F1 Score':<10}{train_metrics['f1_score']:>10.4f}{test_metrics['f1_score']:>10.4f}")
    print(f"{'Precision':<10}{train_metrics['precision']:>10.4f}{test_metrics['precision']:>10.4f}")
    print(f"{'Recall':<10}{train_metrics['recall']:>10.4f}{test_metrics['recall']:>10.4f}")

    # Save confusion matrix plot for all model to 'plots' folder
    fig = plot_confusion_matrix(y_test, y_pred, name, return_fig=True)
    fig.savefig(f"plots/{name.lower().replace(' ', '_')}_confusion_matrix.png")
    plt.close(fig)

    # Store the model's test metrics
    results.append({
        'model': name,
        'accuracy': test_metrics['accuracy'],
        'f1_score': test_metrics['f1_score'],
        'precision': test_metrics['precision'],
        'recall': test_metrics['recall']
     })
    # Save XGBoost's trained GridSearchCV object for further analysis
    if name == "XGBoost":
        xgb_grid = grid

# Compile results and show model comparison sorted by F1 Score
# Note: Best model should be XGBoost
df_results = pd.DataFrame(results)
print("\nModel Comparison: ")
print(df_results.sort_values(by='f1_score', ascending=False))

# For XGBoost model, generate its ROC curve using predicted probabilities
if xgb_grid:
    y_proba = xgb_grid.predict_proba(X_test)[:, 1] # probability of class 1
    plot_roc_curve(y_test, y_proba, 'XGBoost')

"""
Now the we've identified XGBoost as the best model based on F1 score,
the next step is to determine the optimal decision threshold
(instead of using the default 0.5) to maximize performance.
"""
# Create a list for thresholds 
thresholds = []

# Get thresholds from 0.10 to 0.89 and append list
for i in range(10, 90):
    threshold = i / 100
    thresholds.append(threshold)

# Initialize best threshold tracking variables
best_threshold = 0.5
best_f1 = 0

# Iterate over each threshold in the list to find the one that maximizes F1 score 
for threshold in thresholds:
    y_pred_thresh = (y_proba >= threshold).astype(int)
    f1 = f1_score(y_test, y_pred_thresh)
    precision = precision_score(y_test, y_pred_thresh)
    recall = recall_score(y_test, y_pred_thresh)

    print(f"\nThreshold: {threshold:.2f} | F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")

    # Update best threshold if current F1 is better
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold 

print(f"\nBest threshold based on F1: {best_threshold} (F1 = {best_f1:.4f})")


# Compare default (0.5) vs. optimized threshold
y_pred_default = (y_proba >= 0.5).astype(int)
y_pred_optimized = (y_proba >= best_threshold).astype(int)


print(f"\nDefault Threshold (0.5) Performance")
print_metrics(y_test, y_pred_default)

print(f"\nBest Threshold ({best_threshold}) Performance")
print_metrics(y_test, y_pred_optimized)

# Evaluate optimized threshold performance by gender
print("\nGender-Specific Performance at Best Threshold:")
for sex in ['M', 'F']:
    idx = X_test['Sex'] == sex
    print(f"\nPerformance for {sex}:")
    print_metrics(y_test[idx], y_pred_optimized[idx])


# Get the best model pipeline
best_pipeline = xgb_grid.best_estimator_

# Extract the fitted preprocessor and classifier
best_preprocessor = best_pipeline.named_steps['preprocessor']
best_classifier = best_pipeline.named_steps['classifier']

# Get Feature names
feature_names = best_preprocessor.get_feature_names_out()

# Extract feature importances   
feature_importance = best_classifier.feature_importances_

# Sort importances by indices
importance_indices = np.argsort(feature_importance)[::-1]

# Plot and save feature importance bar chart
plt.figure(figsize=(10, 6))
plt.title('Feature Importances - XGBoost')
plt.bar(range(len(feature_importance)), feature_importance[importance_indices], align='center')
plt.xticks(range(len(feature_importance)), feature_names[importance_indices], rotation=90)
plt.tight_layout()
plt.savefig("plots/xgb_feature_importance.png")
plt.close()


# === SHAP Analysis Section ===

# Get transformed features for test set
X_test_transformed = best_preprocessor.transform(X_test)

# Create SHAP explainer with XGBoost model
explainer = shap.Explainer(best_classifier)

# Get SHAP values for test data
shap_values = explainer(X_test_transformed)

# Clean feature names (remove transformer prefixes)
clean_feature_names = []
for name in feature_names:
    clean_name = name.split('__')[-1]
    clean_feature_names.append(clean_name)

shap_values.feature_names = clean_feature_names

# Generate and save SHAP summary plot 
shap.summary_plot(shap_values, show=False)
plt.tight_layout()
plt.savefig("plots/shap_summary_plot.png")
plt.close()

# Generate and save SHAP bar plot
shap.plots.bar(shap_values, show=False)
plt.tight_layout()
plt.savefig("plots/shap_bar_plot.png")
plt.close()

# Save the best XGBoost pipeline

save_model(best_pipeline, "models/best_xgb_pipeline.pkl")
print("✅ Model saved to models/best_xgb_pipeline.pkl")

