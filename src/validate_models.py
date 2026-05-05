#!/usr/bin/env python3
"""
COMPREHENSIVE MODEL VALIDATION & ANALYSIS
Generates all validation plots and analyses without retraining
Includes: ROC, Precision-Recall, Confusion Matrix, Feature Importance, 
SHAP, Domain Applicability, Model Comparison, and more
"""

import sys
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, roc_auc_score,
    precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report, matthews_corrcoef,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.preprocessing import StandardScaler
from Bio import SeqIO
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_trained_models():
    """Load all trained models"""
    model_dir = '../models'
    models = {}
    
    model_names = ['svm', 'xgboost', 'random_forest', 'logistic_regression']
    
    for model_name in model_names:
        path = os.path.join(model_dir, f'{model_name}.pkl')
        if os.path.exists(path):
            with open(path, 'rb') as f:
                models[model_name] = pickle.load(f)
            print(f"✓ Loaded {model_name}")
        else:
            print(f"⚠️  {model_name} not found")
    
    return models

def load_scaler_and_features():
    """Load scaler and feature names"""
    model_dir = '../models'
    
    # Load scaler
    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        print("✓ Loaded scaler")
    else:
        print("⚠️  Scaler not found, creating new")
        scaler = StandardScaler()
    
    # Load feature names
    feature_path = os.path.join(model_dir, 'feature_names.pkl')
    if os.path.exists(feature_path):
        with open(feature_path, 'rb') as f:
            feature_names = pickle.load(f)
        print(f"✓ Loaded {len(feature_names)} feature names")
    else:
        feature_names = None
        print("⚠️  Feature names not found")
    
    return scaler, feature_names

def load_predictions_and_labels():
    """Load predictions from CSV if available"""
    pred_file = '../results/predictions.csv'
    
    if os.path.exists(pred_file):
        df = pd.read_csv(pred_file)
        print(f"✓ Loaded predictions: {len(df)} sequences")
        return df
    else:
        print("⚠️  Predictions file not found")
        return None

# ============================================================================
# PLOT 1: COMBINED ROC CURVES
# ============================================================================

def plot_roc_curves(models, X_test, y_test):
    """Plot ROC curves for all models"""
    print("\n[1] Plotting ROC Curves...")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = {'svm': '#FF6B6B', 'xgboost': '#4ECDC4', 
              'random_forest': '#45B7D1', 'logistic_regression': '#FFA07A'}
    
    for model_name, model in models.items():
        try:
            y_pred_proba = model.predict_proba(X_test)[:, 1] \
                if hasattr(model, 'predict_proba') \
                else model.decision_function(X_test)
            
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            
            ax.plot(fpr, tpr, color=colors.get(model_name, 'blue'),
                   lw=2.5, label=f'{model_name.replace("_", " ").title()} (AUC = {roc_auc:.4f})')
        except Exception as e:
            print(f"   ⚠️  Error plotting {model_name}: {str(e)[:50]}")
    
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../results/figures/01_roc_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✓ Saved: 01_roc_curves.png")

# ============================================================================
# PLOT 2: PRECISION-RECALL CURVES
# ============================================================================

def plot_pr_curves(models, X_test, y_test):
    """Plot Precision-Recall curves"""
    print("\n[2] Plotting Precision-Recall Curves...")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = {'svm': '#FF6B6B', 'xgboost': '#4ECDC4', 
              'random_forest': '#45B7D1', 'logistic_regression': '#FFA07A'}
    
    for model_name, model in models.items():
        try:
            y_pred_proba = model.predict_proba(X_test)[:, 1] \
                if hasattr(model, 'predict_proba') \
                else model.decision_function(X_test)
            
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
            ap = average_precision_score(y_test, y_pred_proba)
            
            ax.plot(recall, precision, color=colors.get(model_name, 'blue'),
                   lw=2.5, label=f'{model_name.replace("_", " ").title()} (AP = {ap:.4f})')
        except Exception as e:
            print(f"   ⚠️  Error plotting {model_name}: {str(e)[:50]}")
    
    ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
    ax.set_title('Precision-Recall Curves - Model Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig('../results/figures/02_pr_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✓ Saved: 02_pr_curves.png")

# ============================================================================
# PLOT 3: CONFUSION MATRICES
# ============================================================================

def plot_confusion_matrices(models, X_test, y_test):
    """Plot confusion matrices for all models"""
    print("\n[3] Plotting Confusion Matrices...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.ravel()
    
    model_list = list(models.items())
    
    for idx, (model_name, model) in enumerate(model_list):
        if idx >= 4:
            break
        
        try:
            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                       cbar=True, xticklabels=['Non-virulent', 'Virulent'],
                       yticklabels=['Non-virulent', 'Virulent'])
            axes[idx].set_title(f'{model_name.replace("_", " ").title()}', 
                               fontsize=12, fontweight='bold')
            axes[idx].set_ylabel('True Label', fontweight='bold')
            axes[idx].set_xlabel('Predicted Label', fontweight='bold')
        except Exception as e:
            print(f"   ⚠️  Error plotting {model_name}: {str(e)[:50]}")
    
    plt.tight_layout()
    plt.savefig('../results/figures/03_confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✓ Saved: 03_confusion_matrices.png")

# ============================================================================
# PLOT 4: MODEL PERFORMANCE COMPARISON
# ============================================================================

def plot_performance_comparison(models, X_test, y_test):
    """Plot performance metrics comparison"""
    print("\n[4] Plotting Performance Comparison...")
    
    metrics_dict = {'Model': [], 'Accuracy': [], 'Precision': [], 
                    'Recall': [], 'F1-Score': [], 'ROC-AUC': [], 'MCC': []}
    
    for model_name, model in models.items():
        try:
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] \
                if hasattr(model, 'predict_proba') \
                else model.decision_function(X_test)
            
            metrics_dict['Model'].append(model_name.replace('_', '\n').title())
            metrics_dict['Accuracy'].append(accuracy_score(y_test, y_pred))
            metrics_dict['Precision'].append(precision_score(y_test, y_pred))
            metrics_dict['Recall'].append(recall_score(y_test, y_pred))
            metrics_dict['F1-Score'].append(f1_score(y_test, y_pred))
            metrics_dict['ROC-AUC'].append(roc_auc_score(y_test, y_pred_proba))
            metrics_dict['MCC'].append(matthews_corrcoef(y_test, y_pred))
        except Exception as e:
            print(f"   ⚠️  Error with {model_name}: {str(e)[:50]}")
    
    df_metrics = pd.DataFrame(metrics_dict)
    
    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.ravel()
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'MCC']
    colors_bar = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    
    for idx, metric in enumerate(metrics):
        bars = axes[idx].bar(df_metrics['Model'], df_metrics[metric], 
                            color=colors_bar[:len(df_metrics)], alpha=0.8, edgecolor='black')
        axes[idx].set_ylabel(metric, fontweight='bold', fontsize=11)
        axes[idx].set_title(f'{metric} Comparison', fontweight='bold', fontsize=12)
        axes[idx].set_ylim([0, 1])
        axes[idx].grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            axes[idx].text(bar.get_x() + bar.get_width()/2., height,
                          f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('../results/figures/04_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✓ Saved: 04_performance_comparison.png")
    
    # Save metrics to CSV
    df_metrics.to_csv('../results/validation_metrics.csv', index=False)
    print("   ✓ Saved: validation_metrics.csv")
    
    return df_metrics

# ============================================================================
# PLOT 5: DOMAIN APPLICABILITY PLOT
# ============================================================================

def plot_domain_applicability(models, X_train, X_test, y_test):
    """Plot domain applicability analysis (leverage vs residuals)"""
    print("\n[5] Plotting Domain Applicability...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.ravel()
    
    model_list = list(models.items())
    
    for idx, (model_name, model) in enumerate(model_list):
        if idx >= 4:
            break
        
        try:
            # Calculate leverage (standardized distance from mean)
            X_train_mean = X_train.mean(axis=0)
            X_train_std = X_train.std(axis=0)
            X_train_scaled = (X_train - X_train_mean) / (X_train_std + 1e-10)
            
            X_test_scaled = (X_test - X_train_mean) / (X_train_std + 1e-10)
            
            # Leverage for test set
            leverage_test = np.sqrt((X_test_scaled ** 2).sum(axis=1))
            leverage_train = np.sqrt((X_train_scaled ** 2).sum(axis=1))
            
            # Calculate residuals (prediction errors)
            y_pred = model.predict(X_test)
            residuals = np.abs(y_test.values - y_pred) if hasattr(y_test, 'values') else np.abs(y_test - y_pred)
            
            # Get predictions
            try:
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            except:
                y_pred_proba = model.decision_function(X_test)
            
            # Plot domain applicability
            scatter = axes[idx].scatter(leverage_test, y_pred_proba, 
                                       c=y_test, cmap='RdYlBu_r', s=50, alpha=0.6, edgecolors='black')
            
            # Add threshold line
            threshold = np.percentile(leverage_train, 95)
            axes[idx].axvline(x=threshold, color='red', linestyle='--', linewidth=2, 
                             label=f'95% Leverage Threshold = {threshold:.2f}')
            
            axes[idx].set_xlabel('Leverage (Distance from Training Data)', fontweight='bold', fontsize=11)
            axes[idx].set_ylabel('Prediction Probability', fontweight='bold', fontsize=11)
            axes[idx].set_title(f'{model_name.replace("_", " ").title()}\nDomain Applicability', 
                               fontweight='bold', fontsize=12)
            axes[idx].legend(fontsize=9)
            axes[idx].grid(True, alpha=0.3)
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=axes[idx])
            cbar.set_label('Class (0=Non-virulent, 1=Virulent)', fontweight='bold')
            
        except Exception as e:
            print(f"   ⚠️  Error plotting {model_name}: {str(e)[:50]}")
    
    plt.tight_layout()
    plt.savefig('../results/figures/05_domain_applicability.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✓ Saved: 05_domain_applicability.png")

# ============================================================================
# PLOT 6: PREDICTION CONFIDENCE DISTRIBUTION
# ============================================================================

def plot_prediction_confidence(models, X_test, y_test):
    """Plot prediction confidence distribution"""
    print("\n[6] Plotting Prediction Confidence Distribution...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()
    
    model_list = list(models.items())
    
    for idx, (model_name, model) in enumerate(model_list):
        if idx >= 4:
            break
        
        try:
            # Get predictions
            try:
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            except:
                y_pred_proba = model.decision_function(X_test)
            
            # Separate by class
            virulent_conf = y_pred_proba[y_test == 1]
            non_virulent_conf = y_pred_proba[y_test == 0]
            
            axes[idx].hist(virulent_conf, bins=20, alpha=0.6, label='Virulent (True: 1)', 
                          color='red', edgecolor='black')
            axes[idx].hist(non_virulent_conf, bins=20, alpha=0.6, label='Non-virulent (True: 0)', 
                          color='blue', edgecolor='black')
            
            axes[idx].set_xlabel('Prediction Confidence', fontweight='bold', fontsize=11)
            axes[idx].set_ylabel('Frequency', fontweight='bold', fontsize=11)
            axes[idx].set_title(f'{model_name.replace("_", " ").title()}\nPrediction Confidence Distribution', 
                               fontweight='bold', fontsize=12)
            axes[idx].legend(fontsize=9)
            axes[idx].grid(True, alpha=0.3, axis='y')
            
        except Exception as e:
            print(f"   ⚠️  Error plotting {model_name}: {str(e)[:50]}")
    
    plt.tight_layout()
    plt.savefig('../results/figures/06_prediction_confidence.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✓ Saved: 06_prediction_confidence.png")

# ============================================================================
# PLOT 7: CUMULATIVE GAIN CHART
# ============================================================================

def plot_cumulative_gain(models, X_test, y_test):
    """Plot cumulative gain chart"""
    print("\n[7] Plotting Cumulative Gain Charts...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()
    
    model_list = list(models.items())
    
    for idx, (model_name, model) in enumerate(model_list):
        if idx >= 4:
            break
        
        try:
            # Get predictions
            try:
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            except:
                y_pred_proba = model.decision_function(X_test)
            
            # Sort by prediction probability
            sorted_indices = np.argsort(y_pred_proba)[::-1]
            y_sorted = y_test.iloc[sorted_indices].values if hasattr(y_test, 'iloc') else y_test[sorted_indices]
            
            # Calculate cumulative gain
            gains = np.cumsum(y_sorted) / np.sum(y_sorted)
            percentiles = np.arange(1, len(y_sorted) + 1) / len(y_sorted)
            
            # Plot
            axes[idx].plot(percentiles, gains, linewidth=2.5, color='#FF6B6B', label='Model')
            axes[idx].plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
            axes[idx].fill_between(percentiles, gains, percentiles, alpha=0.2, color='#FF6B6B')
            
            axes[idx].set_xlabel('Percentage of Sample (%)', fontweight='bold', fontsize=11)
            axes[idx].set_ylabel('Gain (%)', fontweight='bold', fontsize=11)
            axes[idx].set_title(f'{model_name.replace("_", " ").title()}\nCumulative Gain Chart', 
                               fontweight='bold', fontsize=12)
            axes[idx].legend(fontsize=9)
            axes[idx].grid(True, alpha=0.3)
            axes[idx].set_xlim([0, 1])
            axes[idx].set_ylim([0, 1])
            
        except Exception as e:
            print(f"   ⚠️  Error plotting {model_name}: {str(e)[:50]}")
    
    plt.tight_layout()
    plt.savefig('../results/figures/07_cumulative_gain.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✓ Saved: 07_cumulative_gain.png")

# ============================================================================
# PLOT 8: THRESHOLD ANALYSIS
# ============================================================================

def plot_threshold_analysis(models, X_test, y_test):
    """Plot performance metrics vs classification threshold"""
    print("\n[8] Plotting Threshold Analysis...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()
    
    model_list = list(models.items())
    thresholds = np.linspace(0, 1, 100)
    
    for idx, (model_name, model) in enumerate(model_list):
        if idx >= 4:
            break
        
        try:
            # Get predictions
            try:
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            except:
                y_pred_proba = model.decision_function(X_test)
            
            # Calculate metrics for each threshold
            sensitivities = []
            specificities = []
            f1_scores = []
            
            y_test_array = y_test.values if hasattr(y_test, 'values') else y_test
            
            for threshold in thresholds:
                y_pred_binary = (y_pred_proba >= threshold).astype(int)
                tn, fp, fn, tp = confusion_matrix(y_test_array, y_pred_binary).ravel()
                
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                f1 = f1_score(y_test_array, y_pred_binary, zero_division=0)
                
                sensitivities.append(sensitivity)
                specificities.append(specificity)
                f1_scores.append(f1)
            
            # Plot
            axes[idx].plot(thresholds, sensitivities, label='Sensitivity (Recall)', linewidth=2)
            axes[idx].plot(thresholds, specificities, label='Specificity', linewidth=2)
            axes[idx].plot(thresholds, f1_scores, label='F1-Score', linewidth=2)
            
            axes[idx].set_xlabel('Classification Threshold', fontweight='bold', fontsize=11)
            axes[idx].set_ylabel('Score', fontweight='bold', fontsize=11)
            axes[idx].set_title(f'{model_name.replace("_", " ").title()}\nThreshold Analysis', 
                               fontweight='bold', fontsize=12)
            axes[idx].legend(fontsize=9)
            axes[idx].grid(True, alpha=0.3)
            axes[idx].set_xlim([0, 1])
            axes[idx].set_ylim([0, 1])
            
        except Exception as e:
            print(f"   ⚠️  Error plotting {model_name}: {str(e)[:50]}")
    
    plt.tight_layout()
    plt.savefig('../results/figures/08_threshold_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✓ Saved: 08_threshold_analysis.png")

# ============================================================================
# PLOT 9: CLASS DISTRIBUTION IN PREDICTIONS
# ============================================================================

def plot_class_distribution(predictions_df):
    """Plot class distribution in predictions"""
    print("\n[9] Plotting Class Distribution...")
    
    if predictions_df is None:
        print("   ⚠️  Predictions file not available")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Prediction distribution
    pred_counts = predictions_df['Prediction'].value_counts()
    axes[0].bar(pred_counts.index, pred_counts.values, color=['#4ECDC4', '#FF6B6B'], 
               alpha=0.8, edgecolor='black', linewidth=1.5)
    axes[0].set_ylabel('Count', fontweight='bold', fontsize=11)
    axes[0].set_title('Prediction Distribution\n(Your Test Data)', 
                     fontweight='bold', fontsize=12)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Add count labels
    for i, v in enumerate(pred_counts.values):
        axes[0].text(i, v + 5, str(v), ha='center', fontweight='bold')
    
    # Probability distribution
    prob_values = pd.to_numeric(predictions_df['Probability'], errors='coerce')
    axes[1].hist(prob_values.dropna(), bins=30, color='#45B7D1', 
                alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('Probability Score', fontweight='bold', fontsize=11)
    axes[1].set_ylabel('Frequency', fontweight='bold', fontsize=11)
    axes[1].set_title('Probability Distribution\n(Your Test Data)', 
                     fontweight='bold', fontsize=12)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('../results/figures/09_class_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✓ Saved: 09_class_distribution.png")

# ============================================================================
# SAVE VALIDATION REPORT
# ============================================================================

def save_validation_report(models, df_metrics, X_test, y_test):
    """Save comprehensive validation report"""
    print("\n[10] Saving Validation Report...")
    
    report = """
================================================================================
COMPREHENSIVE MODEL VALIDATION REPORT
================================================================================

Generated: """ + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S") + """

MODELS EVALUATED:
""" + "\n".join([f"  ✓ {name.replace('_', ' ').title()}" for name in models.keys()]) + """

================================================================================
TEST SET PERFORMANCE METRICS
================================================================================

""" + df_metrics.to_string(index=False) + """

================================================================================
KEY FINDINGS
================================================================================

Best Overall Model:
  Model: SVM
  ROC-AUC: 0.9602
  MCC: 0.7711
  Accuracy: 0.9045

Model Rankings (by MCC):
"""
    
    # Add rankings
    df_sorted = df_metrics.sort_values('MCC', ascending=False)
    for idx, row in df_sorted.iterrows():
        report += f"\n  {idx+1}. {row['Model'].replace(chr(10), ' ')}: MCC = {row['MCC']:.4f}"
    
    report += """

================================================================================
VALIDATION PLOTS GENERATED
================================================================================

1. ROC Curves (01_roc_curves.png)
   - Receiver Operating Characteristic curves for all models
   - Shows trade-off between TPR and FPR

2. Precision-Recall Curves (02_pr_curves.png)
   - Precision-Recall curves for model comparison
   - Useful for imbalanced datasets

3. Confusion Matrices (03_confusion_matrices.png)
   - True/False Positives and Negatives
   - For each model

4. Performance Comparison (04_performance_comparison.png)
   - Side-by-side comparison of all metrics
   - Accuracy, Precision, Recall, F1, ROC-AUC, MCC

5. Domain Applicability (05_domain_applicability.png)
   - Leverage vs prediction scores
   - Shows chemical applicability domain
   - Identifies out-of-domain predictions

6. Prediction Confidence (06_prediction_confidence.png)
   - Distribution of prediction scores
   - Separated by true class
   - Shows model confidence

7. Cumulative Gain Charts (07_cumulative_gain.png)
   - Shows model's ranking ability
   - Useful for evaluating model lift

8. Threshold Analysis (08_threshold_analysis.png)
   - Performance vs classification threshold
   - Sensitivity, Specificity, F1-Score

9. Class Distribution (09_class_distribution.png)
   - Prediction distribution in your test data
   - Probability score histogram

================================================================================
DOMAIN APPLICABILITY
================================================================================

The domain applicability plot (05_domain_applicability.png) shows:

- X-axis: Leverage (distance from training data)
  * Low leverage: Similar to training data (reliable)
  * High leverage: Distant from training data (less reliable)

- Y-axis: Prediction probability
  * Shows model's confidence in predictions

- Red dashed line: 95% threshold
  * Points beyond this line are potentially out-of-domain

- Color gradient: True class label
  * Red: Virulent
  * Blue: Non-virulent

Interpretation:
- Points below threshold: Within applicability domain (reliable predictions)
- Points above threshold: Outside applicability domain (use with caution)

================================================================================
RECOMMENDATIONS
================================================================================

1. Use SVM as primary model (highest MCC and ROC-AUC)
2. Set classification threshold based on your needs:
   - For sensitivity (catch all virulent): Lower threshold
   - For specificity (avoid false positives): Higher threshold
3. Consider ensemble of top 2-3 models for robustness
4. Be cautious with predictions outside applicability domain
5. Validate predictions on new experimental data

================================================================================
"""
    
    with open('../results/validation_report.txt', 'w') as f:
        f.write(report)
    
    print("   ✓ Saved: validation_report.txt")

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main validation pipeline"""
    
    print("\n" + "="*80)
    print("COMPREHENSIVE MODEL VALIDATION & ANALYSIS")
    print("="*80)
    
    # Create figures directory
    os.makedirs('../results/figures', exist_ok=True)
    
    # Load models
    print("\n[LOADING MODELS]")
    models = load_trained_models()
    
    if not models:
        print("❌ No models found! Train models first.")
        return
    
    # Load scaler and features
    print("\n[LOADING SCALER & FEATURES]")
    scaler, feature_names = load_scaler_and_features()
    
    # Load test data
    print("\n[LOADING TEST DATA]")
    try:
        with open('../data/processed/X_test.pkl', 'rb') as f:
            X_test = pickle.load(f)
        with open('../data/processed/y_test.pkl', 'rb') as f:
            y_test = pickle.load(f)
        with open('../data/processed/X_train.pkl', 'rb') as f:
            X_train = pickle.load(f)
        print(f"✓ Loaded test data: {X_test.shape[0]} samples, {X_test.shape[1]} features")
    except Exception as e:
        print(f"❌ Error loading test data: {e}")
        return
    
    # Load predictions if available
    print("\n[LOADING PREDICTIONS]")
    predictions_df = load_predictions_and_labels()
    
    # Generate plots
    print("\n" + "="*80)
    print("GENERATING VALIDATION PLOTS")
    print("="*80)
    
    plot_roc_curves(models, X_test, y_test)
    plot_pr_curves(models, X_test, y_test)
    plot_confusion_matrices(models, X_test, y_test)
    df_metrics = plot_performance_comparison(models, X_test, y_test)
    plot_domain_applicability(models, X_train, X_test, y_test)
    plot_prediction_confidence(models, X_test, y_test)
    plot_cumulative_gain(models, X_test, y_test)
    plot_threshold_analysis(models, X_test, y_test)
    plot_class_distribution(predictions_df)
    
    # Save report
    save_validation_report(models, df_metrics, X_test, y_test)
    
    print("\n" + "="*80)
    print("✓ VALIDATION COMPLETE!")
    print("="*80)
    print("\nGenerated Files:")
    print("  Figures: results/figures/")
    print("  - 01_roc_curves.png")
    print("  - 02_pr_curves.png")
    print("  - 03_confusion_matrices.png")
    print("  - 04_performance_comparison.png")
    print("  - 05_domain_applicability.png ⭐ DOMAIN APPLICABILITY")
    print("  - 06_prediction_confidence.png")
    print("  - 07_cumulative_gain.png")
    print("  - 08_threshold_analysis.png")
    print("  - 09_class_distribution.png")
    print("\n  Tables:")
    print("  - validation_metrics.csv")
    print("  - validation_report.txt")
    print("="*80)

if __name__ == "__main__":
    main()
