"""
Model Evaluation and Visualization Module - WINDOWS FIXED VERSION
Creates publication-quality plots including:
- Combined ROC-AUC curves (MAIN FIGURE)
- Precision-Recall curves
- Confusion matrices
- Feature importance
- Learning curves
- Model comparison charts
"""

# FIX FOR WINDOWS - Add this BEFORE importing matplotlib
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (roc_curve, auc, precision_recall_curve, 
                             confusion_matrix, ConfusionMatrixDisplay)
from sklearn.model_selection import learning_curve
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality style
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

class ModelEvaluator:
    """Create comprehensive evaluation plots"""
    
    def __init__(self):
        self.models = {}
        self.predictions = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # Color scheme for models
        self.colors = {
            'Random Forest': '#e74c3c',
            'XGBoost': '#3498db',
            'SVM': '#2ecc71',
            'Logistic Regression': '#f39c12'
        }
    
    def load_models_and_data(self):
        """Load trained models and data"""
        print("\n" + "="*80)
        print("LOADING MODELS AND DATA")
        print("="*80)
        
        # Load models
        model_names = ['random_forest', 'xgboost', 'svm', 'logistic_regression']
        display_names = ['Random Forest', 'XGBoost', 'SVM', 'Logistic Regression']
        
        for model_name, display_name in zip(model_names, display_names):
            self.models[display_name] = joblib.load(f'../models/{model_name}.pkl')
            print(f"✓ Loaded {display_name}")
        
        # Load predictions
        self.predictions = joblib.load('../models/predictions.pkl')
        
        # Load data
        self.X_train = pd.read_csv('../data/processed/X_train.csv')
        self.X_test = pd.read_csv('../data/processed/X_test.csv')
        self.y_train = pd.read_csv('../data/processed/y_train.csv')['Virulence']
        self.y_test = pd.read_csv('../data/processed/y_test.csv')['Virulence']
        
        print("\n✓ All models and data loaded")
        return self
    
    def plot_combined_roc_curves(self):
        """
        ⭐ MAIN FIGURE FOR PUBLICATION ⭐
        Combined ROC curves for all 4 models
        """
        print("\n" + "="*80)
        print("CREATING COMBINED ROC-AUC CURVES (MAIN FIGURE)")
        print("="*80)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot each model
        for name in self.models.keys():
            # Get true labels and predicted probabilities
            y_true = self.predictions['y_test']
            y_proba = self.predictions[f'{name}_test_proba']
            
            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            roc_auc = auc(fpr, tpr)
            
            # Plot
            ax.plot(fpr, tpr, 
                   color=self.colors[name],
                   lw=2.5,
                   label=f'{name} (AUC = {roc_auc:.3f})')
            
            print(f"✓ {name}: AUC = {roc_auc:.4f}")
        
        # Plot diagonal (random classifier)
        ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier (AUC = 0.500)')
        
        # Styling
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=13, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=13, fontweight='bold')
        ax.set_title('ROC Curves - Virulence Protein Classification', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('../results/figures/04_combined_roc_curves.png', dpi=300, bbox_inches='tight')
        plt.close()  # Important: close figure to free memory
        print("\n✓ Saved: 04_combined_roc_curves.png")
        print("  ⭐ USE THIS AS YOUR MAIN FIGURE FOR PUBLICATION ⭐")
        
        return self
    
    def plot_precision_recall_curves(self):
        """Precision-Recall curves for all models"""
        print("\n" + "="*80)
        print("CREATING PRECISION-RECALL CURVES")
        print("="*80)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for name in self.models.keys():
            y_true = self.predictions['y_test']
            y_proba = self.predictions[f'{name}_test_proba']
            
            precision, recall, _ = precision_recall_curve(y_true, y_proba)
            pr_auc = auc(recall, precision)
            
            ax.plot(recall, precision,
                   color=self.colors[name],
                   lw=2.5,
                   label=f'{name} (AUC = {pr_auc:.3f})')
        
        # Baseline
        baseline = sum(self.y_test) / len(self.y_test)
        ax.plot([0, 1], [baseline, baseline], 'k--', lw=2, 
               label=f'Baseline (Prevalence = {baseline:.3f})')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall', fontsize=13, fontweight='bold')
        ax.set_ylabel('Precision', fontsize=13, fontweight='bold')
        ax.set_title('Precision-Recall Curves - Virulence Protein Classification',
                    fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='lower left', frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('../results/figures/05_precision_recall_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: 05_precision_recall_curves.png")
        
        return self
    
    def plot_confusion_matrices(self):
        """Confusion matrices for all models"""
        print("\n" + "="*80)
        print("CREATING CONFUSION MATRICES")
        print("="*80)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.ravel()
        
        for idx, name in enumerate(self.models.keys()):
            y_true = self.predictions['y_test']
            y_pred = self.predictions[f'{name}_test_pred']
            
            cm = confusion_matrix(y_true, y_pred)
            
            # Create display
            disp = ConfusionMatrixDisplay(
                confusion_matrix=cm,
                display_labels=['Non-Virulent', 'Virulent']
            )
            
            disp.plot(ax=axes[idx], cmap='Blues', values_format='d')
            axes[idx].set_title(f'{name}', fontsize=13, fontweight='bold')
            axes[idx].grid(False)
        
        plt.suptitle('Confusion Matrices - Test Set', 
                    fontsize=15, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig('../results/figures/06_confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: 06_confusion_matrices.png")
        
        return self
    
    def plot_feature_importance(self):
        """Feature importance for tree-based models"""
        print("\n" + "="*80)
        print("CREATING FEATURE IMPORTANCE PLOTS")
        print("="*80)
        
        # Get feature names
        feature_names = self.X_train.columns
        
        # Plot for Random Forest and XGBoost
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        for idx, name in enumerate(['Random Forest', 'XGBoost']):
            model = self.models[name]
            
            # Get feature importances
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                
                # Get top 20 features
                indices = np.argsort(importances)[-20:]
                top_features = [feature_names[i] for i in indices]
                top_importances = importances[indices]
                
                # Plot
                axes[idx].barh(range(len(top_importances)), top_importances,
                              color=self.colors[name], alpha=0.8)
                axes[idx].set_yticks(range(len(top_importances)))
                axes[idx].set_yticklabels(top_features, fontsize=9)
                axes[idx].set_xlabel('Importance', fontsize=12, fontweight='bold')
                axes[idx].set_title(f'{name} - Top 20 Features',
                                  fontsize=13, fontweight='bold')
                axes[idx].grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('../results/figures/07_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: 07_feature_importance.png")
        
        # Save feature importance to CSV
        for name in ['Random Forest', 'XGBoost']:
            model = self.models[name]
            if hasattr(model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                filename = name.replace(' ', '_').lower()
                importance_df.to_csv(f'../results/tables/feature_importance_{filename}.csv', index=False)
        
        print("✓ Saved feature importance tables")
        
        return self
    
    def plot_learning_curves(self):
        """Learning curves to check for overfitting"""
        print("\n" + "="*80)
        print("CREATING LEARNING CURVES")
        print("="*80)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.ravel()
        
        train_sizes = np.linspace(0.1, 1.0, 10)
        
        for idx, name in enumerate(self.models.keys()):
            print(f"  Computing learning curve for {name}...")
            model = self.models[name]
            
            train_sizes_abs, train_scores, val_scores = learning_curve(
                model, self.X_train, self.y_train,
                train_sizes=train_sizes,
                cv=5,
                scoring='roc_auc',
                n_jobs=-1,
                random_state=42
            )
            
            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            val_mean = np.mean(val_scores, axis=1)
            val_std = np.std(val_scores, axis=1)
            
            axes[idx].plot(train_sizes_abs, train_mean, 
                          label='Training Score',
                          color=self.colors[name], lw=2)
            axes[idx].fill_between(train_sizes_abs, 
                                  train_mean - train_std,
                                  train_mean + train_std,
                                  alpha=0.2, color=self.colors[name])
            
            axes[idx].plot(train_sizes_abs, val_mean,
                          label='Cross-Validation Score',
                          color=self.colors[name], lw=2, linestyle='--')
            axes[idx].fill_between(train_sizes_abs,
                                  val_mean - val_std,
                                  val_mean + val_std,
                                  alpha=0.2, color=self.colors[name])
            
            axes[idx].set_xlabel('Training Examples', fontsize=11, fontweight='bold')
            axes[idx].set_ylabel('ROC-AUC Score', fontsize=11, fontweight='bold')
            axes[idx].set_title(f'{name}', fontsize=12, fontweight='bold')
            axes[idx].legend(loc='lower right')
            axes[idx].grid(True, alpha=0.3)
        
        plt.suptitle('Learning Curves - Model Validation',
                    fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig('../results/figures/08_learning_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: 08_learning_curves.png")
        
        return self
    
    def plot_model_comparison(self):
        """Bar chart comparing all models"""
        print("\n" + "="*80)
        print("CREATING MODEL COMPARISON CHART")
        print("="*80)
        
        # Load results
        results = pd.read_csv('../results/tables/validation_results.csv')
        test_results = results[results['Dataset'] == 'Test']
        
        # Prepare data
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        models = test_results['Model'].tolist()
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 7))
        
        x = np.arange(len(models))
        width = 0.15
        
        for idx, metric in enumerate(metrics):
            values = test_results[metric].tolist()
            offset = width * (idx - len(metrics)/2 + 0.5)
            ax.bar(x + offset, values, width, 
                  label=metric, alpha=0.8)
        
        ax.set_xlabel('Models', fontsize=13, fontweight='bold')
        ax.set_ylabel('Score', fontsize=13, fontweight='bold')
        ax.set_title('Model Performance Comparison - Test Set',
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=15, ha='right')
        ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True)
        ax.set_ylim([0, 1.05])
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', fontsize=8, padding=3)
        
        plt.tight_layout()
        plt.savefig('../results/figures/09_model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: 09_model_comparison.png")
        
        return self
    
    def create_summary_table(self):
        """Create final summary comparison table"""
        print("\n" + "="*80)
        print("CREATING SUMMARY TABLE")
        print("="*80)
        
        results = pd.read_csv('../results/tables/validation_results.csv')
        
        # Separate by dataset
        val_results = results[results['Dataset'] == 'Validation'].copy()
        test_results = results[results['Dataset'] == 'Test'].copy()
        
        # Merge
        comparison = pd.merge(
            val_results[['Model', 'ROC-AUC', 'F1-Score', 'Accuracy']],
            test_results[['Model', 'ROC-AUC', 'F1-Score', 'Accuracy']],
            on='Model',
            suffixes=('_Val', '_Test')
        )
        
        # Sort by test ROC-AUC
        comparison = comparison.sort_values('ROC-AUC_Test', ascending=False)
        
        # Save
        comparison.to_csv('../results/tables/final_comparison.csv', index=False)
        print("✓ Saved: final_comparison.csv")
        
        # Display
        print("\nFINAL MODEL COMPARISON:")
        print(comparison.to_string(index=False))
        
        return self


def main():
    """Main evaluation pipeline"""
    print("="*80)
    print("MODEL EVALUATION AND VISUALIZATION")
    print("="*80)
    
    evaluator = ModelEvaluator()
    
    evaluator.load_models_and_data() \
             .plot_combined_roc_curves() \
             .plot_precision_recall_curves() \
             .plot_confusion_matrices() \
             .plot_feature_importance() \
             .plot_learning_curves() \
             .plot_model_comparison() \
             .create_summary_table()
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE!")
    print("="*80)
    print("\n✓ All publication-quality figures created")
    print("  Location: results/figures/")
    print("\n⭐ MAIN FIGURE: 04_combined_roc_curves.png")


if __name__ == "__main__":
    main()
