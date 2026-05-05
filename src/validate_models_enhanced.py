"""
Enhanced Model Validation Module
Includes:
- Y-Randomization Test
- Cross-Validation (5-fold and 10-fold)
- Domain Applicability Analysis
- Before/After SMOTE Comparison
- Enhanced ROC curves with MCC
- Statistical significance testing
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    accuracy_score, precision_score, recall_score, f1_score, 
    matthews_corrcoef, confusion_matrix, make_scorer
)
from scipy.stats import ttest_rel
from scipy.spatial.distance import euclidean
import xgboost as xgb
import joblib
import warnings
import os
warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

class EnhancedModelValidator:
    """Comprehensive model validation with multiple techniques"""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.cv_results = {}
        self.y_random_results = {}
        self.X_train_original = None
        self.y_train_original = None
        self.X_train_smote = None
        self.y_train_smote = None
        self.X_test = None
        self.y_test = None
        self.scaler = None
        
        # Create output directories
        os.makedirs('../results/validation', exist_ok=True)
        os.makedirs('../results/figures', exist_ok=True)
        
    def load_data(self):
        """Load all datasets including pre-SMOTE data"""
        print("\n" + "="*80)
        print("LOADING DATA FOR VALIDATION")
        print("="*80)
        
        # Load original features before SMOTE
        print("\nLoading original data (before SMOTE)...")
        features_df = pd.read_csv('../data/processed/protein_features.csv')
        X_full = features_df.drop(['Protein_ID', 'Virulence'], axis=1)
        y_full = features_df['Virulence']
        
        # Split to get original training set
        from sklearn.model_selection import train_test_split
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            X_full, y_full, test_size=0.2, random_state=42, stratify=y_full
        )
        
        self.X_train_original, _, self.y_train_original, _ = train_test_split(
            X_temp, y_temp, test_size=0.125, random_state=42, stratify=y_temp
        )
        
        # Scale original data
        self.scaler = StandardScaler()
        self.X_train_original = pd.DataFrame(
            self.scaler.fit_transform(self.X_train_original),
            columns=X_full.columns
        )
        self.X_test = pd.DataFrame(
            self.scaler.transform(self.X_test),
            columns=X_full.columns
        )
        
        print(f"✓ Original training set: {self.X_train_original.shape}")
        print(f"  Class distribution: Virulent={sum(self.y_train_original==1)}, "
              f"Non-virulent={sum(self.y_train_original==0)}")
        
        # Load SMOTE data
        print("\nLoading SMOTE-balanced data...")
        self.X_train_smote = pd.read_csv('../data/processed/X_train.csv')
        self.y_train_smote = pd.read_csv('../data/processed/y_train.csv')['Virulence']
        
        print(f"✓ SMOTE training set: {self.X_train_smote.shape}")
        print(f"  Class distribution: Virulent={sum(self.y_train_smote==1)}, "
              f"Non-virulent={sum(self.y_train_smote==0)}")
        
        print(f"\n✓ Test set: {self.X_test.shape}")
        print(f"  Class distribution: Virulent={sum(self.y_test==1)}, "
              f"Non-virulent={sum(self.y_test==0)}")
        
        return self
    
    def initialize_models(self):
        """Initialize models with same parameters as training"""
        print("\n" + "="*80)
        print("INITIALIZING MODELS")
        print("="*80)
        
        self.models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=200, max_depth=20, min_samples_split=5,
                min_samples_leaf=2, max_features='sqrt', random_state=42,
                n_jobs=-1, class_weight='balanced'
            ),
            'XGBoost': xgb.XGBClassifier(
                n_estimators=200, max_depth=8, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8, random_state=42,
                n_jobs=-1, scale_pos_weight=1
            ),
            'SVM': SVC(
                C=1.0, kernel='rbf', gamma='scale', random_state=42,
                class_weight='balanced', probability=True
            ),
            'Logistic Regression': LogisticRegression(
                C=1.0, penalty='l2', solver='lbfgs', max_iter=1000,
                random_state=42, class_weight='balanced', n_jobs=-1
            )
        }
        
        print(f"\n✓ Initialized {len(self.models)} models")
        return self
    
    def compare_before_after_smote(self):
        """Compare model performance before and after SMOTE"""
        print("\n" + "="*80)
        print("COMPARING PERFORMANCE: BEFORE vs AFTER SMOTE")
        print("="*80)
        
        comparison_results = []
        
        for name, model in self.models.items():
            print(f"\nTesting {name}...")
            
            # Train on original data (before SMOTE)
            model_before = type(model)(**model.get_params())
            model_before.fit(self.X_train_original, self.y_train_original)
            
            y_pred_before = model_before.predict(self.X_test)
            y_proba_before = model_before.predict_proba(self.X_test)[:, 1]
            
            # Train on SMOTE data
            model_after = type(model)(**model.get_params())
            model_after.fit(self.X_train_smote, self.y_train_smote)
            
            y_pred_after = model_after.predict(self.X_test)
            y_proba_after = model_after.predict_proba(self.X_test)[:, 1]
            
            # Calculate metrics
            metrics_before = {
                'Model': name,
                'Dataset': 'Before SMOTE',
                'Accuracy': accuracy_score(self.y_test, y_pred_before),
                'Precision': precision_score(self.y_test, y_pred_before, zero_division=0),
                'Recall': recall_score(self.y_test, y_pred_before, zero_division=0),
                'F1-Score': f1_score(self.y_test, y_pred_before, zero_division=0),
                'MCC': matthews_corrcoef(self.y_test, y_pred_before),
                'ROC-AUC': auc(*roc_curve(self.y_test, y_proba_before)[:2])
            }
            
            metrics_after = {
                'Model': name,
                'Dataset': 'After SMOTE',
                'Accuracy': accuracy_score(self.y_test, y_pred_after),
                'Precision': precision_score(self.y_test, y_pred_after, zero_division=0),
                'Recall': recall_score(self.y_test, y_pred_after, zero_division=0),
                'F1-Score': f1_score(self.y_test, y_pred_after, zero_division=0),
                'MCC': matthews_corrcoef(self.y_test, y_pred_after),
                'ROC-AUC': auc(*roc_curve(self.y_test, y_proba_after)[:2])
            }
            
            comparison_results.extend([metrics_before, metrics_after])
            
            # Store for plotting
            self.results[name] = {
                'before_proba': y_proba_before,
                'after_proba': y_proba_after,
                'before_pred': y_pred_before,
                'after_pred': y_pred_after
            }
        
        # Convert to DataFrame
        comparison_df = pd.DataFrame(comparison_results)
        
        # Display results
        print("\n" + "-"*80)
        print("COMPARISON RESULTS:")
        print("-"*80)
        print(comparison_df.to_string(index=False))
        
        # Save results
        comparison_df.to_csv('../results/validation/before_after_smote_comparison.csv', index=False)
        print("\n✓ Saved comparison to: before_after_smote_comparison.csv")
        
        # Plot comparison
        self._plot_smote_comparison(comparison_df)
        
        return self
    
    def _plot_smote_comparison(self, comparison_df):
        """Plot before/after SMOTE comparison"""
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'MCC', 'ROC-AUC']
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            
            # Prepare data for plotting
            before_data = comparison_df[comparison_df['Dataset'] == 'Before SMOTE']
            after_data = comparison_df[comparison_df['Dataset'] == 'After SMOTE']
            
            x = np.arange(len(before_data))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, before_data[metric].values, width, 
                          label='Before SMOTE', color='#e74c3c', alpha=0.8)
            bars2 = ax.bar(x + width/2, after_data[metric].values, width,
                          label='After SMOTE', color='#2ecc71', alpha=0.8)
            
            ax.set_xlabel('Model')
            ax.set_ylabel(metric)
            ax.set_title(f'{metric} Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels(before_data['Model'].values, rotation=45, ha='right')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for bar in bars1 + bars2:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig('../results/figures/10_before_after_smote_comparison.png', 
                   dpi=300, bbox_inches='tight')
        print("✓ Saved: 10_before_after_smote_comparison.png")
        plt.close()
    
    def cross_validation_analysis(self, cv_folds=[5, 10]):
        """Perform k-fold cross-validation"""
        print("\n" + "="*80)
        print("CROSS-VALIDATION ANALYSIS")
        print("="*80)
        
        # Define scoring metrics
        scoring = {
            'accuracy': 'accuracy',
            'precision': 'precision',
            'recall': 'recall',
            'f1': 'f1',
            'roc_auc': 'roc_auc',
            'mcc': make_scorer(matthews_corrcoef)
        }
        
        cv_summary = []
        
        for n_folds in cv_folds:
            print(f"\n{n_folds}-Fold Cross-Validation:")
            print("-" * 60)
            
            cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
            
            for name, model in self.models.items():
                print(f"  Validating {name}...")
                
                # Perform cross-validation
                cv_results = cross_validate(
                    model, self.X_train_smote, self.y_train_smote,
                    cv=cv, scoring=scoring, n_jobs=-1, return_train_score=True
                )
                
                # Store results
                for metric in scoring.keys():
                    cv_summary.append({
                        'Model': name,
                        'CV_Folds': n_folds,
                        'Metric': metric.upper(),
                        'Mean_Train': cv_results[f'train_{metric}'].mean(),
                        'Std_Train': cv_results[f'train_{metric}'].std(),
                        'Mean_Test': cv_results[f'test_{metric}'].mean(),
                        'Std_Test': cv_results[f'test_{metric}'].std()
                    })
                
                # Store for detailed analysis
                if name not in self.cv_results:
                    self.cv_results[name] = {}
                self.cv_results[name][f'{n_folds}fold'] = cv_results
        
        # Convert to DataFrame
        cv_df = pd.DataFrame(cv_summary)
        
        # Display results
        print("\n" + "="*80)
        print("CROSS-VALIDATION SUMMARY")
        print("="*80)
        
        for n_folds in cv_folds:
            print(f"\n{n_folds}-FOLD CV RESULTS:")
            print("-" * 80)
            fold_results = cv_df[cv_df['CV_Folds'] == n_folds]
            display_df = fold_results.pivot_table(
                index='Model', columns='Metric', 
                values=['Mean_Test', 'Std_Test']
            )
            print(display_df.to_string())
        
        # Save results
        cv_df.to_csv('../results/validation/cross_validation_results.csv', index=False)
        print("\n✓ Saved: cross_validation_results.csv")
        
        # Plot CV results
        self._plot_cv_results(cv_df)
        
        return self
    
    def _plot_cv_results(self, cv_df):
        """Plot cross-validation results"""
        metrics = ['ACCURACY', 'PRECISION', 'RECALL', 'F1', 'ROC_AUC', 'MCC']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            
            metric_data = cv_df[cv_df['Metric'] == metric]
            
            # Group by model and CV folds
            for model_name in metric_data['Model'].unique():
                model_data = metric_data[metric_data['Model'] == model_name]
                
                x = model_data['CV_Folds'].values
                y = model_data['Mean_Test'].values
                yerr = model_data['Std_Test'].values
                
                ax.errorbar(x, y, yerr=yerr, marker='o', capsize=5, 
                           label=model_name, linewidth=2, markersize=8)
            
            ax.set_xlabel('Number of Folds', fontsize=12)
            ax.set_ylabel(f'{metric} Score', fontsize=12)
            ax.set_title(f'Cross-Validation: {metric}', fontsize=14, fontweight='bold')
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_xticks([5, 10])
        
        plt.tight_layout()
        plt.savefig('../results/figures/11_cross_validation_results.png', 
                   dpi=300, bbox_inches='tight')
        print("✓ Saved: 11_cross_validation_results.png")
        plt.close()
    
    def y_randomization_test(self, n_iterations=50):
        """Y-randomization test to check for overfitting"""
        print("\n" + "="*80)
        print("Y-RANDOMIZATION TEST")
        print("="*80)
        print(f"Running {n_iterations} randomization iterations per model...")
        print("(This may take several minutes)")
        
        randomization_results = []
        
        for name, model in self.models.items():
            print(f"\nTesting {name}...")
            
            # Train on real data
            model_real = type(model)(**model.get_params())
            model_real.fit(self.X_train_smote, self.y_train_smote)
            y_pred_real = model_real.predict(self.X_test)
            
            real_metrics = {
                'accuracy': accuracy_score(self.y_test, y_pred_real),
                'precision': precision_score(self.y_test, y_pred_real, zero_division=0),
                'recall': recall_score(self.y_test, y_pred_real, zero_division=0),
                'f1': f1_score(self.y_test, y_pred_real, zero_division=0),
                'mcc': matthews_corrcoef(self.y_test, y_pred_real)
            }
            
            # Randomization iterations
            random_metrics = {metric: [] for metric in real_metrics.keys()}
            
            for i in range(n_iterations):
                if (i + 1) % 10 == 0:
                    print(f"  Iteration {i+1}/{n_iterations}")
                
                # Randomly shuffle labels
                y_random = self.y_train_smote.sample(frac=1, random_state=i).values
                
                # Train model on randomized data
                model_random = type(model)(**model.get_params())
                model_random.fit(self.X_train_smote, y_random)
                y_pred_random = model_random.predict(self.X_test)
                
                # Calculate metrics
                random_metrics['accuracy'].append(
                    accuracy_score(self.y_test, y_pred_random))
                random_metrics['precision'].append(
                    precision_score(self.y_test, y_pred_random, zero_division=0))
                random_metrics['recall'].append(
                    recall_score(self.y_test, y_pred_random, zero_division=0))
                random_metrics['f1'].append(
                    f1_score(self.y_test, y_pred_random, zero_division=0))
                random_metrics['mcc'].append(
                    matthews_corrcoef(self.y_test, y_pred_random))
            
            # Store results
            for metric in real_metrics.keys():
                randomization_results.append({
                    'Model': name,
                    'Metric': metric.upper(),
                    'Real_Score': real_metrics[metric],
                    'Random_Mean': np.mean(random_metrics[metric]),
                    'Random_Std': np.std(random_metrics[metric]),
                    'Random_Max': np.max(random_metrics[metric]),
                    'Difference': real_metrics[metric] - np.mean(random_metrics[metric])
                })
            
            # Store for plotting
            self.y_random_results[name] = {
                'real': real_metrics,
                'random': random_metrics
            }
        
        # Convert to DataFrame
        y_random_df = pd.DataFrame(randomization_results)
        
        # Display results
        print("\n" + "="*80)
        print("Y-RANDOMIZATION RESULTS")
        print("="*80)
        print(y_random_df.to_string(index=False))
        
        # Save results
        y_random_df.to_csv('../results/validation/y_randomization_results.csv', index=False)
        print("\n✓ Saved: y_randomization_results.csv")
        
        # Plot results
        self._plot_y_randomization(y_random_df)
        
        return self
    
    def _plot_y_randomization(self, y_random_df):
        """Plot Y-randomization results"""
        metrics = ['ACCURACY', 'PRECISION', 'RECALL', 'F1', 'MCC']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            
            metric_data = y_random_df[y_random_df['Metric'] == metric]
            
            models = metric_data['Model'].values
            x = np.arange(len(models))
            
            # Plot real scores
            real_scores = metric_data['Real_Score'].values
            random_means = metric_data['Random_Mean'].values
            random_stds = metric_data['Random_Std'].values
            
            width = 0.35
            bars1 = ax.bar(x - width/2, real_scores, width, label='Real Data',
                          color='#2ecc71', alpha=0.8)
            bars2 = ax.bar(x + width/2, random_means, width, yerr=random_stds,
                          label='Randomized (Mean±SD)', color='#e74c3c', alpha=0.8,
                          capsize=5)
            
            ax.set_xlabel('Model', fontsize=12)
            ax.set_ylabel(f'{metric} Score', fontsize=12)
            ax.set_title(f'Y-Randomization: {metric}', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(models, rotation=45, ha='right')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            
            # Add significance stars
            for i, (real, rand_mean, rand_std) in enumerate(zip(real_scores, random_means, random_stds)):
                if real > rand_mean + 2*rand_std:
                    ax.text(i, real + 0.02, '***', ha='center', fontsize=16, color='green')
                elif real > rand_mean + rand_std:
                    ax.text(i, real + 0.02, '**', ha='center', fontsize=16, color='orange')
        
        # Remove extra subplot
        fig.delaxes(axes[5])
        
        plt.tight_layout()
        plt.savefig('../results/figures/12_y_randomization_test.png', 
                   dpi=300, bbox_inches='tight')
        print("✓ Saved: 12_y_randomization_test.png")
        plt.close()
    
    def applicability_domain_analysis(self):
        """Analyze applicability domain using distance-based approach"""
        print("\n" + "="*80)
        print("APPLICABILITY DOMAIN ANALYSIS")
        print("="*80)
        
        # Calculate centroid of training data
        train_centroid = self.X_train_smote.mean(axis=0).values
        
        # Calculate distances from centroid for training data
        train_distances = np.array([
            euclidean(row, train_centroid) 
            for row in self.X_train_smote.values
        ])
        
        # Calculate distances for test data
        test_distances = np.array([
            euclidean(row, train_centroid) 
            for row in self.X_test.values
        ])
        
        # Define threshold (e.g., 95th percentile of training distances)
        threshold = np.percentile(train_distances, 95)
        
        # Identify test samples within applicability domain
        within_domain = test_distances <= threshold
        outside_domain = ~within_domain
        
        print(f"\nApplicability Domain Analysis:")
        print(f"  Training set size: {len(train_distances)}")
        print(f"  Test set size: {len(test_distances)}")
        print(f"  Distance threshold (95th percentile): {threshold:.4f}")
        print(f"  Test samples within domain: {sum(within_domain)} ({100*sum(within_domain)/len(test_distances):.1f}%)")
        print(f"  Test samples outside domain: {sum(outside_domain)} ({100*sum(outside_domain)/len(test_distances):.1f}%)")
        
        # Analyze prediction performance within and outside domain
        print("\n" + "-"*80)
        print("Performance Analysis by Domain:")
        print("-"*80)
        
        domain_results = []
        
        # Load trained models
        for name in self.models.keys():
            filename = name.replace(' ', '_').lower()
            try:
                model = joblib.load(f'../models/{filename}.pkl')
                
                y_pred = model.predict(self.X_test)
                
                # Within domain performance
                if sum(within_domain) > 0:
                    y_test_within = self.y_test[within_domain]
                    y_pred_within = y_pred[within_domain]
                    
                    within_metrics = {
                        'Model': name,
                        'Domain': 'Within',
                        'N_samples': sum(within_domain),
                        'Accuracy': accuracy_score(y_test_within, y_pred_within),
                        'Precision': precision_score(y_test_within, y_pred_within, zero_division=0),
                        'Recall': recall_score(y_test_within, y_pred_within, zero_division=0),
                        'F1-Score': f1_score(y_test_within, y_pred_within, zero_division=0),
                        'MCC': matthews_corrcoef(y_test_within, y_pred_within)
                    }
                    domain_results.append(within_metrics)
                
                # Outside domain performance
                if sum(outside_domain) > 0:
                    y_test_outside = self.y_test[outside_domain]
                    y_pred_outside = y_pred[outside_domain]
                    
                    outside_metrics = {
                        'Model': name,
                        'Domain': 'Outside',
                        'N_samples': sum(outside_domain),
                        'Accuracy': accuracy_score(y_test_outside, y_pred_outside),
                        'Precision': precision_score(y_test_outside, y_pred_outside, zero_division=0),
                        'Recall': recall_score(y_test_outside, y_pred_outside, zero_division=0),
                        'F1-Score': f1_score(y_test_outside, y_pred_outside, zero_division=0),
                        'MCC': matthews_corrcoef(y_test_outside, y_pred_outside)
                    }
                    domain_results.append(outside_metrics)
                    
            except Exception as e:
                print(f"  Warning: Could not load {name}: {e}")
        
        # Convert to DataFrame
        domain_df = pd.DataFrame(domain_results)
        
        if len(domain_df) > 0:
            print("\n" + domain_df.to_string(index=False))
            
            # Save results
            domain_df.to_csv('../results/validation/applicability_domain_analysis.csv', index=False)
            print("\n✓ Saved: applicability_domain_analysis.csv")
        
        # Plot applicability domain
        self._plot_applicability_domain(train_distances, test_distances, 
                                       within_domain, threshold)
        
        # Save distance information
        distance_df = pd.DataFrame({
            'Sample_Type': ['Train']*len(train_distances) + ['Test']*len(test_distances),
            'Distance': np.concatenate([train_distances, test_distances]),
            'Within_Domain': [True]*len(train_distances) + within_domain.tolist()
        })
        distance_df.to_csv('../results/validation/domain_distances.csv', index=False)
        
        return self
    
    def _plot_applicability_domain(self, train_distances, test_distances, 
                                   within_domain, threshold):
        """Plot applicability domain visualization"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Distance distributions
        ax = axes[0]
        ax.hist(train_distances, bins=50, alpha=0.6, label='Training Set',
               color='#3498db', density=True)
        ax.hist(test_distances[within_domain], bins=30, alpha=0.6,
               label='Test (Within Domain)', color='#2ecc71', density=True)
        ax.hist(test_distances[~within_domain], bins=20, alpha=0.6,
               label='Test (Outside Domain)', color='#e74c3c', density=True)
        ax.axvline(threshold, color='red', linestyle='--', linewidth=2,
                  label=f'Threshold ({threshold:.2f})')
        ax.set_xlabel('Distance from Training Centroid', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title('Applicability Domain Analysis', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Box plot comparison
        ax = axes[1]
        data_to_plot = [train_distances, test_distances[within_domain], 
                       test_distances[~within_domain]]
        positions = [1, 2, 3]
        colors = ['#3498db', '#2ecc71', '#e74c3c']
        
        bp = ax.boxplot(data_to_plot, positions=positions, widths=0.6,
                       patch_artist=True, showmeans=True)
        
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        
        ax.axhline(threshold, color='red', linestyle='--', linewidth=2,
                  label=f'Threshold')
        ax.set_xticklabels(['Training\nSet', 'Test\n(Within)', 'Test\n(Outside)'])
        ax.set_ylabel('Distance from Training Centroid', fontsize=12)
        ax.set_title('Distance Distribution Comparison', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('../results/figures/13_applicability_domain.png', 
                   dpi=300, bbox_inches='tight')
        print("✓ Saved: 13_applicability_domain.png")
        plt.close()
    
    def plot_combined_roc_with_mcc(self):
        """Create combined ROC curve plot with MCC information"""
        print("\n" + "="*80)
        print("GENERATING COMBINED ROC CURVES")
        print("="*80)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
        
        # Plot 1: Before SMOTE
        ax = axes[0]
        for (name, model), color in zip(self.models.items(), colors):
            fpr, tpr, _ = roc_curve(self.y_test, self.results[name]['before_proba'])
            roc_auc = auc(fpr, tpr)
            mcc = matthews_corrcoef(self.y_test, self.results[name]['before_pred'])
            
            ax.plot(fpr, tpr, color=color, lw=2.5, 
                   label=f'{name} (AUC={roc_auc:.3f}, MCC={mcc:.3f})')
        
        ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curves - Before SMOTE', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Plot 2: After SMOTE
        ax = axes[1]
        for (name, model), color in zip(self.models.items(), colors):
            fpr, tpr, _ = roc_curve(self.y_test, self.results[name]['after_proba'])
            roc_auc = auc(fpr, tpr)
            mcc = matthews_corrcoef(self.y_test, self.results[name]['after_pred'])
            
            ax.plot(fpr, tpr, color=color, lw=2.5,
                   label=f'{name} (AUC={roc_auc:.3f}, MCC={mcc:.3f})')
        
        ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curves - After SMOTE', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('../results/figures/14_combined_roc_curves_with_mcc.png',
                   dpi=300, bbox_inches='tight')
        print("✓ Saved: 14_combined_roc_curves_with_mcc.png")
        plt.close()
    
    def generate_validation_report(self):
        """Generate comprehensive validation report"""
        print("\n" + "="*80)
        print("GENERATING VALIDATION REPORT")
        print("="*80)
        
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("COMPREHENSIVE MODEL VALIDATION REPORT")
        report_lines.append("="*80)
        report_lines.append(f"\nGenerated: {pd.Timestamp.now()}")
        
        report_lines.append("\n\n" + "="*80)
        report_lines.append("1. DATA SUMMARY")
        report_lines.append("="*80)
        report_lines.append(f"\nOriginal Training Set: {self.X_train_original.shape}")
        report_lines.append(f"  Virulent: {sum(self.y_train_original==1)}")
        report_lines.append(f"  Non-virulent: {sum(self.y_train_original==0)}")
        report_lines.append(f"\nSMOTE Training Set: {self.X_train_smote.shape}")
        report_lines.append(f"  Virulent: {sum(self.y_train_smote==1)}")
        report_lines.append(f"  Non-virulent: {sum(self.y_train_smote==0)}")
        report_lines.append(f"\nTest Set: {self.X_test.shape}")
        report_lines.append(f"  Virulent: {sum(self.y_test==1)}")
        report_lines.append(f"  Non-virulent: {sum(self.y_test==0)}")
        
        report_lines.append("\n\n" + "="*80)
        report_lines.append("2. VALIDATION TESTS COMPLETED")
        report_lines.append("="*80)
        report_lines.append("\n✓ Before/After SMOTE Comparison")
        report_lines.append("✓ Cross-Validation (5-fold and 10-fold)")
        report_lines.append("✓ Y-Randomization Test")
        report_lines.append("✓ Applicability Domain Analysis")
        report_lines.append("✓ Combined ROC Curves with MCC")
        
        report_lines.append("\n\n" + "="*80)
        report_lines.append("3. KEY FINDINGS")
        report_lines.append("="*80)
        
        # Load comparison results
        try:
            comparison_df = pd.read_csv('../results/validation/before_after_smote_comparison.csv')
            after_smote = comparison_df[comparison_df['Dataset'] == 'After SMOTE']
            best_model = after_smote.loc[after_smote['ROC-AUC'].idxmax()]
            
            report_lines.append(f"\nBest Performing Model: {best_model['Model']}")
            report_lines.append(f"  ROC-AUC: {best_model['ROC-AUC']:.4f}")
            report_lines.append(f"  MCC: {best_model['MCC']:.4f}")
            report_lines.append(f"  F1-Score: {best_model['F1-Score']:.4f}")
        except:
            report_lines.append("\nCould not load comparison results")
        
        report_lines.append("\n\n" + "="*80)
        report_lines.append("4. OUTPUT FILES")
        report_lines.append("="*80)
        report_lines.append("\nValidation Results:")
        report_lines.append("  • before_after_smote_comparison.csv")
        report_lines.append("  • cross_validation_results.csv")
        report_lines.append("  • y_randomization_results.csv")
        report_lines.append("  • applicability_domain_analysis.csv")
        report_lines.append("  • domain_distances.csv")
        report_lines.append("\nFigures:")
        report_lines.append("  • 10_before_after_smote_comparison.png")
        report_lines.append("  • 11_cross_validation_results.png")
        report_lines.append("  • 12_y_randomization_test.png")
        report_lines.append("  • 13_applicability_domain.png")
        report_lines.append("  • 14_combined_roc_curves_with_mcc.png")
        
        report_lines.append("\n" + "="*80)
        report_lines.append("END OF REPORT")
        report_lines.append("="*80)
        
        # Save report
        report_text = '\n'.join(report_lines)
        with open('../results/validation/validation_report.txt', 'w') as f:
            f.write(report_text)
        
        print(report_text)
        print("\n✓ Saved: validation_report.txt")
        
        return self


def main():
    """Main validation pipeline"""
    print("="*80)
    print("ENHANCED MODEL VALIDATION PIPELINE")
    print("="*80)
    
    # Initialize validator
    validator = EnhancedModelValidator()
    
    # Run validation pipeline
    validator.load_data() \
             .initialize_models() \
             .compare_before_after_smote() \
             .cross_validation_analysis(cv_folds=[5, 10]) \
             .y_randomization_test(n_iterations=50) \
             .applicability_domain_analysis() \
             .plot_combined_roc_with_mcc() \
             .generate_validation_report()
    
    print("\n" + "="*80)
    print("VALIDATION COMPLETE!")
    print("="*80)
    print("\n📊 All validation analyses completed successfully!")
    print("\n📁 Results saved in:")
    print("  • results/validation/ (CSV files)")
    print("  • results/figures/ (PNG plots)")


if __name__ == "__main__":
    main()
