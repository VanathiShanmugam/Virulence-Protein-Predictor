"""
Model Training Module
Trains 4 machine learning models for virulence prediction:
- Random Forest
- XGBoost
- Support Vector Machine (SVM)
- Logistic Regression
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, matthews_corrcoef,
                             classification_report, confusion_matrix)
import joblib
import warnings
warnings.filterwarnings('ignore')

class VirulenceModelTrainer:
    """Train and evaluate multiple ML models"""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        
    def load_data(self):
        """Load processed data"""
        print("\n" + "="*80)
        print("LOADING PROCESSED DATA")
        print("="*80)
        
        self.X_train = pd.read_csv('../data/processed/X_train.csv')
        self.X_val = pd.read_csv('../data/processed/X_val.csv')
        self.X_test = pd.read_csv('../data/processed/X_test.csv')
        
        self.y_train = pd.read_csv('../data/processed/y_train.csv')['Virulence']
        self.y_val = pd.read_csv('../data/processed/y_val.csv')['Virulence']
        self.y_test = pd.read_csv('../data/processed/y_test.csv')['Virulence']
        
        print(f"\n✓ Training set: {self.X_train.shape}")
        print(f"✓ Validation set: {self.X_val.shape}")
        print(f"✓ Test set: {self.X_test.shape}")
        
        return self
    
    def initialize_models(self):
        """Initialize all models with optimized parameters"""
        print("\n" + "="*80)
        print("INITIALIZING MODELS")
        print("="*80)
        
        # 1. Random Forest
        self.models['Random Forest'] = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        # 2. XGBoost
        self.models['XGBoost'] = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            scale_pos_weight=1
        )
        
        # 3. Support Vector Machine
        self.models['SVM'] = SVC(
            C=1.0,
            kernel='rbf',
            gamma='scale',
            random_state=42,
            class_weight='balanced',
            probability=True  # For ROC curve
        )
        
        # 4. Logistic Regression
        self.models['Logistic Regression'] = LogisticRegression(
            C=1.0,
            penalty='l2',
            solver='lbfgs',
            max_iter=1000,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
        
        print(f"\n✓ Initialized {len(self.models)} models:")
        for name in self.models.keys():
            print(f"  - {name}")
        
        return self
    
    def train_models(self):
        """Train all models"""
        print("\n" + "="*80)
        print("TRAINING MODELS")
        print("="*80)
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            model.fit(self.X_train, self.y_train)
            print(f"✓ {name} trained successfully")
        
        return self
    
    def evaluate_model(self, name, model, X, y, dataset_name):
        """Evaluate a single model"""
        # Predictions
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else y_pred
        
        # Calculate metrics
        metrics = {
            'Model': name,
            'Dataset': dataset_name,
            'Accuracy': accuracy_score(y, y_pred),
            'Precision': precision_score(y, y_pred, zero_division=0),
            'Recall': recall_score(y, y_pred, zero_division=0),
            'F1-Score': f1_score(y, y_pred, zero_division=0),
            'ROC-AUC': roc_auc_score(y, y_pred_proba),
            'MCC': matthews_corrcoef(y, y_pred)
        }
        
        return metrics, y_pred, y_pred_proba
    
    def evaluate_all_models(self):
        """Evaluate all models on validation and test sets"""
        print("\n" + "="*80)
        print("EVALUATING MODELS")
        print("="*80)
        
        results_list = []
        
        for name, model in self.models.items():
            print(f"\nEvaluating {name}...")
            
            # Validation set
            val_metrics, val_pred, val_proba = self.evaluate_model(
                name, model, self.X_val, self.y_val, 'Validation'
            )
            results_list.append(val_metrics)
            
            # Test set
            test_metrics, test_pred, test_proba = self.evaluate_model(
                name, model, self.X_test, self.y_test, 'Test'
            )
            results_list.append(test_metrics)
            
            # Store predictions for later use
            if name not in self.results:
                self.results[name] = {}
            self.results[name]['val_pred'] = val_pred
            self.results[name]['val_proba'] = val_proba
            self.results[name]['test_pred'] = test_pred
            self.results[name]['test_proba'] = test_proba
            
            print(f"  Validation ROC-AUC: {val_metrics['ROC-AUC']:.4f}")
            print(f"  Test ROC-AUC: {test_metrics['ROC-AUC']:.4f}")
        
        # Convert to DataFrame
        self.results_df = pd.DataFrame(results_list)
        
        return self
    
    def display_results(self):
        """Display model performance results"""
        print("\n" + "="*80)
        print("MODEL PERFORMANCE SUMMARY")
        print("="*80)
        
        # Validation results
        val_results = self.results_df[self.results_df['Dataset'] == 'Validation'].sort_values('ROC-AUC', ascending=False)
        print("\nVALIDATION SET RESULTS:")
        print(val_results[['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'MCC']].to_string(index=False))
        
        # Test results
        test_results = self.results_df[self.results_df['Dataset'] == 'Test'].sort_values('ROC-AUC', ascending=False)
        print("\nTEST SET RESULTS:")
        print(test_results[['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'MCC']].to_string(index=False))
        
        # Best model
        best_model_name = test_results.iloc[0]['Model']
        best_auc = test_results.iloc[0]['ROC-AUC']
        print(f"\n🏆 Best Model: {best_model_name} (Test ROC-AUC: {best_auc:.4f})")
        
        return self
    
    def save_models_and_results(self):
        """Save trained models and results"""
        print("\n" + "="*80)
        print("SAVING MODELS AND RESULTS")
        print("="*80)
        
        import os
        os.makedirs('../models', exist_ok=True)
        os.makedirs('../results/tables', exist_ok=True)
        
        # Save models
        for name, model in self.models.items():
            filename = name.replace(' ', '_').lower()
            joblib.dump(model, f'../models/{filename}.pkl')
            print(f"✓ Saved {name}")
        
        # Save results
        self.results_df.to_csv('../results/tables/validation_results.csv', index=False)
        print("\n✓ Saved validation results")
        
        # Save detailed classification reports
        for name, model in self.models.items():
            # Test set report
            y_pred = model.predict(self.X_test)
            report = classification_report(self.y_test, y_pred, 
                                          target_names=['Non-Virulent', 'Virulent'],
                                          output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            
            filename = name.replace(' ', '_').lower()
            report_df.to_csv(f'../results/tables/classification_report_{filename}.csv')
        
        print("✓ Saved classification reports")
        
        # Save predictions for plotting
        predictions = {
            'y_val': self.y_val,
            'y_test': self.y_test
        }
        
        for name in self.models.keys():
            predictions[f'{name}_val_pred'] = self.results[name]['val_pred']
            predictions[f'{name}_val_proba'] = self.results[name]['val_proba']
            predictions[f'{name}_test_pred'] = self.results[name]['test_pred']
            predictions[f'{name}_test_proba'] = self.results[name]['test_proba']
        
        joblib.dump(predictions, '../models/predictions.pkl')
        print("✓ Saved predictions")
        
        return self


def main():
    """Main training pipeline"""
    print("="*80)
    print("VIRULENCE PROTEIN PREDICTION - MODEL TRAINING")
    print("="*80)
    
    # Initialize trainer
    trainer = VirulenceModelTrainer()
    
    # Run training pipeline
    trainer.load_data() \
           .initialize_models() \
           .train_models() \
           .evaluate_all_models() \
           .display_results() \
           .save_models_and_results()
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print("\n✓ All models trained and saved")
    print("  Next step: python 4_evaluate_models.py")


if __name__ == "__main__":
    main()
