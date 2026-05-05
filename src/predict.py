"""
Prediction Module
Use trained models to predict virulence of new protein sequences
"""

import numpy as np
import pandas as pd
from Bio import SeqIO
import joblib
import sys
import warnings
warnings.filterwarnings('ignore')

# Import feature extractor - now using importable module name
from feature_extractor import ProteinFeatureExtractor

class VirulencePredictor:
    """Predict virulence from FASTA sequences"""
    
    def __init__(self):
        self.models = {}
        self.scaler = None
        self.feature_extractor = ProteinFeatureExtractor()
        self.feature_names = None
        
    def load_models(self):
        """Load all trained models and scaler"""
        print("\n" + "="*80)
        print("LOADING TRAINED MODELS")
        print("="*80)
        
        model_names = {
            'Random Forest': 'random_forest',
            'XGBoost': 'xgboost',
            'SVM': 'svm',
            'Logistic Regression': 'logistic_regression'
        }
        
        for display_name, filename in model_names.items():
            try:
                self.models[display_name] = joblib.load(f'../models/{filename}.pkl')
                print(f"✓ Loaded {display_name}")
            except:
                print(f"✗ Failed to load {display_name}")
        
        # Load scaler
        try:
            self.scaler = joblib.load('../models/scaler.pkl')
            print("✓ Loaded scaler")
        except:
            print("✗ Failed to load scaler")
            
        # Load feature names from training data
        try:
            X_train = pd.read_csv('../data/processed/X_train.csv')
            self.feature_names = X_train.columns.tolist()
            print(f"✓ Loaded {len(self.feature_names)} feature names")
        except:
            print("✗ Failed to load feature names")
        
        return self
    
    def extract_features(self, fasta_file):
        """Extract features from FASTA file"""
        print(f"\nExtracting features from: {fasta_file}")
        
        sequences = []
        for record in SeqIO.parse(fasta_file, "fasta"):
            features = self.feature_extractor.extract_all_features(record.seq, record.id)
            if features:
                sequences.append(features)
        
        df = pd.DataFrame(sequences)
        print(f"✓ Extracted features from {len(df)} sequences")
        
        return df
    
    def preprocess_features(self, features_df):
        """Preprocess features to match training data"""
        # Separate protein IDs
        protein_ids = features_df['Protein_ID'].values
        
        # Remove non-feature columns
        X = features_df.drop(['Protein_ID'], axis=1, errors='ignore')
        
        # Ensure all required features are present
        for feature in self.feature_names:
            if feature not in X.columns:
                X[feature] = 0
        
        # Select only training features in correct order
        X = X[self.feature_names]
        
        # Scale features
        X_scaled = pd.DataFrame(
            self.scaler.transform(X),
            columns=self.feature_names
        )
        
        return X_scaled, protein_ids
    
    def predict(self, X):
        """Make predictions using all models"""
        predictions = {}
        
        for name, model in self.models.items():
            # Predict
            y_pred = model.predict(X)
            y_proba = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else y_pred
            
            predictions[name] = {
                'prediction': y_pred,
                'probability': y_proba
            }
        
        return predictions
    
    def predict_fasta_file(self, fasta_file, output_file=None):
        """Complete prediction pipeline for a FASTA file"""
        print("\n" + "="*80)
        print("VIRULENCE PREDICTION PIPELINE")
        print("="*80)
        
        # Extract features
        features_df = self.extract_features(fasta_file)
        
        if len(features_df) == 0:
            print("✗ No valid sequences found")
            return None
        
        # Preprocess
        X_scaled, protein_ids = self.preprocess_features(features_df)
        
        # Predict
        print("\nMaking predictions...")
        predictions = self.predict(X_scaled)
        
        # Create results DataFrame
        results = pd.DataFrame({'Protein_ID': protein_ids})
        
        for name, pred in predictions.items():
            results[f'{name}_Prediction'] = ['Virulent' if p == 1 else 'Non-Virulent' 
                                              for p in pred['prediction']]
            results[f'{name}_Probability'] = pred['probability']
        
        # Ensemble prediction (majority vote)
        ensemble_pred = []
        for i in range(len(results)):
            votes = [predictions[name]['prediction'][i] for name in self.models.keys()]
            ensemble_pred.append(1 if sum(votes) >= len(votes)/2 else 0)
        
        results['Ensemble_Prediction'] = ['Virulent' if p == 1 else 'Non-Virulent' 
                                          for p in ensemble_pred]
        
        # Average probability
        avg_proba = np.mean([predictions[name]['probability'] 
                            for name in self.models.keys()], axis=0)
        results['Ensemble_Probability'] = avg_proba
        
        # Display results
        print("\n" + "="*80)
        print("PREDICTION RESULTS")
        print("="*80)
        print(f"\nTotal sequences: {len(results)}")
        print(f"Predicted Virulent: {sum(ensemble_pred)}")
        print(f"Predicted Non-Virulent: {len(ensemble_pred) - sum(ensemble_pred)}")
        
        # Show first few predictions
        print("\nFirst 5 predictions:")
        print(results[['Protein_ID', 'Ensemble_Prediction', 'Ensemble_Probability']].head().to_string(index=False))
        
        # Save to file
        if output_file:
            results.to_csv(output_file, index=False)
            print(f"\n✓ Results saved to: {output_file}")
        
        return results
    
    def predict_single_sequence(self, sequence, protein_id="Unknown"):
        """Predict virulence for a single protein sequence"""
        # Extract features
        features = self.feature_extractor.extract_all_features(sequence, protein_id)
        
        if not features:
            print("✗ Could not extract features")
            return None
        
        # Convert to DataFrame
        features_df = pd.DataFrame([features])
        
        # Preprocess
        X_scaled, _ = self.preprocess_features(features_df)
        
        # Predict
        predictions = self.predict(X_scaled)
        
        # Display results
        print(f"\nProtein: {protein_id}")
        print(f"Length: {len(sequence)} aa")
        print("\nPredictions:")
        
        for name, pred in predictions.items():
            label = "Virulent" if pred['prediction'][0] == 1 else "Non-Virulent"
            prob = pred['probability'][0]
            print(f"  {name:20s}: {label:15s} (prob: {prob:.4f})")
        
        # Ensemble
        ensemble_votes = [pred['prediction'][0] for pred in predictions.values()]
        ensemble_label = "Virulent" if sum(ensemble_votes) >= len(ensemble_votes)/2 else "Non-Virulent"
        ensemble_prob = np.mean([pred['probability'][0] for pred in predictions.values()])
        
        print(f"\n  {'Ensemble':20s}: {ensemble_label:15s} (prob: {ensemble_prob:.4f})")
        
        return predictions


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Predict virulence from protein sequences')
    parser.add_argument('--fasta', type=str, help='Input FASTA file')
    parser.add_argument('--output', type=str, default='predictions.csv', 
                       help='Output CSV file (default: predictions.csv)')
    
    args = parser.parse_args()
    
    if not args.fasta:
        print("\n" + "="*80)
        print("VIRULENCE PREDICTION TOOL")
        print("="*80)
        print("\nUsage: python 5_predict_new_sequences_FIXED.py --fasta <input.fasta> [--output <output.csv>]")
        print("\nExample:")
        print("  python 5_predict_new_sequences_FIXED.py --fasta new_sequences.fasta --output results.csv")
        print("\nNote: Make sure you have run the training pipeline first (steps 1-3)")
        print("="*80)
        return
    
    # Initialize predictor
    predictor = VirulencePredictor()
    predictor.load_models()
    
    # Make predictions
    results = predictor.predict_fasta_file(args.fasta, args.output)
    
    if results is not None:
        print("\n" + "="*80)
        print("PREDICTION COMPLETE!")
        print("="*80)
        print(f"\n✓ Results saved to: {args.output}")
        print("\nYou can now:")
        print("  1. Open the CSV file to view predictions")
        print("  2. Use 'Ensemble_Prediction' column for final decisions")
        print("  3. Check 'Ensemble_Probability' for confidence scores")


if __name__ == "__main__":
    main()
