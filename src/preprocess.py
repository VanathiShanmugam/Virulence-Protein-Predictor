"""
Data Preprocessing and Exploratory Analysis
Prepares data for machine learning and creates initial visualizations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

class DataPreprocessor:
    """Preprocess protein feature data for machine learning"""
    
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.scaler = None
        self.feature_names = None
        
    def explore_data(self):
        """Generate exploratory data analysis plots"""
        print("\n" + "="*80)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*80)
        
        print(f"\nDataset Shape: {self.data.shape}")
        print(f"Number of Features: {len(self.data.columns) - 2}")
        print(f"\nClass Distribution:")
        print(self.data['Virulence'].value_counts())
        print(f"\nClass Balance:")
        print(self.data['Virulence'].value_counts(normalize=True))
        
        # Create figure directory
        import os
        os.makedirs('../results/figures', exist_ok=True)
        
        # 1. Class Distribution
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Count plot
        self.data['Virulence'].value_counts().plot(kind='bar', ax=axes[0], color=['#2ecc71', '#e74c3c'])
        axes[0].set_xlabel('Class')
        axes[0].set_ylabel('Count')
        axes[0].set_title('Class Distribution')
        axes[0].set_xticklabels(['Non-Virulent (0)', 'Virulent (1)'], rotation=0)
        
        # Pie chart
        self.data['Virulence'].value_counts().plot(kind='pie', ax=axes[1], 
                                                     autopct='%1.1f%%',
                                                     colors=['#2ecc71', '#e74c3c'],
                                                     labels=['Non-Virulent', 'Virulent'])
        axes[1].set_ylabel('')
        axes[1].set_title('Class Proportion')
        
        plt.tight_layout()
        plt.savefig('../results/figures/01_class_distribution.png', dpi=300, bbox_inches='tight')
        print("\n✓ Saved: 01_class_distribution.png")
        plt.close()
        
        # 2. Sequence Length Distribution
        if 'Length' in self.data.columns:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            
            # Histogram
            self.data[self.data['Virulence']==0]['Length'].hist(bins=50, alpha=0.6, 
                                                                  label='Non-Virulent', 
                                                                  color='#2ecc71', ax=axes[0])
            self.data[self.data['Virulence']==1]['Length'].hist(bins=50, alpha=0.6, 
                                                                  label='Virulent', 
                                                                  color='#e74c3c', ax=axes[0])
            axes[0].set_xlabel('Sequence Length')
            axes[0].set_ylabel('Frequency')
            axes[0].set_title('Sequence Length Distribution')
            axes[0].legend()
            
            # Box plot
            self.data.boxplot(column='Length', by='Virulence', ax=axes[1])
            axes[1].set_xlabel('Class')
            axes[1].set_ylabel('Sequence Length')
            axes[1].set_title('Sequence Length by Class')
            axes[1].set_xticklabels(['Non-Virulent', 'Virulent'])
            plt.suptitle('')
            
            plt.tight_layout()
            plt.savefig('../results/figures/02_length_distribution.png', dpi=300, bbox_inches='tight')
            print("✓ Saved: 02_length_distribution.png")
            plt.close()
        
        # 3. Missing values
        missing = self.data.isnull().sum()
        if missing.sum() > 0:
            print(f"\nWarning: Found {missing.sum()} missing values")
            print(missing[missing > 0])
        else:
            print("\n✓ No missing values found")
        
        return self
    
    def prepare_data(self, test_size=0.2, val_size=0.1, random_state=42):
        """Split data into train, validation, and test sets"""
        print("\n" + "="*80)
        print("DATA PREPARATION")
        print("="*80)
        
        # Separate features and target
        X = self.data.drop(['Protein_ID', 'Virulence'], axis=1)
        y = self.data['Virulence']
        self.feature_names = X.columns.tolist()
        
        print(f"\nOriginal dataset: {X.shape}")
        print(f"Virulent: {sum(y==1)}, Non-virulent: {sum(y==0)}")
        
        # First split: separate test set
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Second split: separate validation from training
        val_size_adjusted = val_size / (1 - test_size)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, stratify=y_temp
        )
        
        print(f"\nTrain set: {self.X_train.shape}")
        print(f"  Virulent: {sum(self.y_train==1)}, Non-virulent: {sum(self.y_train==0)}")
        print(f"\nValidation set: {self.X_val.shape}")
        print(f"  Virulent: {sum(self.y_val==1)}, Non-virulent: {sum(self.y_val==0)}")
        print(f"\nTest set: {self.X_test.shape}")
        print(f"  Virulent: {sum(self.y_test==1)}, Non-virulent: {sum(self.y_test==0)}")
        
        return self
    
    def apply_smote(self, sampling_strategy='auto', random_state=42):
        """Apply SMOTE to balance training data"""
        print("\n" + "="*80)
        print("APPLYING SMOTE (Synthetic Minority Over-sampling)")
        print("="*80)
        
        print(f"\nBefore SMOTE:")
        print(f"  Virulent: {sum(self.y_train==1)}, Non-virulent: {sum(self.y_train==0)}")
        
        smote = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
        self.X_train, self.y_train = smote.fit_resample(self.X_train, self.y_train)
        
        print(f"\nAfter SMOTE:")
        print(f"  Virulent: {sum(self.y_train==1)}, Non-virulent: {sum(self.y_train==0)}")
        print(f"  New training set shape: {self.X_train.shape}")
        
        return self
    
    def scale_features(self):
        """Standardize features using StandardScaler"""
        print("\n" + "="*80)
        print("FEATURE SCALING")
        print("="*80)
        
        self.scaler = StandardScaler()
        
        # Fit on training data only
        self.X_train = pd.DataFrame(
            self.scaler.fit_transform(self.X_train),
            columns=self.feature_names
        )
        
        # Transform validation and test data
        self.X_val = pd.DataFrame(
            self.scaler.transform(self.X_val),
            columns=self.feature_names
        )
        
        self.X_test = pd.DataFrame(
            self.scaler.transform(self.X_test),
            columns=self.feature_names
        )
        
        print("\n✓ Features scaled using StandardScaler")
        print("  (fitted on training data only)")
        
        return self
    
    def feature_correlation_analysis(self, threshold=0.95):
        """Analyze and visualize feature correlations"""
        print("\n" + "="*80)
        print("FEATURE CORRELATION ANALYSIS")
        print("="*80)
        
        # Calculate correlation matrix
        corr_matrix = self.X_train.corr().abs()
        
        # Find highly correlated features
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        high_corr_features = [column for column in upper_triangle.columns 
                              if any(upper_triangle[column] > threshold)]
        
        print(f"\nFound {len(high_corr_features)} features with correlation > {threshold}")
        
        # Plot correlation heatmap (sample of features)
        sample_features = np.random.choice(self.feature_names, 
                                          size=min(50, len(self.feature_names)), 
                                          replace=False)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(self.X_train[sample_features].corr(), 
                   cmap='coolwarm', center=0, 
                   square=True, linewidths=0.5,
                   cbar_kws={"shrink": 0.8})
        plt.title('Feature Correlation Heatmap (Sample of 50 features)')
        plt.tight_layout()
        plt.savefig('../results/figures/03_feature_correlation.png', dpi=300, bbox_inches='tight')
        print("\n✓ Saved: 03_feature_correlation.png")
        plt.close()
        
        return self
    
    def save_processed_data(self):
        """Save processed datasets"""
        print("\n" + "="*80)
        print("SAVING PROCESSED DATA")
        print("="*80)
        
        import os
        os.makedirs('../data/processed', exist_ok=True)
        
        # Save splits
        self.X_train.to_csv('../data/processed/X_train.csv', index=False)
        self.X_val.to_csv('../data/processed/X_val.csv', index=False)
        self.X_test.to_csv('../data/processed/X_test.csv', index=False)
        
        pd.Series(self.y_train).to_csv('../data/processed/y_train.csv', index=False, header=['Virulence'])
        pd.Series(self.y_val).to_csv('../data/processed/y_val.csv', index=False, header=['Virulence'])
        pd.Series(self.y_test).to_csv('../data/processed/y_test.csv', index=False, header=['Virulence'])
        
        # Save scaler
        import joblib
        joblib.dump(self.scaler, '../models/scaler.pkl')
        
        print("\n✓ Saved all processed datasets")
        print("✓ Saved scaler to models/scaler.pkl")
        
        return self


def main():
    """Main preprocessing pipeline"""
    print("="*80)
    print("DATA PREPROCESSING PIPELINE")
    print("="*80)
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor('../data/processed/protein_features.csv')
    
    # Run preprocessing steps
    preprocessor.explore_data() \
                .prepare_data(test_size=0.2, val_size=0.1) \
                .apply_smote() \
                .scale_features() \
                .feature_correlation_analysis() \
                .save_processed_data()
    
    print("\n" + "="*80)
    print("PREPROCESSING COMPLETE!")
    print("="*80)
    print("\n✓ Ready for model training")
    print("  Run: python 3_train_models.py")


if __name__ == "__main__":
    main()
