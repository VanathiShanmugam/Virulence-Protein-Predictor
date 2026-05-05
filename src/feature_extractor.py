"""
Feature Extraction Module for Virulence Protein Prediction
Extracts 500+ features from protein FASTA sequences
"""

import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class ProteinFeatureExtractor:
    """Extract comprehensive features from protein sequences"""
    
    def __init__(self):
        self.amino_acids = list('ACDEFGHIKLMNPQRSTVWY')
        
    def extract_aac(self, sequence):
        """Amino Acid Composition (20 features)"""
        seq_len = len(sequence)
        aac = {}
        for aa in self.amino_acids:
            aac[f'AAC_{aa}'] = sequence.count(aa) / seq_len if seq_len > 0 else 0
        return aac
    
    def extract_dpc(self, sequence):
        """Dipeptide Composition (400 features)"""
        seq_len = len(sequence) - 1
        dpc = {}
        for aa1 in self.amino_acids:
            for aa2 in self.amino_acids:
                dipeptide = aa1 + aa2
                count = sum(1 for i in range(len(sequence)-1) if sequence[i:i+2] == dipeptide)
                dpc[f'DPC_{dipeptide}'] = count / seq_len if seq_len > 0 else 0
        return dpc
    
    def extract_physicochemical(self, sequence):
        """Physicochemical Properties (8 features)"""
        try:
            analyzed = ProteinAnalysis(str(sequence))
            return {
                'Molecular_Weight': analyzed.molecular_weight(),
                'Isoelectric_Point': analyzed.isoelectric_point(),
                'Aromaticity': analyzed.aromaticity(),
                'Instability_Index': analyzed.instability_index(),
                'GRAVY': analyzed.gravy(),
                'Aliphatic_Index': analyzed.aliphatic_index() if hasattr(analyzed, 'aliphatic_index') else 0,
                'Length': len(sequence),
                'Net_Charge_pH7': self._calculate_net_charge(sequence)
            }
        except:
            return {k: 0 for k in ['Molecular_Weight', 'Isoelectric_Point', 'Aromaticity',
                                   'Instability_Index', 'GRAVY', 'Aliphatic_Index', 
                                   'Length', 'Net_Charge_pH7']}
    
    def _calculate_net_charge(self, sequence):
        """Calculate net charge at pH 7"""
        positive = sequence.count('K') + sequence.count('R') + sequence.count('H')
        negative = sequence.count('D') + sequence.count('E')
        return positive - negative
    
    def extract_composition_features(self, sequence):
        """Compositional Features (8 features)"""
        seq_len = len(sequence)
        if seq_len == 0:
            return {k: 0 for k in ['Hydrophobic_Fraction', 'Hydrophilic_Fraction',
                                   'Polar_Fraction', 'NonPolar_Fraction',
                                   'Positive_Fraction', 'Negative_Fraction',
                                   'Small_AA_Fraction', 'Bulky_AA_Fraction']}
        
        hydrophobic = 'AILMFWV'
        hydrophilic = 'RNDQEHK'
        polar = 'STNQCYWH'
        nonpolar = 'GAVLIMFPW'
        positive = 'KRH'
        negative = 'DE'
        small = 'AGSV'
        bulky = 'FYWH'
        
        return {
            'Hydrophobic_Fraction': sum(sequence.count(aa) for aa in hydrophobic) / seq_len,
            'Hydrophilic_Fraction': sum(sequence.count(aa) for aa in hydrophilic) / seq_len,
            'Polar_Fraction': sum(sequence.count(aa) for aa in polar) / seq_len,
            'NonPolar_Fraction': sum(sequence.count(aa) for aa in nonpolar) / seq_len,
            'Positive_Fraction': sum(sequence.count(aa) for aa in positive) / seq_len,
            'Negative_Fraction': sum(sequence.count(aa) for aa in negative) / seq_len,
            'Small_AA_Fraction': sum(sequence.count(aa) for aa in small) / seq_len,
            'Bulky_AA_Fraction': sum(sequence.count(aa) for aa in bulky) / seq_len
        }
    
    def extract_signal_peptide_features(self, sequence):
        """Simple signal peptide prediction (4 features)"""
        n_term = sequence[:25] if len(sequence) >= 25 else sequence
        
        hydrophobic_aa = 'AILMFWV'
        positively_charged = 'KR'
        
        n_hydrophobic = sum(n_term.count(aa) for aa in hydrophobic_aa) / len(n_term) if len(n_term) > 0 else 0
        n_positive = sum(n_term[:10].count(aa) for aa in positively_charged) / 10 if len(n_term) >= 10 else 0
        
        # Simple heuristic for signal peptide
        signal_present = 1 if (n_positive > 0.2 and n_hydrophobic > 0.4) else 0
        
        return {
            'SignalPeptide_Present': signal_present,
            'SignalPeptide_Length': 25 if signal_present else 0,
            'Nterm_GRAVY_25': self._calculate_gravy(n_term),
            'Cleavage_Confidence': n_hydrophobic if signal_present else 0
        }
    
    def _calculate_gravy(self, sequence):
        """Calculate GRAVY for a sequence"""
        gravy_values = {
            'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
            'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
            'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
            'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
        }
        if len(sequence) == 0:
            return 0
        return sum(gravy_values.get(aa, 0) for aa in sequence) / len(sequence)
    
    def extract_tm_features(self, sequence):
        """Transmembrane domain prediction (2 features)"""
        # Simple sliding window approach
        window_size = 20
        hydrophobic = 'AILMFWV'
        
        tm_count = 0
        max_hydrophobic = 0
        
        for i in range(len(sequence) - window_size + 1):
            window = sequence[i:i+window_size]
            hydrophobic_content = sum(window.count(aa) for aa in hydrophobic) / window_size
            max_hydrophobic = max(max_hydrophobic, hydrophobic_content)
            if hydrophobic_content > 0.6:  # Threshold for TM region
                tm_count += 1
        
        return {
            'TM_Present': 1 if tm_count > 0 else 0,
            'MultiPass_TM': 1 if tm_count > window_size else 0
        }
    
    def extract_structural_features(self, sequence):
        """Structural features (3 features)"""
        seq_len = len(sequence)
        if seq_len == 0:
            return {'LowComplexity_Fraction': 0, 'Mean_Disorder_Score': 0, 'Repeat_Density': 0}
        
        # Low complexity: regions with limited amino acid diversity
        window = 20
        low_complexity_count = 0
        for i in range(len(sequence) - window + 1):
            subseq = sequence[i:i+window]
            unique_aa = len(set(subseq))
            if unique_aa < 5:  # Less than 5 different amino acids
                low_complexity_count += 1
        
        # Disorder score (simplified - based on charge and hydrophobicity)
        disorder_promoting = 'RKEDQSP'
        disorder_score = sum(sequence.count(aa) for aa in disorder_promoting) / seq_len
        
        # Repeat density
        repeat_count = self._count_repeats(sequence)
        
        return {
            'LowComplexity_Fraction': low_complexity_count / max(1, len(sequence) - window + 1),
            'Mean_Disorder_Score': disorder_score,
            'Repeat_Density': repeat_count / seq_len
        }
    
    def _count_repeats(self, sequence, min_length=3, max_length=10):
        """Count short tandem repeats"""
        repeat_count = 0
        for length in range(min_length, min(max_length + 1, len(sequence) // 2 + 1)):
            for i in range(len(sequence) - 2 * length + 1):
                motif = sequence[i:i+length]
                if sequence[i+length:i+2*length] == motif:
                    repeat_count += 1
        return repeat_count
    
    def extract_pseudo_aac(self, sequence, lambda_value=30):
        """Pseudo Amino Acid Composition (30 features)"""
        # Simplified version - uses physicochemical correlation
        features = {}
        seq_len = len(sequence)
        
        if seq_len < lambda_value:
            lambda_value = seq_len - 1
        
        # Hydrophobicity values
        hydrophobicity = {
            'A': 0.62, 'R': -2.53, 'N': -0.78, 'D': -0.90, 'C': 0.29,
            'Q': -0.85, 'E': -0.74, 'G': 0.48, 'H': -0.40, 'I': 1.38,
            'L': 1.06, 'K': -1.50, 'M': 0.64, 'F': 1.19, 'P': 0.12,
            'S': -0.18, 'T': -0.05, 'W': 0.81, 'Y': 0.26, 'V': 1.08
        }
        
        # Calculate correlation factors
        for lambda_i in range(1, min(lambda_value + 1, 31)):
            correlation = 0
            for j in range(seq_len - lambda_i):
                aa1 = sequence[j]
                aa2 = sequence[j + lambda_i]
                correlation += (hydrophobicity.get(aa1, 0) - hydrophobicity.get(aa2, 0)) ** 2
            features[f'PseAAC_lambda{lambda_i}'] = correlation / max(1, seq_len - lambda_i)
        
        # Fill remaining with zeros if sequence too short
        for lambda_i in range(len(features) + 1, 31):
            features[f'PseAAC_lambda{lambda_i}'] = 0
        
        return features
    
    def extract_secretion_motifs(self, sequence):
        """Check for common secretion motifs (1 feature)"""
        # Common virulence-associated secretion motifs
        motifs = ['RGD', 'LPXTG', 'YXXPhi']  # Simplified
        
        motif_present = 0
        for motif in motifs:
            if motif in sequence:
                motif_present = 1
                break
        
        return {'Secretion_Motif_Present': motif_present}
    
    def extract_all_features(self, sequence, protein_id):
        """Extract all features for a single sequence"""
        # Clean sequence
        sequence = ''.join([aa for aa in str(sequence).upper() if aa in self.amino_acids])
        
        if len(sequence) == 0:
            print(f"Warning: Empty sequence for {protein_id}")
            return None
        
        features = {'Protein_ID': protein_id}
        
        # Extract all feature types
        features.update(self.extract_structural_features(sequence))
        features.update(self.extract_aac(sequence))
        features.update(self.extract_dpc(sequence))
        features.update(self.extract_physicochemical(sequence))
        features.update(self.extract_composition_features(sequence))
        features.update(self.extract_pseudo_aac(sequence))
        features.update(self.extract_signal_peptide_features(sequence))
        features.update(self.extract_tm_features(sequence))
        features.update(self.extract_secretion_motifs(sequence))
        
        return features
    
    def process_fasta_file(self, fasta_file, label):
        """Process entire FASTA file and extract features"""
        print(f"\nProcessing {fasta_file} (Label: {label})")
        features_list = []
        
        try:
            for record in SeqIO.parse(fasta_file, "fasta"):
                features = self.extract_all_features(record.seq, record.id)
                if features:
                    features['Virulence'] = label
                    features_list.append(features)
            
            print(f"Extracted features from {len(features_list)} sequences")
            return pd.DataFrame(features_list)
        
        except Exception as e:
            print(f"Error processing {fasta_file}: {e}")
            return None


def main():
    """Main function to extract features from FASTA files"""
    print("="*80)
    print("VIRULENCE PROTEIN FEATURE EXTRACTION")
    print("="*80)
    
    # Initialize extractor
    extractor = ProteinFeatureExtractor()
    
    # Process virulent proteins
    virulent_df = extractor.process_fasta_file(
        '../data/raw/virulent.fasta', 
        label=1
    )
    
    # Process non-virulent proteins
    non_virulent_df = extractor.process_fasta_file(
        '../data/raw/non_virulent.fasta', 
        label=0
    )
    
    # Combine datasets
    if virulent_df is not None and non_virulent_df is not None:
        combined_df = pd.concat([virulent_df, non_virulent_df], ignore_index=True)
        
        # Save to file
        output_file = '../data/processed/protein_features.csv'
        combined_df.to_csv(output_file, index=False)
        print(f"\n✓ Features saved to: {output_file}")
        print(f"✓ Total sequences: {len(combined_df)}")
        print(f"✓ Virulent: {sum(combined_df['Virulence']==1)}")
        print(f"✓ Non-virulent: {sum(combined_df['Virulence']==0)}")
        print(f"✓ Total features: {len(combined_df.columns)-2}")  # Exclude Protein_ID and Virulence
        
        return combined_df
    else:
        print("\n✗ Error: Could not process FASTA files")
        return None


if __name__ == "__main__":
    main()
