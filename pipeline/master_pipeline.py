"""
Master Pipeline Script
Run the entire virulence prediction pipeline with a single command
"""

import subprocess
import sys
import os
from datetime import datetime

class PipelineRunner:
    """Orchestrate the complete pipeline"""
    
    def __init__(self):
        self.steps = {
            '1': ('Feature Extraction', '1_feature_extraction.py'),
            '2': ('Data Preprocessing', '2_preprocess_data.py'),
            '3': ('Model Training', '3_train_models.py'),
            '4': ('Model Evaluation', '4_evaluate_models.py')
        }
        self.start_time = None
        
    def print_header(self):
        """Print pipeline header"""
        print("="*80)
        print("VIRULENCE PROTEIN PREDICTION PIPELINE")
        print("="*80)
        print(f"\nStart Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("\nPipeline Steps:")
        for step_num, (name, _) in self.steps.items():
            print(f"  {step_num}. {name}")
        print("="*80)
    
    def check_data_files(self):
        """Check if required FASTA files exist"""
        print("\n" + "="*80)
        print("CHECKING DATA FILES")
        print("="*80)
        
        files_to_check = [
            '../data/raw/virulent.fasta',
            '../data/raw/non_virulent.fasta'
        ]
        
        all_exist = True
        for file_path in files_to_check:
            if os.path.exists(file_path):
                size = os.path.getsize(file_path)
                print(f"✓ Found: {file_path} ({size:,} bytes)")
            else:
                print(f"✗ Missing: {file_path}")
                all_exist = False
        
        if not all_exist:
            print("\n⚠️  WARNING: Missing FASTA files!")
            print("\nPlease place your FASTA files in data/raw/:")
            print("  - virulent.fasta (your virulent protein sequences)")
            print("  - non_virulent.fasta (your non-virulent protein sequences)")
            response = input("\nContinue anyway? (yes/no): ")
            if response.lower() != 'yes':
                print("Exiting...")
                sys.exit(1)
        
        return all_exist
    
    def run_step(self, step_num):
        """Run a single pipeline step"""
        if step_num not in self.steps:
            print(f"✗ Invalid step: {step_num}")
            return False
        
        name, script = self.steps[step_num]
        
        print("\n" + "="*80)
        print(f"STEP {step_num}: {name}")
        print("="*80)
        print(f"Running: {script}")
        print("-"*80)
        
        try:
            result = subprocess.run(
                [sys.executable, script],
                check=True,
                capture_output=False
            )
            
            print("-"*80)
            print(f"✓ Step {step_num} completed successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            print("-"*80)
            print(f"✗ Step {step_num} failed with error code {e.returncode}")
            return False
    
    def run_all_steps(self):
        """Run all pipeline steps"""
        self.start_time = datetime.now()
        self.print_header()
        
        # Check data files
        self.check_data_files()
        
        # Run each step
        for step_num in sorted(self.steps.keys()):
            success = self.run_step(step_num)
            if not success:
                print("\n⚠️  Pipeline stopped due to error")
                return False
        
        # Print summary
        self.print_summary()
        return True
    
    def print_summary(self):
        """Print pipeline completion summary"""
        end_time = datetime.now()
        duration = end_time - self.start_time
        
        print("\n" + "="*80)
        print("PIPELINE COMPLETE!")
        print("="*80)
        print(f"\nStart Time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Duration: {duration}")
        
        print("\n📊 OUTPUTS:")
        print("\nModels (in models/):")
        print("  ✓ random_forest.pkl")
        print("  ✓ xgboost.pkl")
        print("  ✓ svm.pkl")
        print("  ✓ logistic_regression.pkl")
        print("  ✓ scaler.pkl")
        
        print("\nFigures (in results/figures/):")
        figures = [
            "01_class_distribution.png",
            "02_length_distribution.png",
            "03_feature_correlation.png",
            "04_combined_roc_curves.png ⭐ MAIN FIGURE",
            "05_precision_recall_curves.png",
            "06_confusion_matrices.png",
            "07_feature_importance.png",
            "08_learning_curves.png",
            "09_model_comparison.png"
        ]
        for fig in figures:
            print(f"  ✓ {fig}")
        
        print("\nTables (in results/tables/):")
        print("  ✓ validation_results.csv")
        print("  ✓ classification_reports")
        print("  ✓ final_comparison.csv")
        print("  ✓ feature_importance tables")
        
        print("\n🎯 NEXT STEPS:")
        print("  1. Review figures in results/figures/")
        print("  2. Check model performance in results/tables/")
        print("  3. Use models for prediction:")
        print("     python 5_predict_new_sequences.py --fasta new.fasta --output results.csv")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Run the complete virulence prediction pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline
  python master_pipeline.py --all
  
  # Run specific step
  python master_pipeline.py --step 1
  
  # Run from step 2 onwards
  python master_pipeline.py --from 2
        """
    )
    
    parser.add_argument('--all', action='store_true',
                       help='Run all pipeline steps')
    parser.add_argument('--step', type=str,
                       help='Run specific step (1-4)')
    parser.add_argument('--from', type=str, dest='from_step',
                       help='Run from this step onwards (1-4)')
    
    args = parser.parse_args()
    
    runner = PipelineRunner()
    
    if args.all:
        # Run all steps
        runner.run_all_steps()
        
    elif args.step:
        # Run specific step
        runner.print_header()
        runner.run_step(args.step)
        
    elif args.from_step:
        # Run from specific step onwards
        runner.start_time = datetime.now()
        runner.print_header()
        
        for step_num in sorted(runner.steps.keys()):
            if step_num >= args.from_step:
                success = runner.run_step(step_num)
                if not success:
                    break
        
        runner.print_summary()
    
    else:
        # No arguments - show help
        parser.print_help()


if __name__ == "__main__":
    main()
