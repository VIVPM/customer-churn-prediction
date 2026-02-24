"""
Customer Churn Prediction - Main Pipeline

Run the complete pipeline or individual steps:
    python main.py --all          # Run complete pipeline
    python main.py --eda          # Run EDA only
    python main.py --preprocess   # Run preprocessing only
    python main.py --features     # Run feature engineering only
    python main.py --train        # Run training only
    python main.py --evaluate     # Run evaluation only
    python main.py --predict      # Run interactive prediction

Based on the original notebook implementation.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from backend.training.data_loader import load_data
from backend.training.eda import run_eda
from backend.training.preprocessing import preprocess_data
from backend.training.feature_engineering import run_feature_engineering
from backend.training.train import train_models
from backend.training.evaluate import evaluate_models
from backend.training.predict import interactive_prediction


def print_banner():
    """Print project banner."""
    print("\n" + "=" * 60)
    print("  CUSTOMER CHURN PREDICTION PIPELINE")
    print("  Based on multi-sheet Excel data analysis")
    print("=" * 60)


def run_full_pipeline():
    """Run the complete ML pipeline."""
    print_banner()
    
    print("\n[1/5] Running Exploratory Data Analysis...")
    run_eda()
    
    print("\n[2/5] Running Data Preprocessing...")
    preprocess_data()
    
    print("\n[3/5] Running Feature Engineering...")
    run_feature_engineering()
    
    print("\n[4/5] Training Models...")
    train_models()
    
    print("\n[5/5] Evaluating Models...")
    evaluate_models()
    
    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE!")
    print("=" * 60)
    print("\nNext steps:")
    print("  - Check reports/figures/ for visualizations")
    print("  - Check reports/model_comparison.csv for results")
    print("  - Run 'python main.py --predict' for predictions")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Customer Churn Prediction Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --all          Run complete pipeline
  python main.py --eda          Run EDA only
  python main.py --preprocess   Run preprocessing only
  python main.py --features     Run feature engineering only
  python main.py --train        Run training only
  python main.py --evaluate     Run evaluation only
  python main.py --predict      Run interactive prediction
        """
    )
    
    parser.add_argument('--all', action='store_true', 
                        help='Run complete pipeline')
    parser.add_argument('--eda', action='store_true', 
                        help='Run EDA only')
    parser.add_argument('--preprocess', action='store_true', 
                        help='Run preprocessing only')
    parser.add_argument('--features', action='store_true', 
                        help='Run feature engineering only')
    parser.add_argument('--train', action='store_true', 
                        help='Run training only')
    parser.add_argument('--evaluate', action='store_true', 
                        help='Run evaluation only')
    parser.add_argument('--predict', action='store_true', 
                        help='Run interactive prediction')
    
    args = parser.parse_args()
    
    # If no arguments provided, show help
    if not any(vars(args).values()):
        print_banner()
        print("\nNo arguments provided. Running full pipeline...")
        print("Use --help for available options.\n")
        run_full_pipeline()
        return
    
    # Run specific steps (multiple flags can be combined)
    if args.all:
        run_full_pipeline()
        return
    
    print_banner()
    
    if args.eda:
        print("\nRunning EDA...")
        run_eda()
    if args.preprocess:
        print("\nRunning Preprocessing...")
        preprocess_data()
    if args.features:
        print("\nRunning Feature Engineering...")
        run_feature_engineering()
    if args.train:
        print("\nTraining Models...")
        train_models()
    if args.evaluate:
        print("\nEvaluating Models...")
        evaluate_models()
    if args.predict:
        interactive_prediction()


if __name__ == "__main__":
    main()
