"""
Exploratory Data Analysis Module

Performs EDA on the merged customer churn dataset:
- Basic statistics and data info
- Missing values analysis
- Correlation heatmap
- Distribution plots
- Churn rate analysis by categorical features
- Time series analysis

Based on the original notebook implementation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import FIGURES_DIR, TARGET_COLUMN
from backend.training.utils import create_directories, print_separator
from backend.training.data_loader import load_data


def analyze_basic_info(df):
    """Print basic dataset information."""
    print_separator("BASIC DATASET INFO")
    print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"\nData Types:")
    print(df.dtypes)
    print(f"\nBasic Statistics:")
    print(df.describe())


def analyze_missing_values(df):
    """Analyze and report missing values."""
    print_separator("MISSING VALUES ANALYSIS")
    missing = df.isna().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Count': missing,
        'Missing %': missing_pct
    })
    missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
    
    if len(missing_df) > 0:
        print(missing_df)
    else:
        print("No missing values found!")
    
    return missing_df


def analyze_interaction_nulls(df):
    """
    Analyze interaction-related null patterns.
    Based on notebook cells 14-15.
    """
    print_separator("INTERACTION NULL ANALYSIS")
    
    # Cell 14: DaysSinceLastInteraction value counts
    if 'DaysSinceLastInteraction' in df.columns:
        print("DaysSinceLastInteraction value counts:")
        print(df['DaysSinceLastInteraction'].value_counts())
    
    # Cell 15: Rows where all interaction columns are null
    interaction_cols = ['InteractionID', 'InteractionDate', 'InteractionType',
                        'ResolutionStatus', 'DaysSinceLastInteraction']
    existing_cols = [c for c in interaction_cols if c in df.columns]
    
    if existing_cols:
        all_null_mask = df[existing_cols].isnull().all(axis=1)
        null_count = all_null_mask.sum()
        print(f"\nRows with ALL interaction columns null: {null_count}")
        if null_count > 0:
            print(df[all_null_mask].head())


def analyze_categorical_values(df):
    """Analyze unique values in categorical columns."""
    print_separator("CATEGORICAL VALUE COUNTS")
    
    categorical_cols = ['InteractionType', 'ResolutionStatus', 'ServiceUsage', 'ProductCategory']
    
    for col in categorical_cols:
        if col in df.columns:
            print(f"\n{col}:")
            print(df[col].value_counts())


def plot_correlation_heatmap(df):
    """Plot correlation heatmap for numerical columns."""
    print_separator("CORRELATION ANALYSIS")
    
    cols = [
        'CustomerID', 'TransactionID', 'AmountSpent', 'InteractionID',
        'LoginFrequency', 'DaysSinceLastTransaction', 'DaysSinceLastInteraction',
        'DaysSinceLastLogin', 'TransactionMonth', TARGET_COLUMN
    ]
    
    # Filter to existing columns
    cols = [c for c in cols if c in df.columns]
    
    # Select only numeric columns
    numeric_df = df[cols].select_dtypes(include=[np.number])
    
    plt.figure(figsize=(12, 10))
    correlation_matrix = numeric_df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', 
                linewidths=0.5, square=True)
    plt.title('Correlation Heatmap', fontsize=14)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'correlation_heatmap.png', dpi=150)
    plt.close()
    print("Saved: correlation_heatmap.png")
    
    return correlation_matrix


def plot_amount_spent_distribution(df):
    """Plot distribution of AmountSpent."""
    plt.figure(figsize=(10, 6))
    sns.histplot(df['AmountSpent'], bins=30, kde=True, color='steelblue')
    plt.title('Distribution of Amount Spent', fontsize=14)
    plt.xlabel('Amount Spent')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'amount_spent_distribution.png', dpi=150)
    plt.close()
    print("Saved: amount_spent_distribution.png")


def plot_login_frequency_distribution(df):
    """Plot distribution of LoginFrequency."""
    plt.figure(figsize=(10, 6))
    sns.histplot(df['LoginFrequency'], bins=20, kde=True, color='coral')
    plt.title('Distribution of Login Frequency', fontsize=14)
    plt.xlabel('Login Frequency')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'login_frequency_distribution.png', dpi=150)
    plt.close()
    print("Saved: login_frequency_distribution.png")


def plot_churn_by_product_category(df):
    """Plot churn rate by product category."""
    plt.figure(figsize=(10, 6))
    sns.barplot(x='ProductCategory', y=TARGET_COLUMN, data=df, 
                estimator=np.mean, palette='viridis', errorbar=None)
    plt.title('Churn Rate by Product Category', fontsize=14)
    plt.xlabel('Product Category')
    plt.ylabel('Churn Rate')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'churn_by_product_category.png', dpi=150)
    plt.close()
    print("Saved: churn_by_product_category.png")


def plot_churn_by_interaction_type(df):
    """Plot churn rate by interaction type."""
    plt.figure(figsize=(10, 6))
    sns.barplot(x='InteractionType', y=TARGET_COLUMN, data=df, 
                estimator=np.mean, palette='coolwarm', errorbar=None)
    plt.title('Churn Rate by Interaction Type', fontsize=14)
    plt.xlabel('Interaction Type')
    plt.ylabel('Churn Rate')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'churn_by_interaction_type.png', dpi=150)
    plt.close()
    print("Saved: churn_by_interaction_type.png")


def plot_churn_by_resolution_status(df):
    """Plot churn rate by resolution status."""
    plt.figure(figsize=(8, 6))
    sns.barplot(x='ResolutionStatus', y=TARGET_COLUMN, data=df, 
                estimator=np.mean, palette='Set2', errorbar=None)
    plt.title('Churn Rate by Resolution Status', fontsize=14)
    plt.xlabel('Resolution Status')
    plt.ylabel('Churn Rate')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'churn_by_resolution_status.png', dpi=150)
    plt.close()
    print("Saved: churn_by_resolution_status.png")


def plot_churn_by_service_usage(df):
    """Plot churn rate by service usage."""
    plt.figure(figsize=(8, 6))
    sns.barplot(x='ServiceUsage', y=TARGET_COLUMN, data=df, 
                estimator=np.mean, palette='husl', errorbar=None)
    plt.title('Churn Rate by Service Usage', fontsize=14)
    plt.xlabel('Service Usage')
    plt.ylabel('Churn Rate')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'churn_by_service_usage.png', dpi=150)
    plt.close()
    print("Saved: churn_by_service_usage.png")


def plot_transactions_over_time(df):
    """Plot transactions over time."""
    monthly_tx = df.resample('M', on='TransactionDate')['TransactionID'].count()
    
    plt.figure(figsize=(12, 6))
    monthly_tx.plot(color='steelblue', linewidth=2)
    plt.title('Transactions Over Time', fontsize=14)
    plt.xlabel('Date')
    plt.ylabel('Number of Transactions')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'transactions_over_time.png', dpi=150)
    plt.close()
    print("Saved: transactions_over_time.png")


def plot_interactions_over_time(df):
    """Plot interactions over time."""
    monthly_int = df.resample('M', on='InteractionDate')['InteractionID'].count()
    
    plt.figure(figsize=(12, 6))
    monthly_int.plot(color='coral', linewidth=2)
    plt.title('Interactions Over Time', fontsize=14)
    plt.xlabel('Date')
    plt.ylabel('Number of Interactions')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'interactions_over_time.png', dpi=150)
    plt.close()
    print("Saved: interactions_over_time.png")


def plot_amount_spent_by_churn(df):
    """Plot amount spent by churn status."""
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=TARGET_COLUMN, y='AmountSpent', data=df, palette='Set2')
    plt.title('Amount Spent by Churn Status', fontsize=14)
    plt.xlabel('Churn Status (0=No, 1=Yes)')
    plt.ylabel('Amount Spent')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'amount_spent_by_churn.png', dpi=150)
    plt.close()
    print("Saved: amount_spent_by_churn.png")


def plot_days_since_login_by_churn(df):
    """Plot days since last login by churn status."""
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=TARGET_COLUMN, y='DaysSinceLastLogin', data=df, palette='coolwarm')
    plt.title('Days Since Last Login by Churn Status', fontsize=14)
    plt.xlabel('Churn Status (0=No, 1=Yes)')
    plt.ylabel('Days Since Last Login')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'days_since_login_by_churn.png', dpi=150)
    plt.close()
    print("Saved: days_since_login_by_churn.png")


def plot_churn_distribution(df):
    """Plot target variable distribution."""
    plt.figure(figsize=(8, 6))
    churn_counts = df[TARGET_COLUMN].value_counts()
    colors = ['#2ecc71', '#e74c3c']
    churn_counts.plot(kind='bar', color=colors, edgecolor='black')
    plt.title('Churn Distribution', fontsize=14)
    plt.xlabel('Churn Status (0=No, 1=Yes)')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    
    # Add percentage labels
    total = len(df)
    for i, v in enumerate(churn_counts):
        plt.text(i, v + 50, f'{v/total*100:.1f}%', ha='center', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'churn_distribution.png', dpi=150)
    plt.close()
    print("Saved: churn_distribution.png")


def plot_interactions_per_customer(df):
    """
    Plot count of interactions per customer.
    Based on notebook cell 49.
    """
    plt.figure(figsize=(10, 6))
    sns.countplot(x=df.groupby('CustomerID')['InteractionID'].transform('count'))
    plt.title('Interactions per Customer', fontsize=14)
    plt.xlabel('Number of Interactions')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'interactions_per_customer.png', dpi=150)
    plt.close()
    print("Saved: interactions_per_customer.png")


def run_eda():
    """Run complete EDA pipeline."""
    create_directories(FIGURES_DIR)
    
    # Load data
    print_separator("LOADING DATA")
    df = load_data()
    
    # Basic analysis
    analyze_basic_info(df)
    analyze_missing_values(df)
    analyze_interaction_nulls(df)
    analyze_categorical_values(df)
    
    # Generate plots
    print_separator("GENERATING VISUALIZATIONS")
    
    plot_churn_distribution(df)
    plot_correlation_heatmap(df)
    plot_amount_spent_distribution(df)
    plot_login_frequency_distribution(df)
    plot_churn_by_product_category(df)
    plot_churn_by_interaction_type(df)
    plot_churn_by_resolution_status(df)
    plot_churn_by_service_usage(df)
    plot_transactions_over_time(df)
    plot_interactions_over_time(df)
    plot_amount_spent_by_churn(df)
    plot_days_since_login_by_churn(df)
    plot_interactions_per_customer(df)
    
    print_separator("EDA COMPLETE")
    print(f"All visualizations saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    run_eda()
