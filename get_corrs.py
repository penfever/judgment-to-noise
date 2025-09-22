#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import re
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure compatibility with numpy 2.0+
if not hasattr(np, 'NAN'):
    np.NAN = np.nan

def extract_score_column_name(filename):
    # Extract score column from filename if there's a pattern like 'factor_completeness_score'
    match = re.search(r'factor_(\w+)_score', filename)
    if match:
        return f"{match.group(1)}_score"
    return "score"  # Default column name

def load_and_join_scores(directory_path):
    """
    Load all CSV files in the directory, extract the 'model' column and any 'score' column,
    and join them into a single dataframe.
    """
    directory = Path(directory_path)
    csv_files = list(directory.glob("*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {directory_path}")
        return None
    
    # Dictionary to store dataframes from each file
    dfs = {}
    has_base_score = False
    
    for csv_file in csv_files:
        # Skip the score_correlations CSV if we're running the script again
        if "score_correlations" in csv_file.name:
            continue
            
        # Read the CSV file
        df = pd.read_csv(csv_file)
        
        # Ensure there's a 'model' column
        if 'model' not in df.columns:
            print(f"Skipping {csv_file.name} - no 'model' column found")
            continue
        
        # Identify the score column (might have a prefix)
        score_column = extract_score_column_name(csv_file.name)
        actual_score_column = None
        
        # Find the actual score column in the dataframe
        for col in df.columns:
            if col == score_column or col.endswith("_score"):
                actual_score_column = col
                break
        
        if not actual_score_column:
            actual_score_column = "score" if "score" in df.columns else None
            
        if not actual_score_column:
            print(f"Skipping {csv_file.name} - no score column found")
            continue
        
        # Extract only model and score columns
        column_name = actual_score_column
        if column_name == "score":
            # If there's a non-prefixed "score" column, make sure we always name it "score"
            # to ensure it appears first in our correlation matrix
            has_base_score = True
        elif "_score" not in column_name:
            # If it's another scoring column without standard naming, rename appropriately
            column_name = extract_score_column_name(csv_file.name)
            
        # Create a new dataframe with just the model and score column
        new_df = df[['model', actual_score_column]].copy()
        new_df.rename(columns={actual_score_column: column_name}, inplace=True)
        
        # Store the dataframe
        dfs[column_name] = new_df
    
    if not dfs:
        print("No valid dataframes were created")
        return None
    
    # If we have a base score, start with that dataframe
    if has_base_score and 'score' in dfs:
        result = dfs['score']
        # Remove from the dict so we don't merge it twice
        del dfs['score']
    else:
        # Start with the first dataframe
        result = list(dfs.values())[0]
        key_to_remove = list(dfs.keys())[0]
        del dfs[key_to_remove]
    
    # Merge with the rest of the dataframes on 'model' column
    for column_name, df in dfs.items():
        result = result.merge(df, on='model', how='outer')
    
    return result

def compute_correlations(df):
    """
    Compute correlations between all score columns in the dataframe.
    """
    # Drop the model column for correlation computation
    score_df = df.drop('model', axis=1)
    
    # Reorder columns to ensure 'score' (if present) is the first column
    # and other columns are sorted alphabetically
    if 'score' in score_df.columns:
        other_cols = sorted([col for col in score_df.columns if col != 'score'])
        cols = ['score'] + other_cols
        score_df = score_df[cols]
    else:
        score_df = score_df[sorted(score_df.columns)]
    
    # Remove columns with all NaN values
    score_df = score_df.dropna(axis=1, how='all')
    
    # Remove rows with all NaN values
    score_df = score_df.dropna(axis=0, how='all')
    
    # Compute correlations
    corr = score_df.corr(method='spearman')
    
    # Round to 3 significant digits
    corr = corr.applymap(lambda x: np.round(x, 3))
    
    return corr

def plot_correlation_heatmap(corr_matrix, output_path):
    """
    Create and save a heatmap visualization of the correlation matrix.
    """
    # Remove any empty rows and columns
    corr_matrix = corr_matrix.dropna(how='all').dropna(how='all', axis=1)
    
    # Get dimensions for the figure size - adjust based on the number of variables
    n_vars = len(corr_matrix.columns)
    plt.figure(figsize=(max(8, n_vars * 0.8), max(6, n_vars * 0.7)))
    
    # Create the triangular mask
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    # Create a heatmap
    sns.heatmap(
        corr_matrix, 
        annot=True,
        cmap='coolwarm',
        vmin=-1, 
        vmax=1,
        mask=mask,
        square=True,
        linewidths=.5,
        fmt='.3f',
        annot_kws={"size": 10}
    )
    
    plt.title('Correlation Matrix Heatmap', fontsize=16)
    plt.tight_layout()
    
    # Save the heatmap
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Compute correlations between score columns in CSV files.')
    parser.add_argument('directory', type=str, help='Directory containing CSV files with score columns')
    parser.add_argument('--factor-analysis', action='store_true', 
                        help='Run factor analysis after computing correlations')
    parser.add_argument('--analyze-questions', action='store_true',
                        help='Analyze question reliability when running factor analysis')
    parser.add_argument('--threshold', type=float, default=2.0,
                        help='Threshold for standardized residuals to flag low reliability questions (default: 2.0)')
    args = parser.parse_args()
    
    # Load and join scores
    joined_df = load_and_join_scores(args.directory)
    
    if joined_df is None:
        return
    
    print("Joined dataframe:")
    print(joined_df.head())
    print(f"\nDataframe shape: {joined_df.shape}")
    
    # Compute correlations
    corr_matrix = compute_correlations(joined_df)
    
    print("\nCorrelation matrix (Spearman):")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 120)
    print(corr_matrix)
    
    # Create score_correlations directory if it doesn't exist
    correlations_dir = os.path.join(args.directory, "..", "score_correlations")
    os.makedirs(correlations_dir, exist_ok=True)
    
    # Save the correlation matrix to a CSV file
    output_csv_path = os.path.join(correlations_dir, "score_correlations.csv")
    corr_matrix.to_csv(output_csv_path)
    print(f"\nSaved correlation matrix to {output_csv_path}")
    
    # Create and save heatmap visualization
    output_heatmap_path = os.path.join(correlations_dir, "score_correlations_heatmap.png")
    plot_correlation_heatmap(corr_matrix, output_heatmap_path)
    print(f"Saved correlation heatmap to {output_heatmap_path}")
    
    # Run factor analysis if requested
    if args.factor_analysis and len(joined_df.columns) > 2:  # Need model + at least 2 score columns
        try:
            # Import and run factor analysis
            from factor_analysis import run_factor_analysis
            print("\nRunning factor analysis...")
            run_factor_analysis(joined_df, args.directory, analyze_questions=args.analyze_questions, threshold=args.threshold)
        except ImportError:
            print("\nFactor analysis module not found. Run 'factor_analysis.py' separately.")
        except Exception as e:
            print(f"\nError running factor analysis: {e}")

if __name__ == "__main__":
    main()