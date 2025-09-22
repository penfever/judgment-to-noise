#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from scipy.spatial.distance import cosine
from factor_analyzer import FactorAnalyzer
import json
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Ensure compatibility with numpy 2.0+
if not hasattr(np, 'NAN'):
    np.NAN = np.nan

def load_processed_jsonl_files(directory_path):
    """Load JSONL files and process them for analysis."""
    import json
    
    directory = Path(directory_path)
    jsonl_files = list(directory.glob("*.jsonl"))
    
    if not jsonl_files:
        print(f"No JSONL files found in {directory_path}")
        return None
    
    # List to store dataframes from each file
    dfs = []
    
    # Score mapping for conversion
    score_mapping = {
        '': 3,
        'A>>B': 1,
        'A>B': 2,
        'A=B': 3,
        'B>A': 4,
        'B>>A': 5,
        'A<<B': 5,
        'A<B': 4,
        'B=A': 3,
        'B<A': 2,
        'B<<A': 1
    }
    
    for jsonl_file in jsonl_files:
        print(f"Processing {jsonl_file.name}")
        
        # Extract model name from filename
        model_name = jsonl_file.stem
        if model_name.endswith("_ct"):
            # Skip CT files
            continue
            
        # Read JSONL file and parse into list of dictionaries
        data = []
        with open(jsonl_file, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        
        # If no data was read, skip this file
        if not data:
            print(f"No data found in {jsonl_file.name}")
            continue
            
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Check if the dataframe has the expected structure
        if 'games' not in df.columns:
            print(f"Skipping {jsonl_file.name} - no 'games' column found")
            continue
            
        # Explode the list column so each list item becomes a row
        df_exploded = df.explode('games').reset_index()
        
        # Add model name
        df_exploded['model'] = model_name
        
        # Convert the dictionaries in the exploded column to individual columns
        try:
            df_normalized = pd.json_normalize(df_exploded['games'])
            
            # Get all columns, ensuring unique names
            cols = list(df_normalized.columns)
            # Find duplicate columns
            seen = {}
            dupes = []
            for i, col in enumerate(cols):
                if col in seen:
                    dupes.append((i, col))
                else:
                    seen[col] = i
                    
            # Add a suffix to duplicate columns
            for idx, col in dupes:
                cols[idx] = f"{col}-{str(idx).zfill(2)}"
                
            df_normalized.columns = cols
            
            # Join with original dataframe
            df_final = pd.concat([df_exploded.drop('games', axis=1), df_normalized], axis=1)
            
            # Fill NA values
            df_final = df_final.fillna("")
            
            # Convert scores from text to numeric
            for k in ['score', 'correctness_score', 'safety_score', 'completeness_score', 'conciseness_score', 'style_score']:
                if k in df_final.columns:
                    df_final[k] = df_final[k].astype(str).replace(score_mapping).astype(np.int64)
            
            # Add to list of dataframes
            dfs.append(df_final)
            
        except Exception as e:
            print(f"Error processing {jsonl_file.name}: {e}")
            continue
    
    if not dfs:
        print("No valid dataframes were created")
        return None
        
    # Concatenate all dataframes
    result = pd.concat(dfs, ignore_index=True)
    
    # Keep only the columns we need for analysis
    keep_cols = ['question_id', 'model']
    score_cols = ['score', 'correctness_score', 'safety_score', 'completeness_score', 'conciseness_score', 'style_score']
    for col in score_cols:
        if col in result.columns:
            keep_cols.append(col)
    
    result = result[keep_cols]
    
    return result

def load_table_files(tables_directory_path):
    """Load and combine CSV files from the tables directory."""
    directory = Path(tables_directory_path)
    
    # Pattern to match leaderboard files with factors
    pattern = "arena_hard_leaderboard_*_factor_*_base.csv"
    csv_files = list(directory.glob(pattern))
    
    if not csv_files:
        print(f"No CSV leaderboard files found in {tables_directory_path}")
        return None
    
    # Dictionary to store dataframes for each factor
    factor_dfs = {}
    
    for csv_file in csv_files:
        print(f"Loading {csv_file.name}")
        
        # Extract factor name from filename
        factor_name = None
        score_column = None
        
        if "factor_score_base" in csv_file.name:
            factor_name = "score"
            score_column = "score"
        elif "factor_correctness_score_base" in csv_file.name:
            factor_name = "correctness_score"
            score_column = "correctness_score"
        elif "factor_safety_score_base" in csv_file.name:
            factor_name = "safety_score"
            score_column = "safety_score"
        elif "factor_completeness_score_base" in csv_file.name:
            factor_name = "completeness_score"
            score_column = "completeness_score"
        elif "factor_conciseness_score_base" in csv_file.name:
            factor_name = "conciseness_score"
            score_column = "conciseness_score"
        elif "factor_style_score_base" in csv_file.name:
            factor_name = "style_score"
            score_column = "style_score"
        else:
            # Skip if not a factor file
            continue
        
        # Read the CSV file
        df = pd.read_csv(csv_file)
        
        # Check that the score column exists
        if score_column not in df.columns:
            print(f"Warning: Column '{score_column}' not found in {csv_file.name}")
            print(f"Available columns: {df.columns.tolist()}")
            continue
            
        # Tables use a 0-100 scale, normalize to 1-5 scale
        # 0 -> 1, 25 -> 2, 50 -> 3, 75 -> 4, 100 -> 5
        df['normalized_score'] = 1 + (df[score_column] / 25)
            
        # Create a dataframe with the model and normalized score
        factor_dfs[factor_name] = df[['model', 'normalized_score']]
    
    if not factor_dfs:
        print("No valid factor dataframes were created")
        return None
    
    # Print which factors were found
    print(f"Factors found in tables: {list(factor_dfs.keys())}")
    
    # Merge all factor dataframes on the model column
    result = None
    for factor_name, df in factor_dfs.items():
        df = df.rename(columns={'normalized_score': factor_name})
        if result is None:
            result = df
        else:
            result = pd.merge(result, df, on='model', how='inner')
    
    # Check the final combined dataframe
    print(f"Final table dataframe has columns: {result.columns.tolist()}")
    print(f"Final table dataframe has shape: {result.shape}")
    
    return result

def calculate_integration_bias(df, include_nonlinear=True):
    """
    Calculate metrics to quantify judge's implicit integration process.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing overall score and subscores
    include_nonlinear : bool
        Whether to include non-linear terms in the analysis
        
    Returns:
    --------
    tuple : (dict, pandas.Series, dict)
        - Dictionary of bias metrics
        - Series of implicit weights
        - Dictionary with non-linear model results (if include_nonlinear=True)
    """
    # 1. Get factor columns and overall score
    factor_cols = [col for col in df.columns if 'score' in col.lower() and col != 'score' and col != 'question_id' and col != 'model']
    
    if 'score' not in df.columns or len(factor_cols) < 2:
        print("Error: Need both overall score and at least 2 factor scores")
        return None, None, None
    
    # 2. Create a pivot table to get average scores by model/question
    df_pivot = df.pivot_table(
        index=['question_id', 'model'],
        values=['score'] + factor_cols,
        aggfunc='mean'
    ).reset_index()
    
    # 3. Standardize all scores
    X = df_pivot[factor_cols].apply(lambda x: (x - x.mean()) / x.std())
    y = (df_pivot['score'] - df_pivot['score'].mean()) / df_pivot['score'].std()
    
    # 4. Calculate expected equal weighting
    expected_scores = X.mean(axis=1)
    expected_corr = np.corrcoef(expected_scores, y)[0, 1]
    
    # 5. Linear regression to find actual weights
    linear_model = LinearRegression().fit(X, y)
    predicted_scores_linear = linear_model.predict(X)
    
    # 6. Calculate basic metrics for linear model
    weights = pd.Series(linear_model.coef_, index=factor_cols)
    # Normalize weights to sum to 1 for interpretability
    normalized_weights = weights / np.abs(weights).sum()
    linear_explained_variance = linear_model.score(X, y)
    
    # Calculate weight entropy (using absolute values to handle negative weights)
    abs_weights = np.abs(weights) / np.abs(weights).sum()
    weight_entropy = -(abs_weights * np.log2(abs_weights)).sum()
    max_entropy = np.log2(len(factor_cols))  # Entropy if all weights were equal
    
    # Calculate mean absolute error between expected and actual scores
    mae_linear = np.mean(np.abs(predicted_scores_linear - expected_scores))
    
    # 7. Integration bias metrics for linear model
    integration_bias = {
        'weight_disparity': abs_weights.std() / abs_weights.mean(),  # Higher = more bias
        'unexplained_variance': 1 - linear_explained_variance,  # Higher = more hidden factors
        'weight_entropy_ratio': weight_entropy / max_entropy,  # Lower = more biased weighting
        'equal_weighting_correlation': expected_corr,  # Lower = more different from equal weights
        'mean_absolute_error': mae_linear  # Higher = more deviation from equal weighting
    }
    
    # 8. If requested, include non-linear model analysis
    nonlinear_results = None
    if include_nonlinear:
        from sklearn.preprocessing import PolynomialFeatures
        
        # Create polynomial features (degree 2)
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_poly = poly.fit_transform(X)
        
        # Get feature names for the polynomial features
        poly_feature_names = []
        # Add original features
        for name in factor_cols:
            poly_feature_names.append(name)
        # Add squared terms
        for name in factor_cols:
            poly_feature_names.append(f"{name}²")
        # Add interaction terms
        for i, name1 in enumerate(factor_cols):
            for j, name2 in enumerate(factor_cols):
                if j > i:  # Only upper triangular elements to avoid duplicates
                    poly_feature_names.append(f"{name1}×{name2}")
        
        # Train polynomial regression model
        poly_model = LinearRegression().fit(X_poly, y)
        predicted_scores_poly = poly_model.predict(X_poly)
        
        # Get feature importances for polynomial model
        poly_weights = pd.Series(poly_model.coef_, index=poly_feature_names)
        poly_explained_variance = poly_model.score(X_poly, y)
        
        # Calculate mean absolute error for polynomial model
        mae_poly = np.mean(np.abs(predicted_scores_poly - expected_scores))
        
        # Get importance of non-linear terms
        linear_term_indices = poly_feature_names[:len(factor_cols)]
        nonlinear_term_indices = poly_feature_names[len(factor_cols):]
        
        # Calculate contribution of non-linear terms
        nonlinear_contribution = poly_explained_variance - linear_explained_variance
        nonlinear_proportion = nonlinear_contribution / (1 - linear_explained_variance) if linear_explained_variance < 1 else 0
        
        nonlinear_results = {
            'poly_weights': poly_weights,
            'explained_variance': poly_explained_variance,
            'nonlinear_contribution': nonlinear_contribution,
            'nonlinear_proportion': nonlinear_proportion,
            'unexplained_variance': 1 - poly_explained_variance,
            'mean_absolute_error': mae_poly,
            'linear_weights': poly_weights[linear_term_indices],
            'nonlinear_weights': poly_weights[nonlinear_term_indices],
            'most_important_nonlinear_terms': poly_weights[nonlinear_term_indices].abs().nlargest(5).index.tolist()
        }
        
        # Update the integration bias metrics with polynomial model results
        integration_bias['poly_unexplained_variance'] = 1 - poly_explained_variance
        integration_bias['nonlinear_contribution'] = nonlinear_contribution
        integration_bias['nonlinear_proportion_of_unexplained'] = nonlinear_proportion
    
    return integration_bias, normalized_weights, nonlinear_results

def calculate_integration_bias_from_tables(df_tables, include_nonlinear=True):
    """
    Calculate metrics to quantify judge's implicit integration process from table data.
    
    Parameters:
    -----------
    df_tables : pandas.DataFrame
        DataFrame containing overall score and subscores from tables
    include_nonlinear : bool
        Whether to include non-linear terms in the analysis
        
    Returns:
    --------
    tuple : (dict, pandas.Series, dict)
        - Dictionary of bias metrics
        - Series of implicit weights
        - Dictionary with non-linear model results (if include_nonlinear=True)
    """
    # 1. Get factor columns and overall score
    factor_cols = [col for col in df_tables.columns if 'score' in col.lower() and col != 'score' and col != 'model']
    
    if 'score' not in df_tables.columns or len(factor_cols) < 2:
        print("Error: Need both overall score and at least 2 factor scores")
        return None, None, None
    
    # 2. Standardize all scores
    X = df_tables[factor_cols].apply(lambda x: (x - x.mean()) / x.std())
    y = (df_tables['score'] - df_tables['score'].mean()) / df_tables['score'].std()
    
    # 3. Calculate expected equal weighting
    expected_scores = X.mean(axis=1)
    expected_corr = np.corrcoef(expected_scores, y)[0, 1]
    
    # 4. Linear regression to find actual weights
    linear_model = LinearRegression().fit(X, y)
    predicted_scores_linear = linear_model.predict(X)
    
    # 5. Calculate basic metrics for linear model
    weights = pd.Series(linear_model.coef_, index=factor_cols)
    # Normalize weights to sum to 1 for interpretability
    normalized_weights = weights / np.abs(weights).sum()
    linear_explained_variance = linear_model.score(X, y)
    
    # Calculate weight entropy (using absolute values to handle negative weights)
    abs_weights = np.abs(weights) / np.abs(weights).sum()
    weight_entropy = -(abs_weights * np.log2(abs_weights)).sum()
    max_entropy = np.log2(len(factor_cols))  # Entropy if all weights were equal
    
    # Calculate mean absolute error between expected and actual scores
    mae_linear = np.mean(np.abs(predicted_scores_linear - expected_scores))
    
    # 6. Integration bias metrics for linear model
    integration_bias = {
        'weight_disparity': abs_weights.std() / abs_weights.mean(),  # Higher = more bias
        'unexplained_variance': 1 - linear_explained_variance,  # Higher = more hidden factors
        'weight_entropy_ratio': weight_entropy / max_entropy,  # Lower = more biased weighting
        'equal_weighting_correlation': expected_corr,  # Lower = more different from equal weights
        'mean_absolute_error': mae_linear  # Higher = more deviation from equal weighting
    }
    
    # 7. If requested, include non-linear model analysis
    nonlinear_results = None
    if include_nonlinear:
        from sklearn.preprocessing import PolynomialFeatures
        
        # Create polynomial features (degree 2)
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_poly = poly.fit_transform(X)
        
        # Get feature names for the polynomial features
        poly_feature_names = []
        # Add original features
        for name in factor_cols:
            poly_feature_names.append(name)
        # Add squared terms
        for name in factor_cols:
            poly_feature_names.append(f"{name}²")
        # Add interaction terms
        for i, name1 in enumerate(factor_cols):
            for j, name2 in enumerate(factor_cols):
                if j > i:  # Only upper triangular elements to avoid duplicates
                    poly_feature_names.append(f"{name1}×{name2}")
        
        # Train polynomial regression model
        poly_model = LinearRegression().fit(X_poly, y)
        predicted_scores_poly = poly_model.predict(X_poly)
        
        # Get feature importances for polynomial model
        poly_weights = pd.Series(poly_model.coef_, index=poly_feature_names)
        poly_explained_variance = poly_model.score(X_poly, y)
        
        # Calculate mean absolute error for polynomial model
        mae_poly = np.mean(np.abs(predicted_scores_poly - expected_scores))
        
        # Get importance of non-linear terms
        linear_term_indices = poly_feature_names[:len(factor_cols)]
        nonlinear_term_indices = poly_feature_names[len(factor_cols):]
        
        # Calculate contribution of non-linear terms
        nonlinear_contribution = poly_explained_variance - linear_explained_variance
        nonlinear_proportion = nonlinear_contribution / (1 - linear_explained_variance) if linear_explained_variance < 1 else 0
        
        nonlinear_results = {
            'poly_weights': poly_weights,
            'explained_variance': poly_explained_variance,
            'nonlinear_contribution': nonlinear_contribution,
            'nonlinear_proportion': nonlinear_proportion,
            'unexplained_variance': 1 - poly_explained_variance,
            'mean_absolute_error': mae_poly,
            'linear_weights': poly_weights[linear_term_indices],
            'nonlinear_weights': poly_weights[nonlinear_term_indices],
            'most_important_nonlinear_terms': poly_weights[nonlinear_term_indices].abs().nlargest(5).index.tolist()
        }
        
        # Update the integration bias metrics with polynomial model results
        integration_bias['poly_unexplained_variance'] = 1 - poly_explained_variance
        integration_bias['nonlinear_contribution'] = nonlinear_contribution
        integration_bias['nonlinear_proportion_of_unexplained'] = nonlinear_proportion
    
    return integration_bias, normalized_weights, nonlinear_results

def calculate_factor_loading_alignment(df, n_factors=2):
    """
    Calculate the alignment between factor loadings from overall judgment 
    and the expected loadings from factor-wise judgments.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing overall score and subscores
    n_factors : int
        Number of factors to extract
        
    Returns:
    --------
    dict : Dictionary with alignment metrics
    """
    # 1. Get factor columns and overall score
    factor_cols = [col for col in df.columns if 'score' in col.lower() and col != 'score' and col != 'question_id' and col != 'model']
    
    if 'score' not in df.columns or len(factor_cols) < 2:
        print("Error: Need both overall score and at least 2 factor scores")
        return None
    
    # 2. Create a pivot table to get average scores
    df_pivot = df.pivot_table(
        index=['question_id', 'model'],
        values=['score'] + factor_cols,
        aggfunc='mean'
    ).reset_index()
    
    # 3. Run a regression to get the relative importance of each factor for the overall score
    X = df_pivot[factor_cols].apply(lambda x: (x - x.mean()) / x.std())
    y = (df_pivot['score'] - df_pivot['score'].mean()) / df_pivot['score'].std()
    
    model = LinearRegression().fit(X, y)
    regression_weights = pd.Series(model.coef_, index=factor_cols)
    normalized_weights = regression_weights / np.abs(regression_weights).sum()
    
    # 4. Perform factor analysis on just the factor scores (excluding overall score)
    fa_factors = FactorAnalyzer(rotation='varimax', n_factors=n_factors)
    fa_factors.fit(X)
    factor_loadings = pd.DataFrame(
        fa_factors.loadings_,
        index=factor_cols,
        columns=[f'Factor{i+1}' for i in range(n_factors)]
    )
    
    # Get factor scores for the factor-only analysis
    factor_scores = fa_factors.transform(X)
    
    # 5. Calculate how much each factor predicts the overall score
    # Create a DataFrame with factor scores
    factor_score_df = pd.DataFrame(
        factor_scores,
        columns=[f'Factor{i+1}' for i in range(n_factors)]
    )
    
    # Run regression of overall score on factor scores
    factor_model = LinearRegression().fit(factor_score_df, y)
    factor_importances = pd.Series(factor_model.coef_, index=[f'Factor{i+1}' for i in range(n_factors)])
    
    # Normalize to sum to 1 (absolute values to handle negative importances)
    factor_importances_normalized = factor_importances / np.abs(factor_importances).sum()
    
    # 6. Calculate alignment metrics
    
    # How much of the overall score is explained by the factors?
    overall_explained = factor_model.score(factor_score_df, y)
    
    # How aligned is each factor with the implicit weights?
    factor_alignment = {}
    for i in range(n_factors):
        factor_name = f'Factor{i+1}'
        # Get the loadings for this factor
        these_loadings = factor_loadings[factor_name]
        
        # Dot product of normalized loadings and normalized weights
        # (this measures how aligned this factor is with the implicit weights)
        these_loadings_norm = these_loadings / np.abs(these_loadings).sum()
        alignment = np.dot(these_loadings_norm, normalized_weights)
        factor_alignment[factor_name] = alignment
    
    # 7. Find most aligned factor
    most_aligned = max(factor_alignment, key=factor_alignment.get)
    
    # 8. Calculate overall alignment score (weighted average of factor alignments)
    overall_alignment = np.sum([
        abs(factor_importances_normalized[f]) * factor_alignment[f]
        for f in factor_importances_normalized.index
    ])
    
    alignment_metrics = {
        'regression_weights': normalized_weights.to_dict(),
        'factor_loadings': factor_loadings.to_dict(),
        'factor_importances': factor_importances_normalized.to_dict(),
        'factor_alignment': factor_alignment,
        'most_aligned_factor': most_aligned,
        'highest_alignment': factor_alignment[most_aligned],
        'overall_alignment': overall_alignment,
        'overall_explained': overall_explained,
        'loadings_without_overall': factor_loadings
    }
    
    return alignment_metrics

def calculate_factor_loading_alignment_from_tables(df_tables, n_factors=2):
    """
    Calculate the alignment between factor loadings from overall judgment
    and the expected loadings from factor-wise judgments using table data.
    
    Parameters:
    -----------
    df_tables : pandas.DataFrame
        DataFrame containing overall score and subscores from tables
    n_factors : int
        Number of factors to extract
        
    Returns:
    --------
    dict : Dictionary with alignment metrics
    """
    # 1. Get factor columns and overall score
    factor_cols = [col for col in df_tables.columns if 'score' in col.lower() and col != 'score' and col != 'model']
    
    if 'score' not in df_tables.columns or len(factor_cols) < 2:
        print("Error: Need both overall score and at least 2 factor scores")
        return None
    
    # 2. Run a regression to get the relative importance of each factor for the overall score
    X = df_tables[factor_cols].apply(lambda x: (x - x.mean()) / x.std())
    y = (df_tables['score'] - df_tables['score'].mean()) / df_tables['score'].std()
    
    model = LinearRegression().fit(X, y)
    regression_weights = pd.Series(model.coef_, index=factor_cols)
    normalized_weights = regression_weights / np.abs(regression_weights).sum()
    
    # 3. Perform factor analysis on just the factor scores (excluding overall score)
    fa_factors = FactorAnalyzer(rotation='varimax', n_factors=n_factors)
    fa_factors.fit(X)
    factor_loadings = pd.DataFrame(
        fa_factors.loadings_,
        index=factor_cols,
        columns=[f'Factor{i+1}' for i in range(n_factors)]
    )
    
    # Get factor scores for the factor-only analysis
    factor_scores = fa_factors.transform(X)
    
    # 4. Calculate how much each factor predicts the overall score
    # Create a DataFrame with factor scores
    factor_score_df = pd.DataFrame(
        factor_scores,
        columns=[f'Factor{i+1}' for i in range(n_factors)]
    )
    
    # Run regression of overall score on factor scores
    factor_model = LinearRegression().fit(factor_score_df, y)
    factor_importances = pd.Series(factor_model.coef_, index=[f'Factor{i+1}' for i in range(n_factors)])
    
    # Normalize to sum to 1 (absolute values to handle negative importances)
    factor_importances_normalized = factor_importances / np.abs(factor_importances).sum()
    
    # 5. Calculate alignment metrics
    
    # How much of the overall score is explained by the factors?
    overall_explained = factor_model.score(factor_score_df, y)
    
    # How aligned is each factor with the implicit weights?
    factor_alignment = {}
    for i in range(n_factors):
        factor_name = f'Factor{i+1}'
        # Get the loadings for this factor
        these_loadings = factor_loadings[factor_name]
        
        # Dot product of normalized loadings and normalized weights
        # (this measures how aligned this factor is with the implicit weights)
        these_loadings_norm = these_loadings / np.abs(these_loadings).sum()
        alignment = np.dot(these_loadings_norm, normalized_weights)
        factor_alignment[factor_name] = alignment
    
    # 6. Find most aligned factor
    most_aligned = max(factor_alignment, key=factor_alignment.get)
    
    # 7. Calculate overall alignment score (weighted average of factor alignments)
    overall_alignment = np.sum([
        abs(factor_importances_normalized[f]) * factor_alignment[f]
        for f in factor_importances_normalized.index
    ])
    
    alignment_metrics = {
        'regression_weights': normalized_weights.to_dict(),
        'factor_loadings': factor_loadings.to_dict(),
        'factor_importances': factor_importances_normalized.to_dict(),
        'factor_alignment': factor_alignment,
        'most_aligned_factor': most_aligned,
        'highest_alignment': factor_alignment[most_aligned],
        'overall_alignment': overall_alignment,
        'overall_explained': overall_explained,
        'loadings_without_overall': factor_loadings
    }
    
    return alignment_metrics

def plot_factor_importance_radar(weights, output_path=None):
    """
    Create a radar chart showing the relative importance of each factor.
    
    Parameters:
    -----------
    weights : pandas.Series
        Series containing factor weights
    output_path : str, optional
        Path to save the figure. If None, the figure is shown but not saved.
    """
    # Normalize weights to [0, 1] range
    normed_weights = (weights - weights.min()) / (weights.max() - weights.min())
    
    # For negative weights, use absolute values but indicate with different color
    is_negative = weights < 0
    abs_weights = np.abs(weights)
    normed_abs_weights = (abs_weights - abs_weights.min()) / (abs_weights.max() - abs_weights.min())
    
    # Compute theoretical equal weights
    n_factors = len(weights)
    equal_weights = pd.Series([1/n_factors] * n_factors, index=weights.index)
    
    # Set up the radar chart
    angles = np.linspace(0, 2 * np.pi, n_factors, endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    # Add the values for the chart (append first value to close the loop)
    values = normed_abs_weights.values.tolist()
    values += values[:1]
    equal_values = equal_weights.values.tolist()
    equal_values += equal_values[:1]
    labels = normed_abs_weights.index.tolist()
    labels += labels[:1]
    is_negative = is_negative.tolist()
    is_negative += is_negative[:1]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
    
    # Plot the actual weights
    ax.plot(angles, values, 'o-', linewidth=2, label='Actual Weights')
    
    # Use different colors for positive and negative weights
    for i, (angle, value, is_neg) in enumerate(zip(angles, values, is_negative)):
        # Skip the last point which is duplicate of the first
        if i == len(angles) - 1:
            continue
        color = 'red' if is_neg else 'green'
        ax.plot([angles[i], angles[i]], [0, value], color=color, linewidth=2)
    
    # Plot the equal weights for comparison
    ax.plot(angles, equal_values, 'k--', linewidth=1, alpha=0.5, label='Equal Weights')
    
    # Fill area for actual weights
    ax.fill(angles, values, alpha=0.25)
    
    # Set the labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels[:-1])
    
    # Add a title and a legend
    ax.set_title('Factor Importance in Judge\'s Decision-Making', fontsize=15)
    ax.legend(loc='upper right')
    
    # Adjust the layout and show or save the plot
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved radar chart to {output_path}")
    else:
        plt.show()
    
    plt.close()

def plot_weight_comparison(raw_weights, table_weights, output_path=None):
    """
    Create a bar chart comparing factor weights from raw judgments and table data.
    
    Parameters:
    -----------
    raw_weights : pandas.Series
        Series containing factor weights from raw judgments
    table_weights : pandas.Series
        Series containing factor weights from table data
    output_path : str, optional
        Path to save the figure. If None, the figure is shown but not saved.
    """
    # Create a DataFrame for comparison
    comparison_df = pd.DataFrame({
        'Raw Judgments': raw_weights,
        'Table Rankings': table_weights
    })
    
    # Sort by the raw judgment weights
    comparison_df = comparison_df.sort_values('Raw Judgments', ascending=False)
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    comparison_df.plot(kind='bar', figsize=(12, 6))
    plt.title('Factor Weight Comparison: Raw Judgments vs. Table Rankings', fontsize=15)
    plt.ylabel('Normalized Weight', fontsize=12)
    plt.xlabel('Factor', fontsize=12)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved weight comparison chart to {output_path}")
    else:
        plt.show()
    
    plt.close()
    
    # Create a secondary visualization: weight difference plot
    plt.figure(figsize=(12, 6))
    
    # Calculate difference (raw - table)
    comparison_df['Difference'] = comparison_df['Raw Judgments'] - comparison_df['Table Rankings']
    
    # Sort by absolute difference
    comparison_df = comparison_df.reindex(comparison_df['Difference'].abs().sort_values(ascending=False).index)
    
    # Plot the differences
    bars = plt.bar(comparison_df.index, comparison_df['Difference'], color=['g' if x >= 0 else 'r' for x in comparison_df['Difference']])
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., 
                 height + 0.01 * (1 if height >= 0 else -1),
                 f'{height:.3f}',
                 ha='center', va='bottom' if height >= 0 else 'top', fontsize=9)
    
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.5)
    plt.title('Factor Weight Differences: Raw Judgments - Table Rankings', fontsize=15)
    plt.ylabel('Weight Difference', fontsize=12)
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    
    # Save the difference plot
    if output_path:
        diff_path = output_path.replace('.png', '_differences.png')
        plt.savefig(diff_path, dpi=300, bbox_inches='tight')
        print(f"Saved weight differences chart to {diff_path}")
    else:
        plt.show()
    
    plt.close()

def plot_bias_metrics_comparison(raw_metrics, table_metrics, output_path=None):
    """
    Create a bar chart comparing bias metrics from raw judgments and table data.
    
    Parameters:
    -----------
    raw_metrics : dict
        Dictionary containing bias metrics from raw judgments
    table_metrics : dict
        Dictionary containing bias metrics from table data
    output_path : str, optional
        Path to save the figure. If None, the figure is shown but not saved.
    """
    # Create a DataFrame for comparison
    metrics_to_include = ['weight_disparity', 'unexplained_variance', 'weight_entropy_ratio']
    raw_values = [raw_metrics[m] for m in metrics_to_include]
    table_values = [table_metrics[m] for m in metrics_to_include]
    
    comparison_df = pd.DataFrame({
        'Raw Judgments': raw_values,
        'Table Rankings': table_values
    }, index=metrics_to_include)
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    comparison_df.plot(kind='bar', figsize=(12, 6))
    plt.title('Bias Metrics Comparison: Raw Judgments vs. Table Rankings', fontsize=15)
    plt.ylabel('Metric Value', fontsize=12)
    plt.xlabel('Metric', fontsize=12)
    plt.grid(True, axis='y', alpha=0.3)
    
    # Add value labels on top of bars
    for i, (metric, values) in enumerate(comparison_df.iterrows()):
        for j, value in enumerate(values):
            plt.text(i - 0.2 + j*0.4, value + 0.02, f'{value:.3f}',
                     ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved bias metrics comparison chart to {output_path}")
    else:
        plt.show()
    
    plt.close()

def plot_nonlinear_importance(nonlinear_results, output_path):
    """
    Create a bar chart showing the most important non-linear terms.
    
    Parameters:
    -----------
    nonlinear_results : dict
        Dictionary containing non-linear model results
    output_path : str
        Path to save the figure
    """
    # Get the top non-linear terms by importance
    nonlinear_weights = nonlinear_results['nonlinear_weights']
    top_terms = nonlinear_weights.abs().nlargest(10)
    
    plt.figure(figsize=(12, 8))
    
    # Create the bar chart with colors based on weight sign
    colors = ['green' if w > 0 else 'red' for w in top_terms]
    bars = plt.barh(top_terms.index, top_terms.values, color=colors)
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        plt.text(width * 1.05, bar.get_y() + bar.get_height()/2, 
                f'{width:.3f}', ha='left', va='center', fontsize=10)
    
    # Add title and labels
    plt.title('Top 10 Most Important Non-linear Terms', fontsize=14)
    plt.xlabel('Coefficient Magnitude', fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.3)
    
    # Add a legend for coefficient signs
    plt.figtext(0.5, 0.01, 
                f"R² improvement from non-linear terms: {nonlinear_results['nonlinear_contribution']:.4f} " + 
                f"({nonlinear_results['nonlinear_proportion']:.1%} of previously unexplained variance)", 
                ha="center", fontsize=11, 
                bbox={"boxstyle": "round,pad=0.5", "facecolor": "lightgrey", "alpha": 0.8})
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_nonlinear_variance_comparison(raw_nonlinear, table_nonlinear, output_path):
    """
    Create a visualization comparing linear and non-linear variance explained.
    
    Parameters:
    -----------
    raw_nonlinear : dict
        Dictionary containing non-linear results for raw judgments
    table_nonlinear : dict
        Dictionary containing non-linear results for table rankings
    output_path : str
        Path to save the figure
    """
    # Create data for the plot
    df = pd.DataFrame({
        'Raw Linear': [1 - raw_nonlinear['unexplained_variance'] - raw_nonlinear['nonlinear_contribution']],
        'Raw Nonlinear': [raw_nonlinear['nonlinear_contribution']],
        'Raw Unexplained': [raw_nonlinear['unexplained_variance']],
        'Table Linear': [1 - table_nonlinear['unexplained_variance'] - table_nonlinear['nonlinear_contribution']],
        'Table Nonlinear': [table_nonlinear['nonlinear_contribution']],
        'Table Unexplained': [table_nonlinear['unexplained_variance']]
    }, index=['Variance'])
    
    # Transpose for better visualization
    df = df.T.reset_index()
    df.columns = ['Source', 'Value']
    
    # Split into categories
    df['Type'] = 'Unexplained'
    df.loc[df['Source'].str.contains('Linear'), 'Type'] = 'Linear'
    df.loc[df['Source'].str.contains('Nonlinear'), 'Type'] = 'Nonlinear'
    
    df['Dataset'] = 'Table'
    df.loc[df['Source'].str.contains('Raw'), 'Dataset'] = 'Raw'
    
    # Create plot
    plt.figure(figsize=(10, 6))
    
    # Plot grouped bars
    sns.barplot(
        x='Dataset', 
        y='Value', 
        hue='Type', 
        data=df,
        palette={'Linear': '#1f77b4', 'Nonlinear': '#2ca02c', 'Unexplained': '#d62728'}
    )
    
    # Add text labels
    for i, row in df.iterrows():
        if row['Value'] > 0.03:  # Only label if value is large enough
            plt.text(
                0 if row['Dataset'] == 'Raw' else 1,  # x-position
                row['Value']/2 + df.loc[(df['Dataset'] == row['Dataset']) & (df['Type'] < row['Type']), 'Value'].sum(),  # y-position
                f"{row['Value']:.3f}",
                ha='center',
                va='center',
                color='white',
                fontweight='bold'
            )
    
    # Customize plot
    plt.title('Variance Explained: Linear vs Non-linear Models', fontsize=14)
    plt.ylabel('Proportion of Variance', fontsize=12)
    plt.ylim(0, 1.05)
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def run_integration_bias_analysis(df, df_tables, output_dir):
    """
    Run the full integration bias analysis, comparing raw judgments and table data.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing overall score and subscores from raw judgments
    df_tables : pandas.DataFrame
        DataFrame containing overall score and subscores from table data
    output_dir : str
        Directory to save output files
    """
    print("\nRunning integration bias analysis...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Calculate integration bias metrics for raw judgments
    raw_bias_metrics, raw_weights, raw_nonlinear = calculate_integration_bias(df)
    
    if raw_bias_metrics is None:
        print("Failed to calculate integration bias metrics for raw judgments")
        return

    # 2. Calculate integration bias metrics for table data
    table_bias_metrics, table_weights, table_nonlinear = calculate_integration_bias_from_tables(df_tables)
    
    if table_bias_metrics is None:
        print("Failed to calculate integration bias metrics for table data")
        return
    
    # 3. Calculate factor loading alignment for raw judgments
    raw_alignment_metrics = calculate_factor_loading_alignment(df)
    
    if raw_alignment_metrics is None:
        print("Failed to calculate factor loading alignment metrics for raw judgments")
        return
    
    # 4. Calculate factor loading alignment for table data
    table_alignment_metrics = calculate_factor_loading_alignment_from_tables(df_tables)
    
    if table_alignment_metrics is None:
        print("Failed to calculate factor loading alignment metrics for table data")
        return
    
    # 5. Save the weights and bias metrics to a JSON file
    results = {
        'raw_judgments': {
            'integration_bias': raw_bias_metrics,
            'factor_weights': raw_weights.to_dict(),
            'regression_weights': raw_alignment_metrics['regression_weights'],
            'factor_importances': raw_alignment_metrics['factor_importances'],
            'factor_alignment': raw_alignment_metrics['factor_alignment'],
            'most_aligned_factor': raw_alignment_metrics['most_aligned_factor'],
            'highest_alignment': raw_alignment_metrics['highest_alignment'],
            'overall_alignment': raw_alignment_metrics['overall_alignment'],
            'overall_explained': raw_alignment_metrics['overall_explained']
        },
        'table_rankings': {
            'integration_bias': table_bias_metrics,
            'factor_weights': table_weights.to_dict(),
            'regression_weights': table_alignment_metrics['regression_weights'],
            'factor_importances': table_alignment_metrics['factor_importances'],
            'factor_alignment': table_alignment_metrics['factor_alignment'],
            'most_aligned_factor': table_alignment_metrics['most_aligned_factor'],
            'highest_alignment': table_alignment_metrics['highest_alignment'],
            'overall_alignment': table_alignment_metrics['overall_alignment'],
            'overall_explained': table_alignment_metrics['overall_explained']
        },
        'comparison': {
            'weight_correlation': np.corrcoef(raw_weights, table_weights)[0, 1],
            'weight_difference_mean': np.mean(np.abs(raw_weights - table_weights)),
            'weight_difference_max': np.max(np.abs(raw_weights - table_weights)),
            'weight_difference_by_factor': (raw_weights - table_weights).to_dict(),
            'explained_variance_difference': raw_alignment_metrics['overall_explained'] - table_alignment_metrics['overall_explained']
        }
    }
    
    # Add nonlinear results if available
    if raw_nonlinear is not None:
        results['raw_judgments']['nonlinear'] = {
            'explained_variance': raw_nonlinear['explained_variance'],
            'nonlinear_contribution': raw_nonlinear['nonlinear_contribution'],
            'nonlinear_proportion': raw_nonlinear['nonlinear_proportion'],
            'most_important_nonlinear_terms': raw_nonlinear['most_important_nonlinear_terms']
        }
    
    if table_nonlinear is not None:
        results['table_rankings']['nonlinear'] = {
            'explained_variance': table_nonlinear['explained_variance'],
            'nonlinear_contribution': table_nonlinear['nonlinear_contribution'],
            'nonlinear_proportion': table_nonlinear['nonlinear_proportion'],
            'most_important_nonlinear_terms': table_nonlinear['most_important_nonlinear_terms']
        }
    
    # Print the results
    print("\n----- RAW JUDGMENTS ANALYSIS -----")
    print("\nIntegration Bias Metrics (Raw Judgments):")
    for metric, value in raw_bias_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nNormalized Factor Weights (Raw Judgments):")
    for factor, weight in raw_weights.items():
        print(f"  {factor}: {weight:.4f}")
    
    print("\nFactor Alignment Metrics (Raw Judgments):")
    print(f"  Most aligned factor: {raw_alignment_metrics['most_aligned_factor']}")
    print(f"  Highest alignment: {raw_alignment_metrics['highest_alignment']:.4f}")
    print(f"  Overall alignment score: {raw_alignment_metrics['overall_alignment']:.4f}")
    print(f"  Variance explained by factors: {raw_alignment_metrics['overall_explained']:.4f}")
    
    if raw_nonlinear is not None:
        print("\nNon-linear Model Results (Raw Judgments):")
        print(f"  Linear model R²: {1 - raw_bias_metrics['unexplained_variance']:.4f}")
        print(f"  Non-linear model R²: {raw_nonlinear['explained_variance']:.4f}")
        print(f"  R² improvement from non-linear terms: {raw_nonlinear['nonlinear_contribution']:.4f}")
        print(f"  Proportion of unexplained variance captured by non-linear terms: {raw_nonlinear['nonlinear_proportion']:.4f}")
        print("\nTop 5 most important non-linear terms:")
        for i, term in enumerate(raw_nonlinear['most_important_nonlinear_terms']):
            coef = raw_nonlinear['nonlinear_weights'][term]
            print(f"  {i+1}. {term}: {coef:.4f}")
    
    print("\n----- TABLE RANKINGS ANALYSIS -----")
    print("\nIntegration Bias Metrics (Table Rankings):")
    for metric, value in table_bias_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nNormalized Factor Weights (Table Rankings):")
    for factor, weight in table_weights.items():
        print(f"  {factor}: {weight:.4f}")
    
    print("\nFactor Alignment Metrics (Table Rankings):")
    print(f"  Most aligned factor: {table_alignment_metrics['most_aligned_factor']}")
    print(f"  Highest alignment: {table_alignment_metrics['highest_alignment']:.4f}")
    print(f"  Overall alignment score: {table_alignment_metrics['overall_alignment']:.4f}")
    print(f"  Variance explained by factors: {table_alignment_metrics['overall_explained']:.4f}")
    
    if table_nonlinear is not None:
        print("\nNon-linear Model Results (Table Rankings):")
        print(f"  Linear model R²: {1 - table_bias_metrics['unexplained_variance']:.4f}")
        print(f"  Non-linear model R²: {table_nonlinear['explained_variance']:.4f}")
        print(f"  R² improvement from non-linear terms: {table_nonlinear['nonlinear_contribution']:.4f}")
        print(f"  Proportion of unexplained variance captured by non-linear terms: {table_nonlinear['nonlinear_proportion']:.4f}")
        print("\nTop 5 most important non-linear terms:")
        for i, term in enumerate(table_nonlinear['most_important_nonlinear_terms']):
            coef = table_nonlinear['nonlinear_weights'][term]
            print(f"  {i+1}. {term}: {coef:.4f}")
    
    print("\n----- COMPARISON BETWEEN RAW JUDGMENTS AND TABLE RANKINGS -----")
    print(f"  Weight correlation: {results['comparison']['weight_correlation']:.4f}")
    print(f"  Mean absolute weight difference: {results['comparison']['weight_difference_mean']:.4f}")
    print(f"  Max absolute weight difference: {results['comparison']['weight_difference_max']:.4f}")
    print("\nWeight differences by factor (Raw - Table):")
    for factor, diff in results['comparison']['weight_difference_by_factor'].items():
        print(f"  {factor}: {diff:.4f}")
    
    # Save the results to a JSON file
    results_path = os.path.join(output_dir, "integration_bias_metrics_comparison.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved integration bias metrics comparison to {results_path}")
    
    # 6. Generate visualizations
    
    # Factor importance radar charts
    raw_radar_path = os.path.join(output_dir, "factor_importance_radar_raw.png")
    plot_factor_importance_radar(raw_weights, raw_radar_path)
    
    table_radar_path = os.path.join(output_dir, "factor_importance_radar_table.png")
    plot_factor_importance_radar(table_weights, table_radar_path)
    
    # Weight comparison chart
    weight_comparison_path = os.path.join(output_dir, "weight_comparison_raw_vs_table.png")
    plot_weight_comparison(raw_weights, table_weights, weight_comparison_path)
    
    # Bias metrics comparison chart
    bias_metrics_path = os.path.join(output_dir, "bias_metrics_comparison.png")
    plot_bias_metrics_comparison(raw_bias_metrics, table_bias_metrics, bias_metrics_path)
    
    # Non-linear importance charts
    if raw_nonlinear is not None:
        raw_nonlinear_path = os.path.join(output_dir, "nonlinear_importance_raw.png")
        plot_nonlinear_importance(raw_nonlinear, raw_nonlinear_path)
    
    if table_nonlinear is not None:
        table_nonlinear_path = os.path.join(output_dir, "nonlinear_importance_table.png")
        plot_nonlinear_importance(table_nonlinear, table_nonlinear_path)
    
    if raw_nonlinear is not None and table_nonlinear is not None:
        # Compare non-linear variance explained
        nonlinear_variance_path = os.path.join(output_dir, "nonlinear_variance_comparison.png")
        plot_nonlinear_variance_comparison(raw_nonlinear, table_nonlinear, nonlinear_variance_path)
    
    # Factor loadings heatmaps
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        raw_alignment_metrics['loadings_without_overall'],
        annot=True,
        cmap='coolwarm',
        vmin=-1,
        vmax=1,
        linewidths=.5,
        fmt='.2f'
    )
    plt.title('Factor Loadings (Raw Judgments)', fontsize=15)
    plt.tight_layout()
    raw_factor_loadings_path = os.path.join(output_dir, "factor_loadings_raw.png")
    plt.savefig(raw_factor_loadings_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved raw judgments factor loadings heatmap to {raw_factor_loadings_path}")
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        table_alignment_metrics['loadings_without_overall'],
        annot=True,
        cmap='coolwarm',
        vmin=-1,
        vmax=1,
        linewidths=.5,
        fmt='.2f'
    )
    plt.title('Factor Loadings (Table Rankings)', fontsize=15)
    plt.tight_layout()
    table_factor_loadings_path = os.path.join(output_dir, "factor_loadings_table.png")
    plt.savefig(table_factor_loadings_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved table rankings factor loadings heatmap to {table_factor_loadings_path}")
    
    # Create explained variance comparison visualization
    plt.figure(figsize=(8, 6))
    
    # Create data for the chart
    explained_df = pd.DataFrame({
        'Raw Judgments': [raw_alignment_metrics['overall_explained'], 1 - raw_alignment_metrics['overall_explained']],
        'Table Rankings': [table_alignment_metrics['overall_explained'], 1 - table_alignment_metrics['overall_explained']]
    }, index=['Explained Variance', 'Unexplained Variance'])
    
    # Plot stacked bars
    explained_df.plot(kind='bar', stacked=True, figsize=(8, 6), 
                      color=['#2ca02c', '#d62728'])
    
    plt.title('Explained vs. Unexplained Variance Comparison', fontsize=15)
    plt.ylabel('Proportion of Variance', fontsize=12)
    plt.xticks(rotation=0)
    
    # Add value labels
    for i, (source, values) in enumerate(explained_df.T.iterrows()):
        explained = values['Explained Variance']
        unexplained = values['Unexplained Variance']
        plt.text(i, explained/2, f'{explained:.3f}', ha='center', va='center', 
                color='white', fontweight='bold')
        plt.text(i, explained + unexplained/2, f'{unexplained:.3f}', ha='center', va='center', 
                color='white', fontweight='bold')
    
    plt.tight_layout()
    
    # Save the explained variance comparison
    explained_variance_path = os.path.join(output_dir, "explained_variance_comparison.png")
    plt.savefig(explained_variance_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved explained variance comparison chart to {explained_variance_path}")

def main():
    parser = argparse.ArgumentParser(description='Analyze judge integration bias in LLM evaluations.')
    parser.add_argument('directory', type=str, help='Directory containing processed JSONL files with judgments')
    parser.add_argument('--tables-dir', type=str, help='Directory containing table CSV files (defaults to directory/tables)')
    parser.add_argument('--output-dir', type=str, help='Directory to save output files (defaults to input_dir/bias_analysis)')
    parser.add_argument('--raw-only', action='store_true', help='Only analyze raw judgment data (skip table data)')
    args = parser.parse_args()
    
    # Set tables directory
    tables_dir = args.tables_dir if args.tables_dir else os.path.join(args.directory, "tables")
    
    # Set output directory
    output_dir = args.output_dir if args.output_dir else os.path.join(args.directory, "bias_analysis")
    
    # Load raw judgment data
    df = load_processed_jsonl_files(args.directory)
    
    if df is None:
        print("Failed to load raw judgment data")
        return
    
    print(f"Loaded raw judgments dataframe with shape: {df.shape}")
    print(f"Raw judgments columns: {df.columns.tolist()}")
    print("Raw judgments sample:")
    print(df.head())
    
    # Check if we should proceed with table data
    if args.raw_only:
        print("\nSkipping table data analysis as requested with --raw-only")
        
        # Perform analysis on raw judgments only
        print("\nRunning integration bias analysis on raw judgments only...")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Calculate integration bias metrics for raw judgments
        raw_bias_metrics, raw_weights = calculate_integration_bias(df)
        
        if raw_bias_metrics is None:
            print("Failed to calculate integration bias metrics for raw judgments")
            return
        
        # Calculate factor loading alignment for raw judgments
        raw_alignment_metrics = calculate_factor_loading_alignment(df)
        
        if raw_alignment_metrics is None:
            print("Failed to calculate factor loading alignment metrics for raw judgments")
            return
        
        # Print the results
        print("\n----- RAW JUDGMENTS ANALYSIS -----")
        print("\nIntegration Bias Metrics (Raw Judgments):")
        for metric, value in raw_bias_metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        print("\nNormalized Factor Weights (Raw Judgments):")
        for factor, weight in raw_weights.items():
            print(f"  {factor}: {weight:.4f}")
        
        print("\nFactor Alignment Metrics (Raw Judgments):")
        print(f"  Most aligned factor: {raw_alignment_metrics['most_aligned_factor']}")
        print(f"  Highest alignment: {raw_alignment_metrics['highest_alignment']:.4f}")
        print(f"  Overall alignment score: {raw_alignment_metrics['overall_alignment']:.4f}")
        print(f"  Variance explained by factors: {raw_alignment_metrics['overall_explained']:.4f}")
        
        # Save the results to a JSON file
        results = {
            'raw_judgments': {
                'integration_bias': raw_bias_metrics,
                'factor_weights': raw_weights.to_dict(),
                'regression_weights': raw_alignment_metrics['regression_weights'],
                'factor_importances': raw_alignment_metrics['factor_importances'],
                'factor_alignment': raw_alignment_metrics['factor_alignment'],
                'most_aligned_factor': raw_alignment_metrics['most_aligned_factor'],
                'highest_alignment': raw_alignment_metrics['highest_alignment'],
                'overall_alignment': raw_alignment_metrics['overall_alignment'],
                'overall_explained': raw_alignment_metrics['overall_explained']
            }
        }
        
        results_path = os.path.join(output_dir, "integration_bias_metrics_raw.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved integration bias metrics to {results_path}")
        
        # Generate visualizations
        raw_radar_path = os.path.join(output_dir, "factor_importance_radar_raw.png")
        plot_factor_importance_radar(raw_weights, raw_radar_path)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            raw_alignment_metrics['loadings_without_overall'],
            annot=True,
            cmap='coolwarm',
            vmin=-1,
            vmax=1,
            linewidths=.5,
            fmt='.2f'
        )
        plt.title('Factor Loadings (Raw Judgments)', fontsize=15)
        plt.tight_layout()
        raw_factor_loadings_path = os.path.join(output_dir, "factor_loadings_raw.png")
        plt.savefig(raw_factor_loadings_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved raw judgments factor loadings heatmap to {raw_factor_loadings_path}")
        return
    
    # Load table data
    df_tables = load_table_files(tables_dir)
    
    if df_tables is None:
        print("Failed to load table data. Running raw judgment analysis only.")
        main_args = argparse.Namespace(
            directory=args.directory,
            tables_dir=args.tables_dir,
            output_dir=args.output_dir,
            raw_only=True
        )
        main_with_args(main_args)
        return
    
    # Check that df_tables has all required columns
    required_columns = ['score', 'correctness_score', 'safety_score', 'completeness_score', 'conciseness_score', 'style_score']
    missing_columns = [col for col in required_columns if col not in df_tables.columns]
    
    if missing_columns:
        print(f"Table data is missing required columns: {missing_columns}")
        print("Running raw judgment analysis only.")
        main_args = argparse.Namespace(
            directory=args.directory,
            tables_dir=args.tables_dir,
            output_dir=args.output_dir,
            raw_only=True
        )
        main_with_args(main_args)
        return
    
    print(f"Loaded table rankings dataframe with shape: {df_tables.shape}")
    print(f"Table rankings columns: {df_tables.columns.tolist()}")
    print("Table rankings sample:")
    print(df_tables.head())
    
    # Run integration bias analysis
    run_integration_bias_analysis(df, df_tables, output_dir)

def main_with_args(args):
    """Helper function to run main with specific args (for internal use)."""
    parser = argparse.ArgumentParser(description='Analyze judge integration bias in LLM evaluations.')
    parser.parse_args(namespace=args)
    main()

if __name__ == "__main__":
    main()