#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from pathlib import Path
import argparse
import json
from collections import Counter
import re

# Ensure compatibility with numpy 2.0+
if not hasattr(np, 'NAN'):
    np.NAN = np.nan
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, PCA
import textwrap
from sklearn.mixture import GaussianMixture
from scipy.spatial.distance import pdist, squareform

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
    
    # Dictionary to store prompt texts by question_id
    question_prompts = {}
    
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
        
        # Extract question prompts if available
        for _, row in df.iterrows():
            question_id = row['question_id']
            if question_id not in question_prompts and 'games' in row and row['games']:
                for game in row['games']:
                    if 'user_prompt' in game:
                        # Extract the prompt text (strip HTML and other marks if present)
                        prompt_text = game['user_prompt']
                        # Remove HTML-like tags
                        prompt_text = re.sub(r'<\|.*?\|>', '', prompt_text)
                        question_prompts[question_id] = prompt_text
                        break
            
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
        return None, None
        
    # Concatenate all dataframes
    result = pd.concat(dfs, ignore_index=True)
    
    # Keep only the columns we need for analysis
    keep_cols = ['question_id', 'model']
    score_cols = ['score', 'correctness_score', 'safety_score', 'completeness_score', 'conciseness_score', 'style_score']
    for col in score_cols:
        if col in result.columns:
            keep_cols.append(col)
    
    result = result[keep_cols]
    
    return result, question_prompts

def create_question_feature_vectors(df, question_prompts):
    """
    Create feature vectors for each question based on factor score patterns.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing scores for different models and questions
    question_prompts : dict
        Dictionary mapping question_id to prompt text
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with feature vectors for each question
    """
    # Group by question_id to get statistics for each question
    question_stats = []
    
    # Get factor columns
    factor_cols = [col for col in df.columns if 'score' in col.lower() and col != 'question_id' and col != 'model']
    
    # Process each question
    for question_id, group in df.groupby('question_id'):
        # Calculate statistics for this question
        stats = {
            'question_id': question_id,
            'num_models': len(group['model'].unique()),
        }
        
        # Feature 1: How much each factor variance for this question
        for col in factor_cols:
            stats[f'{col}_std'] = group[col].std()
            
        # Feature 2: Average values for each factor
        for col in factor_cols:
            stats[f'{col}_mean'] = group[col].mean()
        
        # Feature 3: Correlation between factors for this question
        if len(factor_cols) > 1:
            corr = group[factor_cols].corr()
            for i, col1 in enumerate(factor_cols):
                for j, col2 in enumerate(factor_cols):
                    if j > i:  # Only upper triangle
                        stats[f'corr_{col1}_{col2}'] = corr.loc[col1, col2]
        
        # Feature 4: R² of overall score explained by factors for this question
        if 'score' in factor_cols:
            other_factors = [col for col in factor_cols if col != 'score']
            if len(other_factors) >= 2:  # Need at least 2 factors for regression
                try:
                    X = group[other_factors]
                    y = group['score']
                    model = LinearRegression().fit(X, y)
                    stats['r2_score'] = model.score(X, y)
                    
                    # Feature 5: Coefficient patterns for this question
                    for i, col in enumerate(other_factors):
                        stats[f'coef_{col}'] = model.coef_[i]
                except:
                    # In case of singular matrix or other errors
                    stats['r2_score'] = np.nan
                    for col in other_factors:
                        stats[f'coef_{col}'] = 0
        
        # Feature 6: Add prompt text if available
        if question_id in question_prompts:
            stats['prompt_text'] = question_prompts[question_id]
        else:
            stats['prompt_text'] = ""
        
        question_stats.append(stats)
    
    # Convert to DataFrame
    question_vectors = pd.DataFrame(question_stats)
    
    return question_vectors

def cluster_questions(question_vectors, n_clusters=5, use_text=True, text_weight=0.5):
    """
    Cluster questions based on their feature vectors.
    
    Parameters:
    -----------
    question_vectors : pandas.DataFrame
        DataFrame with feature vectors for each question
    n_clusters : int
        Number of clusters to create
    use_text : bool
        Whether to use prompt text features
    text_weight : float
        Weight to give to text features (0-1)
        
    Returns:
    --------
    tuple : (pandas.DataFrame, dict)
        - DataFrame with cluster assignments
        - Dictionary with cluster information
    """
    # Get numeric features
    feature_cols = [col for col in question_vectors.columns 
                   if col not in ['question_id', 'prompt_text']]
    
    # Create a copy of data for clustering
    X = question_vectors[feature_cols].copy()
    
    # Replace NaN values with zeros
    X = X.fillna(0)
    
    # Standardize numeric features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Add text features if requested
    if use_text and 'prompt_text' in question_vectors.columns:
        # Create TF-IDF vectors from prompt text
        texts = question_vectors['prompt_text'].fillna("").tolist()
        
        # Create TF-IDF vectors
        tfidf = TfidfVectorizer(max_features=100, stop_words='english')
        text_features = tfidf.fit_transform(texts)
        
        # Reduce dimensionality of text features
        svd = TruncatedSVD(n_components=min(20, text_features.shape[1]), random_state=42)
        text_features_reduced = svd.fit_transform(text_features)
        
        # Standardize text features
        text_scaler = StandardScaler()
        text_features_scaled = text_scaler.fit_transform(text_features_reduced)
        
        # Combine numeric and text features with weighting
        X_combined = np.hstack([
            X_scaled * (1 - text_weight),
            text_features_scaled * text_weight
        ])
    else:
        X_combined = X_scaled
    
    # Try different clustering methods
    
    # K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_combined)
    
    # Calculate silhouette score to evaluate cluster quality
    silhouette = silhouette_score(X_combined, clusters) if n_clusters > 1 else 0
    
    # Create a dataframe with cluster assignments
    result = question_vectors.copy()
    result['cluster'] = clusters
    
    # Get cluster information
    cluster_info = {}
    
    # Get cluster centers in original feature space
    if use_text:
        # For combined features, we need to extract just the numeric part
        centers_numeric = kmeans.cluster_centers_[:, :X_scaled.shape[1]] / (1 - text_weight)
        # Convert back to original scale
        centers_original = scaler.inverse_transform(centers_numeric)
    else:
        centers_original = scaler.inverse_transform(kmeans.cluster_centers_)
    
    # Create cluster info
    for i in range(n_clusters):
        # Count number of questions in this cluster
        cluster_size = sum(clusters == i)
        
        # Get the cluster center in original feature space
        center = centers_original[i]
        
        # Create dictionary mapping feature names to center values
        center_dict = dict(zip(feature_cols, center))
        
        # Get top features for this cluster
        if use_text:
            # For text features, find most distinctive words
            cluster_texts = texts[clusters == i]
            if cluster_texts:
                # Use TF-IDF to find distinctive words
                cluster_tfidf = TfidfVectorizer(max_features=10, stop_words='english')
                try:
                    cluster_tfidf.fit_transform(cluster_texts)
                    top_words = cluster_tfidf.get_feature_names_out()
                except:
                    top_words = []
            else:
                top_words = []
        else:
            top_words = []
        
        # Get example questions from this cluster
        example_ids = result[result['cluster'] == i]['question_id'].values[:5]
        example_prompts = [question_vectors[question_vectors['question_id'] == qid]['prompt_text'].values[0] 
                         for qid in example_ids if qid in question_vectors['question_id'].values]
        
        # Find distinguishing features for this cluster
        # (features where this cluster's center is far from the overall mean)
        feature_means = X.mean()
        feature_stds = X.std()
        
        distinguishing_features = {}
        for j, feat in enumerate(feature_cols):
            # Z-score of this feature's center relative to overall distribution
            if feature_stds[feat] > 0:
                z_score = (center_dict[feat] - feature_means[feat]) / feature_stds[feat]
                if abs(z_score) > 1.0:  # Only include features that are significantly different
                    distinguishing_features[feat] = z_score
        
        # Sort by absolute z-score
        distinguishing_features = dict(sorted(distinguishing_features.items(), 
                                           key=lambda x: abs(x[1]), reverse=True)[:10])
        
        # Store cluster information
        cluster_info[f'Cluster {i}'] = {
            'size': int(cluster_size),
            'percentage': float(cluster_size / len(clusters) * 100),
            'center': center_dict,
            'distinguishing_features': distinguishing_features,
            'top_words': top_words.tolist() if hasattr(top_words, 'tolist') else list(top_words),
            'example_ids': example_ids.tolist(),
            'example_prompts': example_prompts,
            'question_ids': result[result['cluster'] == i]['question_id'].tolist()
        }
    
    # Add overall clustering quality metrics
    cluster_info['metadata'] = {
        'n_clusters': n_clusters,
        'silhouette_score': float(silhouette),
        'use_text': use_text,
        'text_weight': text_weight
    }
    
    return result, cluster_info

def analyze_factor_weights_by_cluster(df, clusters, output_dir):
    """
    Analyze how factor weights vary across different question clusters.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing scores for different models and questions
    clusters : pandas.DataFrame
        DataFrame with cluster assignments for each question
    output_dir : str
        Directory to save output files
        
    Returns:
    --------
    dict : Dictionary with analysis results
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get factor columns (excluding overall score)
    factor_cols = [col for col in df.columns 
                  if 'score' in col.lower() 
                  and col != 'score' 
                  and col != 'question_id' 
                  and col != 'model']
    
    # Dictionary to store results
    results = {
        'cluster_weights': {},
        'explained_variance': {},
        'factor_importance': {}
    }
    
    # Dictionary to store R² values for each cluster
    r2_by_cluster = {}
    
    # Number of clusters
    n_clusters = clusters['cluster'].nunique()
    
    # Set up the plot grid
    fig, axes = plt.subplots(1, n_clusters, figsize=(5 * n_clusters, 4), sharey=True)
    if n_clusters == 1:
        axes = [axes]  # Make iterable if only one cluster
    
    # For each cluster, run a regression and get weights
    for cluster_id in range(n_clusters):
        # Get questions in this cluster
        cluster_questions = clusters[clusters['cluster'] == cluster_id]['question_id'].unique()
        
        # Filter data to only include these questions
        cluster_data = df[df['question_id'].isin(cluster_questions)]
        
        if len(cluster_data) == 0:
            print(f"No data for cluster {cluster_id}")
            continue
        
        # Create a pivot table
        pivot_df = cluster_data.pivot_table(
            index=['question_id', 'model'],
            values=['score'] + factor_cols,
            aggfunc='mean'
        ).reset_index()
        
        # Standardize the data
        X = pivot_df[factor_cols].apply(lambda x: (x - x.mean()) / x.std())
        y = pivot_df['score']
        
        # To handle cases where a factor has zero variance in a cluster
        X = X.fillna(0)
        
        # Run a linear regression
        model = LinearRegression()
        model.fit(X, y)
        
        # Get weights and normalize
        weights = pd.Series(model.coef_, index=factor_cols)
        normalized_weights = weights / np.abs(weights).sum()
        
        # Calculate explained variance
        explained_variance = model.score(X, y)
        r2_by_cluster[cluster_id] = explained_variance
        
        # Store results
        results['cluster_weights'][f'Cluster {cluster_id}'] = normalized_weights.to_dict()
        results['explained_variance'][f'Cluster {cluster_id}'] = explained_variance
        
        # Plot the weights
        ax = axes[cluster_id]
        colors = ['green' if w >= 0 else 'red' for w in normalized_weights]
        bars = ax.bar(normalized_weights.index, normalized_weights, color=colors)
        
        # Add labels
        for bar in bars:
            height = bar.get_height()
            if abs(height) > 0.05:  # Only label significant weights
                ax.text(bar.get_x() + bar.get_width()/2., 
                        height + 0.01 if height >= 0 else height - 0.05, 
                        f'{height:.2f}', 
                        ha='center', va='bottom' if height >= 0 else 'top',
                        fontsize=8)
        
        # Set title and labels
        cluster_size = len(cluster_questions)
        total_questions = len(clusters)
        ax.set_title(f'Cluster {cluster_id}\n({cluster_size} questions, {cluster_size/total_questions:.1%})\nR²: {explained_variance:.2f}', 
                     fontsize=12)
        ax.set_ylim(-0.6, 1.0)  # Set y-limits for consistency
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
        # Rotate x-labels
        ax.set_xticklabels(normalized_weights.index, rotation=45, ha='right')
    
    plt.tight_layout()
    weights_path = os.path.join(output_dir, f"factor_weights_by_cluster.png")
    plt.savefig(weights_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Calculate the weighted average R² across all clusters
    cluster_sizes = clusters['cluster'].value_counts().sort_index()
    total_questions = len(clusters)
    weighted_r2 = sum(r2_by_cluster[i] * (cluster_sizes[i] / total_questions) 
                      for i in range(n_clusters) if i in r2_by_cluster)
    
    # Find the overall R² (without clustering)
    pivot_df = df.pivot_table(
        index=['question_id', 'model'],
        values=['score'] + factor_cols,
        aggfunc='mean'
    ).reset_index()
    
    X = pivot_df[factor_cols].apply(lambda x: (x - x.mean()) / x.std())
    X = X.fillna(0)  # Handle zero variance
    y = pivot_df['score']
    
    model = LinearRegression()
    model.fit(X, y)
    overall_r2 = model.score(X, y)
    
    # Calculate the improvement in R² from clustering
    r2_improvement = weighted_r2 - overall_r2
    
    # Calculate how much of the unexplained variance is now explained
    if overall_r2 < 1:
        proportion_unexplained_captured = r2_improvement / (1 - overall_r2)
    else:
        proportion_unexplained_captured = 0
    
    # Store overall results
    results['overall'] = {
        'overall_r2': overall_r2,
        'weighted_cluster_r2': weighted_r2,
        'r2_improvement': r2_improvement,
        'proportion_unexplained_captured': proportion_unexplained_captured
    }
    
    # Print results
    print(f"\nFactor Weights Analysis by Cluster:")
    print(f"Overall R² (no clustering): {overall_r2:.4f}")
    print(f"Weighted average R² with clustering: {weighted_r2:.4f}")
    print(f"Improvement in R² from clustering: {r2_improvement:.4f}")
    print(f"Proportion of unexplained variance captured: {proportion_unexplained_captured:.4f}")
    
    # For each cluster, print results
    for i in range(n_clusters):
        if f"Cluster {i}" in results['cluster_weights']:
            print(f"\nCluster {i} (R²: {results['explained_variance'][f'Cluster {i}']:.4f}):")
            weights = results['cluster_weights'][f"Cluster {i}"]
            for factor, weight in sorted(weights.items(), key=lambda x: abs(x[1]), reverse=True):
                print(f"  {factor}: {weight:.4f}")
    
    # Create barplot showing R² by cluster
    plt.figure(figsize=(10, 6))
    cluster_ids = [f"Cluster {i}" for i in range(n_clusters) if f"Cluster {i}" in results['explained_variance']]
    r2_values = [results['explained_variance'][c_id] for c_id in cluster_ids]
    
    # Sort by R²
    sorted_idx = np.argsort(r2_values)[::-1]
    cluster_ids = [cluster_ids[i] for i in sorted_idx]
    r2_values = [r2_values[i] for i in sorted_idx]
    
    # Add overall and weighted average
    cluster_ids.append("Overall (no clustering)")
    r2_values.append(overall_r2)
    cluster_ids.append("Weighted Avg w/ Clustering")
    r2_values.append(weighted_r2)
    
    # Create barplot
    bars = plt.bar(cluster_ids, r2_values, color=['skyblue'] * n_clusters + ['orange', 'green'])
    plt.axhline(y=overall_r2, color='orange', linestyle='--', alpha=0.7)
    plt.axhline(y=weighted_r2, color='green', linestyle='--', alpha=0.7)
    
    # Add labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom')
    
    plt.title('R² by Cluster: How Well Factors Explain Overall Score', fontsize=14)
    plt.ylabel('R² (Explained Variance)', fontsize=12)
    plt.ylim(0, 1.0)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    r2_path = os.path.join(output_dir, f"r2_by_cluster.png")
    plt.savefig(r2_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save results to JSON
    results_path = os.path.join(output_dir, f"factor_analysis_by_cluster.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

def plot_cluster_characteristics(clusters, cluster_info, output_dir):
    """
    Create visualizations showing the characteristics of each cluster.
    
    Parameters:
    -----------
    clusters : pandas.DataFrame
        DataFrame with cluster assignments
    cluster_info : dict
        Dictionary with cluster information
    output_dir : str
        Directory to save output files
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Number of clusters
    n_clusters = clusters['cluster'].nunique()
    
    # 1. Plot cluster sizes
    plt.figure(figsize=(10, 6))
    
    # Extract cluster sizes
    cluster_ids = [f"Cluster {i}" for i in range(n_clusters)]
    cluster_sizes = [cluster_info[c_id]['size'] for c_id in cluster_ids]
    
    # Create barplot
    plt.bar(cluster_ids, cluster_sizes, color='skyblue')
    
    # Add percentages as labels
    for i, size in enumerate(cluster_sizes):
        percentage = size / sum(cluster_sizes) * 100
        plt.text(i, size + 5, f'{percentage:.1f}%', ha='center')
    
    plt.title('Cluster Sizes', fontsize=14)
    plt.ylabel('Number of Questions', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    sizes_path = os.path.join(output_dir, f"cluster_sizes.png")
    plt.savefig(sizes_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Create profile plots showing distinguishing features for each cluster
    plt.figure(figsize=(15, 10))
    
    # Get common distinguishing features across all clusters
    common_features = set()
    for i in range(n_clusters):
        features = cluster_info[f'Cluster {i}']['distinguishing_features'].keys()
        common_features.update(features)
    
    common_features = list(common_features)
    if len(common_features) > 15:
        # If too many features, select the most important ones
        feature_importance = {}
        for feature in common_features:
            max_abs_score = 0
            for i in range(n_clusters):
                if feature in cluster_info[f'Cluster {i}']['distinguishing_features']:
                    abs_score = abs(cluster_info[f'Cluster {i}']['distinguishing_features'][feature])
                    max_abs_score = max(max_abs_score, abs_score)
            feature_importance[feature] = max_abs_score
        
        # Select top 15 features
        common_features = sorted(feature_importance.keys(), 
                               key=lambda x: feature_importance[x], reverse=True)[:15]
    
    # Create a matrix of z-scores for each cluster and feature
    z_scores = np.zeros((n_clusters, len(common_features)))
    for i in range(n_clusters):
        for j, feature in enumerate(common_features):
            if feature in cluster_info[f'Cluster {i}']['distinguishing_features']:
                z_scores[i, j] = cluster_info[f'Cluster {i}']['distinguishing_features'][feature]
    
    # Create heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(z_scores, annot=True, cmap='coolwarm', center=0,
                xticklabels=[textwrap.fill(f, 15) for f in common_features],
                yticklabels=[f"Cluster {i}" for i in range(n_clusters)],
                fmt='.1f', linewidths=.5)
    
    plt.title('Cluster Feature Profiles (Z-scores)', fontsize=14)
    plt.tight_layout()
    
    profile_path = os.path.join(output_dir, f"cluster_profiles.png")
    plt.savefig(profile_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Word clouds for text-based clusters (if available)
    has_text = any([len(cluster_info[f'Cluster {i}']['top_words']) > 0 for i in range(n_clusters)])
    
    if has_text:
        try:
            from wordcloud import WordCloud
            
            plt.figure(figsize=(15, 3 * n_clusters))
            
            for i in range(n_clusters):
                top_words = cluster_info[f'Cluster {i}']['top_words']
                
                if not top_words:
                    continue
                
                # Create word frequencies
                word_freqs = {word: 10 - j for j, word in enumerate(top_words) if j < 10}
                
                # Generate word cloud
                wordcloud = WordCloud(width=800, height=400, background_color='white',
                                      max_words=10, prefer_horizontal=1.0)
                wordcloud.generate_from_frequencies(word_freqs)
                
                # Plot
                plt.subplot(n_clusters, 1, i + 1)
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.title(f'Cluster {i} - Top Words', fontsize=12)
                plt.axis('off')
            
            plt.tight_layout()
            wordcloud_path = os.path.join(output_dir, f"cluster_wordclouds.png")
            plt.savefig(wordcloud_path, dpi=300, bbox_inches='tight')
            plt.close()
        except ImportError:
            print("WordCloud package not found. Skipping word clouds.")
    
    # 4. Example prompts for each cluster
    with open(os.path.join(output_dir, f"cluster_examples.txt"), 'w') as f:
        f.write("CLUSTER EXAMPLE PROMPTS\n")
        f.write("======================\n\n")
        
        for i in range(n_clusters):
            f.write(f"CLUSTER {i} ({cluster_info[f'Cluster {i}']['size']} questions)\n")
            f.write("-" * 40 + "\n")
            
            # Print distinguishing features
            f.write("Distinguishing Features:\n")
            for feature, score in cluster_info[f'Cluster {i}']['distinguishing_features'].items():
                f.write(f"  {feature}: {score:.2f}\n")
            f.write("\n")
            
            # Print example prompts
            f.write("Example Prompts:\n")
            for j, prompt in enumerate(cluster_info[f'Cluster {i}']['example_prompts'][:3]):
                f.write(f"Example {j+1}:\n")
                f.write(f"{textwrap.fill(prompt, 100)}\n\n")
            
            f.write("\n\n")
    
    # 5. Save detailed cluster info to JSON
    cluster_info_path = os.path.join(output_dir, f"cluster_info.json")
    with open(cluster_info_path, 'w') as f:
        json.dump(cluster_info, f, indent=2)

def analyze_question_clusters(df, question_prompts, output_dir, n_clusters=5, use_text=True):
    """
    Perform a full clustering analysis of questions based on their characteristics.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing scores for different models and questions
    question_prompts : dict
        Dictionary mapping question_id to prompt text
    output_dir : str
        Directory to save output files
    n_clusters : int
        Number of clusters to create
    use_text : bool
        Whether to use prompt text features
    """
    print(f"Creating feature vectors for {len(df['question_id'].unique())} questions...")
    question_vectors = create_question_feature_vectors(df, question_prompts)
    
    print(f"Clustering questions into {n_clusters} clusters...")
    clusters, cluster_info = cluster_questions(question_vectors, n_clusters=n_clusters, use_text=use_text)
    
    print(f"Analyzing factor weights by cluster...")
    factor_analysis = analyze_factor_weights_by_cluster(df, clusters, output_dir)
    
    print(f"Creating visualizations...")
    plot_cluster_characteristics(clusters, cluster_info, output_dir)
    
    # Save the cluster assignments
    clusters_path = os.path.join(output_dir, "question_clusters.csv")
    clusters[['question_id', 'cluster']].to_csv(clusters_path, index=False)
    
    # Return the results
    return {
        'clusters': clusters,
        'cluster_info': cluster_info,
        'factor_analysis': factor_analysis
    }

def main():
    parser = argparse.ArgumentParser(description='Analyze question clusters in LLM evaluations.')
    parser.add_argument('directory', type=str, help='Directory containing processed JSONL files with judgments')
    parser.add_argument('--output-dir', type=str, help='Directory to save output files (defaults to input_dir/cluster_analysis)')
    parser.add_argument('--n-clusters', type=int, default=5, help='Number of clusters to create (default: 5)')
    parser.add_argument('--use-text', action='store_true', help='Whether to use prompt text features')
    args = parser.parse_args()
    
    # Set output directory
    output_dir = args.output_dir if args.output_dir else os.path.join(args.directory, "cluster_analysis")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    df, question_prompts = load_processed_jsonl_files(args.directory)
    
    if df is None:
        print("Failed to load data")
        return
    
    print(f"Loaded dataframe with shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Number of unique questions: {len(df['question_id'].unique())}")
    print(f"Number of questions with prompt text: {len(question_prompts)}")
    
    # Run cluster analysis
    results = analyze_question_clusters(df, question_prompts, output_dir, 
                                     n_clusters=args.n_clusters, use_text=args.use_text)
    
    print(f"\nAnalysis complete. Results saved to {output_dir}")
    
    # Print summary of results
    overall_r2 = results['factor_analysis']['overall']['overall_r2']
    cluster_r2 = results['factor_analysis']['overall']['weighted_cluster_r2']
    improvement = results['factor_analysis']['overall']['r2_improvement']
    proportion = results['factor_analysis']['overall']['proportion_unexplained_captured']
    
    print(f"\nSUMMARY:")
    print(f"Number of questions analyzed: {len(df['question_id'].unique())}")
    print(f"Number of clusters: {args.n_clusters}")
    print(f"Overall R² (no clustering): {overall_r2:.4f}")
    print(f"Weighted average R² with clustering: {cluster_r2:.4f}")
    print(f"Improvement in R² from clustering: {improvement:.4f}")
    print(f"Proportion of unexplained variance captured by clustering: {proportion:.4f}")

if __name__ == "__main__":
    main()