#!/bin/bash
# Flexible LLM Judge Analysis Pipeline
# This script automates the process of running various analysis tools on LLM evaluation data
# with configurable input and output directories

# Enable error handling
set -e

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --input-dir PATH        Path to directory containing JSONL files (required)"
    echo "  --output-dir PATH       Path to output directory (required)"
    echo "  --judge-name NAME       Judge model name (required)"
    echo "  --suffix STRING         Optional suffix for output files"
    echo "  --skip-gen-subscores    Skip gen_subscores.py step (use if data already processed)"
    echo "  --help                  Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --input-dir /path/to/base --output-dir /path/to/results --judge-name DeepSeek-R1-32B"
    echo "  $0 --input-dir /path/to/base_processed_debiased --output-dir /path/to/debiased_results --judge-name DeepSeek-R1-32B --suffix debiased_psychometric --skip-gen-subscores"
}

# Initialize variables
INPUT_DIR=""
OUTPUT_DIR=""
JUDGE_NAME=""
SUFFIX=""
SKIP_GEN_SUBSCORES=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --input-dir)
            INPUT_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --judge-name)
            JUDGE_NAME="$2"
            shift 2
            ;;
        --suffix)
            SUFFIX="$2"
            shift 2
            ;;
        --skip-gen-subscores)
            SKIP_GEN_SUBSCORES=true
            shift
            ;;
        --help)
            show_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$INPUT_DIR" || -z "$OUTPUT_DIR" || -z "$JUDGE_NAME" ]]; then
    echo "Error: Missing required arguments"
    show_usage
    exit 1
fi

# Convert to absolute paths
INPUT_DIR=$(realpath "$INPUT_DIR")
# Create output dir first, then get realpath
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR=$(realpath "$OUTPUT_DIR")

# Check if input directory exists
if [[ ! -d "$INPUT_DIR" ]]; then
    echo "Error: Input directory does not exist: $INPUT_DIR"
    exit 1
fi

# Check if input directory contains JSONL files
if [[ -z "$(find "$INPUT_DIR" -name "*.jsonl" 2>/dev/null)" ]]; then
    echo "Error: No JSONL files found in $INPUT_DIR"
    exit 1
fi

# Create output directory structure
mkdir -p "$OUTPUT_DIR"
PROCESSED_DIR="$OUTPUT_DIR/base_processed"
TABLES_DIR="$OUTPUT_DIR/tables"

mkdir -p "$PROCESSED_DIR"
mkdir -p "$TABLES_DIR"
mkdir -p "$TABLES_DIR/counts"
mkdir -p "$TABLES_DIR/factor_analysis"
mkdir -p "$TABLES_DIR/factor_reliability"
mkdir -p "$TABLES_DIR/factor_scores_original_cis"
mkdir -p "$TABLES_DIR/factor_scores_updated_cis"
mkdir -p "$TABLES_DIR/question_reliability"
mkdir -p "$TABLES_DIR/score_correlations"

echo "=== Flexible LLM Judge Analysis Pipeline ==="
echo "Input directory: $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Judge name: $JUDGE_NAME"
if [[ -n "$SUFFIX" ]]; then
    echo "Suffix: $SUFFIX"
fi
echo "Skip gen_subscores: $SKIP_GEN_SUBSCORES"
echo ""

# Step 1: Run gen_subscores.py (unless skipped)
if [[ "$SKIP_GEN_SUBSCORES" == false ]]; then
    echo "----- Step 1: Running gen_subscores.py -----"
    eval "python gen_subscores.py \"$INPUT_DIR\""
    
    # Check if gen_subscores generated any files
    if [[ ! -d "$PROCESSED_DIR" ]] || [[ -z "$(find "$PROCESSED_DIR" -name "*.jsonl" 2>/dev/null)" ]]; then
        echo "Error: gen_subscores.py failed to generate processed JSONL files."
        echo "Please check the errors above and fix any issues before proceeding."
        exit 1
    fi
else
    echo "----- Step 1: Skipping gen_subscores.py (using existing processed data) -----"
    # Copy/link the input files directly to processed directory
    for jsonl_file in "$INPUT_DIR"/*.jsonl; do
        if [[ -f "$jsonl_file" ]]; then
            cp "$jsonl_file" "$PROCESSED_DIR/"
        fi
    done
    echo "Copied $(find "$PROCESSED_DIR" -name "*.jsonl" | wc -l) JSONL files to processed directory"
fi

# Set environment variables for subsequent steps
export MY_PATH="$PROCESSED_DIR"
export MY_JUDGE="$JUDGE_NAME"
export MY_FACTOR_PATH="$TABLES_DIR"

echo "Environment variables set:"
echo "MY_PATH=$MY_PATH"
echo "MY_JUDGE=$MY_JUDGE"
echo "MY_FACTOR_PATH=$MY_FACTOR_PATH"

# Create leaderboard directory
mkdir -p leaderboard

# Step 2: Run first round of show_result.py for each metric
echo "----- Step 2: Running show_result.py for each metric (original CIs) -----"

# Determine output suffix for files
FILE_SUFFIX=""
if [[ -n "$SUFFIX" ]]; then
    FILE_SUFFIX="_$SUFFIX"
fi

eval "python show_result.py --output --judge-name \"$MY_JUDGE\" --judgment-dir \"$MY_PATH\" --target-metric \"score\"" || {
    echo "Warning: show_result.py for 'score' metric failed. Continuing anyway..."
}

for metric in "safety_score" "style_score" "conciseness_score" "completeness_score" "correctness_score"; do
    echo "Running show_result.py for metric: $metric"
    eval "python show_result.py --output --judge-name \"$MY_JUDGE\" --judgment-dir \"$MY_PATH\" --target-metric \"$metric\"" || {
        echo "Warning: show_result.py for '$metric' failed. Continuing anyway..."
    }
done

# Check if any result files were generated
if [[ -z "$(find leaderboard -name "arena_hard_leaderboard_*_${MY_JUDGE}_judge_*.csv" 2>/dev/null)" ]]; then
    echo "Warning: No leaderboard files were generated. This may indicate a problem with the input data."
    echo "Continuing with the analysis, but results may be incomplete."
else
    # Move the result files to the appropriate directory
    echo "Moving original CI result files to $TABLES_DIR/factor_scores_original_cis"
    find leaderboard -name "arena_hard_leaderboard_*_${MY_JUDGE}_judge_*.csv" -exec mv {} "$TABLES_DIR/factor_scores_original_cis/" \;
    
    # Rename files with suffix if provided
    if [[ -n "$SUFFIX" ]]; then
        for file in "$TABLES_DIR/factor_scores_original_cis"/*.csv; do
            if [[ -f "$file" ]]; then
                base=$(basename "$file" .csv)
                dir=$(dirname "$file")
                mv "$file" "$dir/${base}${FILE_SUFFIX}.csv"
            fi
        done
    fi
fi

# Step 3: Compute correlations between factors
echo "----- Step 3: Computing correlations between factors -----"
eval "python get_corrs.py \"$TABLES_DIR/factor_scores_original_cis\"" || {
    echo "Warning: get_corrs.py failed. This may indicate insufficient data."
    echo "Creating empty correlation files..."
    touch "$TABLES_DIR/factor_scores_original_cis/score_correlations.csv"
    touch "$TABLES_DIR/factor_scores_original_cis/score_correlations_heatmap.png"
}

# Move correlation files to the score_correlations directory if they exist
echo "Moving correlation files to $TABLES_DIR/score_correlations"
[[ -f "$TABLES_DIR/factor_scores_original_cis/score_correlations.csv" ]] && \
    mv "$TABLES_DIR/factor_scores_original_cis/score_correlations.csv" "$TABLES_DIR/score_correlations/" || \
    echo "Warning: score_correlations.csv not found"

[[ -f "$TABLES_DIR/factor_scores_original_cis/score_correlations_heatmap.png" ]] && \
    mv "$TABLES_DIR/factor_scores_original_cis/score_correlations_heatmap.png" "$TABLES_DIR/score_correlations/" || \
    echo "Warning: score_correlations_heatmap.png not found"

# Rename correlation files with suffix if provided
if [[ -n "$SUFFIX" ]]; then
    [[ -f "$TABLES_DIR/score_correlations/score_correlations.csv" ]] && \
        mv "$TABLES_DIR/score_correlations/score_correlations.csv" "$TABLES_DIR/score_correlations/score_correlations${FILE_SUFFIX}.csv"
    
    [[ -f "$TABLES_DIR/score_correlations/score_correlations_heatmap.png" ]] && \
        mv "$TABLES_DIR/score_correlations/score_correlations_heatmap.png" "$TABLES_DIR/score_correlations/score_correlations_heatmap${FILE_SUFFIX}.png"
fi

# Step 4: Run factor analysis on raw judgments
echo "----- Step 4: Running factor analysis on raw judgments -----"
eval "python factor_analysis.py \"$MY_PATH\" --analyze-questions" || {
    echo "Warning: factor_analysis.py on raw judgments failed. This may indicate insufficient data."
}

# Move factor analysis files to the factor_analysis directory if they exist
echo "Moving factor analysis files to $TABLES_DIR/factor_analysis"
# First check if we have any files to move
if [[ -n "$(find "$MY_PATH/tables/factor_analysis" -name "factor_*" 2>/dev/null)" ]]; then
    # Files are already in tables/factor_analysis, move them up one level
    find "$MY_PATH/tables/factor_analysis" -type f -exec mv {} "$TABLES_DIR/factor_analysis/" \; 2>/dev/null || true
elif [[ -n "$(find "$MY_PATH" -name "factor_*" 2>/dev/null)" ]]; then
    # Files are in the base processed dir
    find "$MY_PATH" -name "factor_*" -exec mv {} "$TABLES_DIR/factor_analysis/" \; 2>/dev/null || true
    find "$MY_PATH" -name "polynomial_*" -exec mv {} "$TABLES_DIR/factor_analysis/" \; 2>/dev/null || true
    [[ -f "$MY_PATH/r2_comparison.txt" ]] && mv "$MY_PATH/r2_comparison.txt" "$TABLES_DIR/factor_analysis/" || true
fi

# Rename factor analysis files with suffix if provided
if [[ -n "$SUFFIX" ]]; then
    for file in "$TABLES_DIR/factor_analysis"/factor_*.csv "$TABLES_DIR/factor_analysis"/polynomial_*.csv; do
        if [[ -f "$file" ]]; then
            base=$(basename "$file" .csv)
            dir=$(dirname "$file")
            mv "$file" "$dir/${base}${FILE_SUFFIX}.csv"
        fi
    done
    
    [[ -f "$TABLES_DIR/factor_analysis/r2_comparison.txt" ]] && \
        mv "$TABLES_DIR/factor_analysis/r2_comparison.txt" "$TABLES_DIR/factor_analysis/r2_comparison${FILE_SUFFIX}.txt"
fi

# Step 5: Run factor reliability analysis
echo "----- Step 5: Running factor reliability analysis -----"
eval "python factor_reliability_improved.py \"$MY_PATH\" --output-dir \"$MY_FACTOR_PATH/factor_reliability\" --debug --skip-bootstrap" || {
    echo "Warning: factor_reliability_improved.py failed. This may indicate insufficient data."
    echo "Creating empty reliability metrics file..."
    touch "$MY_FACTOR_PATH/factor_reliability/factor_reliability_metrics.csv"
}

# Rename reliability files with suffix if provided
if [[ -n "$SUFFIX" ]]; then
    for file in "$MY_FACTOR_PATH/factor_reliability"/*.csv; do
        if [[ -f "$file" ]]; then
            base=$(basename "$file" .csv)
            dir=$(dirname "$file")
            mv "$file" "$dir/${base}${FILE_SUFFIX}.csv"
        fi
    done
fi

# Check if the necessary files for updated CIs exist
communalities_file="$MY_FACTOR_PATH/factor_analysis/factor_communalities${FILE_SUFFIX}.csv"
reliability_file="$MY_FACTOR_PATH/factor_reliability/factor_reliability_metrics${FILE_SUFFIX}.csv"

# Fallback to files without suffix if suffix versions don't exist
[[ ! -f "$communalities_file" ]] && communalities_file="$MY_FACTOR_PATH/factor_analysis/factor_communalities.csv"
[[ ! -f "$reliability_file" ]] && reliability_file="$MY_FACTOR_PATH/factor_reliability/factor_reliability_metrics.csv"

communalities_arg=""
reliability_arg=""

# Add quotes around the file paths to handle spaces in the path
[[ -f "$communalities_file" ]] && communalities_arg="--communalities-file \"$communalities_file\"" || \
    echo "Warning: factor_communalities.csv not found, not using it for updated CIs"

[[ -f "$reliability_file" ]] && reliability_arg="--reliability-file \"$reliability_file\"" || \
    echo "Warning: factor_reliability_metrics.csv not found, not using it for updated CIs"

# Step 6: Re-run show_result.py with the reliability metrics
echo "----- Step 6: Running show_result.py with updated confidence intervals -----"
if [[ -n "$communalities_arg" ]] || [[ -n "$reliability_arg" ]]; then
    # Use eval to handle the quoted arguments correctly
    eval "python show_result.py --output --judge-name \"$MY_JUDGE\" --judgment-dir \"$MY_PATH\" --target-metric \"score\" --bootstrap-method bayesian $communalities_arg $reliability_arg" || {
        echo "Warning: show_result.py for 'score' with updated CIs failed. Continuing anyway..."
    }

    for metric in "safety_score" "style_score" "conciseness_score" "completeness_score" "correctness_score"; do
        echo "Running show_result.py for metric: $metric with reliability metrics"
        eval "python show_result.py --output --judge-name \"$MY_JUDGE\" --judgment-dir \"$MY_PATH\" --target-metric \"$metric\" --bootstrap-method bayesian $reliability_arg" || {
            echo "Warning: show_result.py for '$metric' with updated CIs failed. Continuing anyway..."
        }
    done
else
    echo "Skipping updated CIs because neither communalities nor reliability files were found"
fi

# Check if any updated result files were generated
if [[ -z "$(find leaderboard -name "arena_hard_leaderboard_*_${MY_JUDGE}_judge_*.csv" 2>/dev/null)" ]]; then
    echo "Warning: No updated leaderboard files were generated."
else
    # Move the updated result files to the appropriate directory
    echo "Moving updated CI result files to $TABLES_DIR/factor_scores_updated_cis"
    find leaderboard -name "arena_hard_leaderboard_*_${MY_JUDGE}_judge_*.csv" -exec mv {} "$TABLES_DIR/factor_scores_updated_cis/" \;
    
    # Rename files with suffix if provided
    if [[ -n "$SUFFIX" ]]; then
        for file in "$TABLES_DIR/factor_scores_updated_cis"/*.csv; do
            if [[ -f "$file" ]]; then
                base=$(basename "$file" .csv)
                dir=$(dirname "$file")
                mv "$file" "$dir/${base}${FILE_SUFFIX}.csv"
            fi
        done
    fi
fi

# Move question reliability files to the right directory if they exist
echo "Moving question reliability files to $TABLES_DIR/question_reliability"
# First check if files are in the tables/question_reliability directory
if [[ -n "$(find "$MY_PATH/tables/question_reliability" -name "question_reliability*" 2>/dev/null)" ]]; then
    # Files are already in tables/question_reliability, move them up one level
    find "$MY_PATH/tables/question_reliability" -type f -exec mv {} "$TABLES_DIR/question_reliability/" \; 2>/dev/null || true
elif [[ -n "$(find "$MY_PATH" -name "question_reliability*" 2>/dev/null)" ]]; then
    # Files are in the base processed dir
    find "$MY_PATH" -name "question_reliability*" -exec mv {} "$TABLES_DIR/question_reliability/" \; 2>/dev/null || true
fi

# Rename question reliability files with suffix if provided
if [[ -n "$SUFFIX" ]]; then
    for file in "$TABLES_DIR/question_reliability"/question_reliability*.csv; do
        if [[ -f "$file" ]]; then
            base=$(basename "$file" .csv)
            dir=$(dirname "$file")
            mv "$file" "$dir/${base}${FILE_SUFFIX}.csv"
        fi
    done
fi

# Note: We don't need to move count files because gen_subscores.py now places them directly in the correct location
echo "Count files are already in $TABLES_DIR/counts"

# Step 7: Run factor analysis on post-processed rankings
echo "----- Step 7: Running factor analysis on post-processed rankings -----"

# First check for original CIs
if [[ -n "$(find "$TABLES_DIR/factor_scores_original_cis" -name "*.csv" 2>/dev/null)" ]]; then
    # Create directory for original CIs analysis
    mkdir -p "$TABLES_DIR/rankings_factor_analysis/original_cis"
    
    echo "Running factor analysis on original CI rankings..."
    eval "python factor_analysis.py \"$TABLES_DIR/factor_scores_original_cis\" --output-dir \"$TABLES_DIR/rankings_factor_analysis/original_cis\"" || {
        echo "Warning: Factor analysis on original CI rankings failed."
    }
else
    echo "No original CI rankings found to analyze."
fi

# Then check for updated CIs
if [[ -n "$(find "$TABLES_DIR/factor_scores_updated_cis" -name "*.csv" 2>/dev/null)" ]]; then
    # Create directory for updated CIs analysis
    mkdir -p "$TABLES_DIR/rankings_factor_analysis/updated_cis"
    
    echo "Running factor analysis on updated CI rankings..."
    eval "python factor_analysis.py \"$TABLES_DIR/factor_scores_updated_cis\" --output-dir \"$TABLES_DIR/rankings_factor_analysis/updated_cis\"" || {
        echo "Warning: Factor analysis on updated CI rankings failed."
    }
else
    echo "No updated CI rankings found to analyze."
fi

echo "----- All analysis completed -----"
echo "Results are available in $TABLES_DIR/"

# Provide a summary of what was created
echo ""
echo "SUMMARY OF RESULTS:"
echo "==================="

# Check original CIs
orig_ci_count=$(find "$TABLES_DIR/factor_scores_original_cis" -name "*.csv" | wc -l)
echo "Factor scores (original CIs): $orig_ci_count files"

# Check updated CIs
updated_ci_count=$(find "$TABLES_DIR/factor_scores_updated_cis" -name "*.csv" | wc -l)
echo "Factor scores (updated CIs): $updated_ci_count files"

# Check correlations
correlation_files=$(find "$TABLES_DIR/score_correlations" -name "*.csv" | wc -l)
if [[ $correlation_files -gt 0 ]]; then
    echo "Score correlations: Created"
else
    echo "Score correlations: Not created"
fi

# Check factor analysis results
if [[ -f "$TABLES_DIR/factor_analysis/factor_loadings.csv" ]] || [[ -f "$TABLES_DIR/factor_analysis/factor_loadings${FILE_SUFFIX}.csv" ]]; then
    echo "Factor analysis: Completed"
else
    echo "Factor analysis: Not completed"
fi

# Check factor reliability results
if [[ -f "$TABLES_DIR/factor_reliability/factor_reliability_metrics.csv" ]] || [[ -f "$TABLES_DIR/factor_reliability/factor_reliability_metrics${FILE_SUFFIX}.csv" ]]; then
    echo "Factor reliability: Completed"
else
    echo "Factor reliability: Not completed"
fi

# Check rankings factor analysis results
if [[ -d "$TABLES_DIR/rankings_factor_analysis/original_cis" ]]; then
    if [[ -f "$TABLES_DIR/rankings_factor_analysis/original_cis/tables/factor_analysis/r2_comparison.txt" ]]; then
        echo "Factor analysis on original CI rankings: Completed"
        echo "   R² values can be found in: $TABLES_DIR/rankings_factor_analysis/original_cis/tables/factor_analysis/r2_comparison.txt"
    else
        echo "Factor analysis on original CI rankings: Partially completed (no R² values generated)"
    fi
else
    echo "Factor analysis on original CI rankings: Not completed"
fi

if [[ -d "$TABLES_DIR/rankings_factor_analysis/updated_cis" ]]; then
    if [[ -f "$TABLES_DIR/rankings_factor_analysis/updated_cis/tables/factor_analysis/r2_comparison.txt" ]]; then
        echo "Factor analysis on updated CI rankings: Completed"
        echo "   R² values can be found in: $TABLES_DIR/rankings_factor_analysis/updated_cis/tables/factor_analysis/r2_comparison.txt"
    else
        echo "Factor analysis on updated CI rankings: Partially completed (no R² values generated)"
    fi
else
    echo "Factor analysis on updated CI rankings: Not completed"
fi

# Check question reliability results
question_reliability_count=$(find "$TABLES_DIR/question_reliability" -name "*.csv" | wc -l)
echo "Question reliability files: $question_reliability_count"

echo ""
if [[ -n "$SUFFIX" ]]; then
    echo "All files have been suffixed with: $SUFFIX"
fi
echo "If any steps failed, you can run them manually or fix the issues and re-run the script."