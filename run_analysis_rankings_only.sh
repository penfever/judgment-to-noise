#!/bin/bash
# Fast Rankings-Only Analysis Pipeline
# This script generates rankings from debiased judge data without expensive computations

# Enable error handling
set -e

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --input-dir PATH        Path to directory containing debiased JSONL files (required)"
    echo "  --output-dir PATH       Path to output directory (required)"
    echo "  --judge-name NAME       Judge model name (required)"
    echo "  --suffix STRING         Optional suffix for output files"
    echo "  --help                  Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --input-dir /path/to/base_processed_debiased --output-dir /path/to/results --judge-name DeepSeek-R1-32B"
}

# Initialize variables
INPUT_DIR=""
OUTPUT_DIR=""
JUDGE_NAME=""
SUFFIX=""

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

# Create output directory structure (only what we need)
TABLES_DIR="$OUTPUT_DIR/tables"
mkdir -p "$TABLES_DIR"
mkdir -p "$TABLES_DIR/factor_scores_original_cis"
mkdir -p "$TABLES_DIR/factor_scores_updated_cis"

echo "=== Fast Rankings-Only Analysis Pipeline ==="
echo "Input directory: $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Judge name: $JUDGE_NAME"
if [[ -n "$SUFFIX" ]]; then
    echo "Suffix: $SUFFIX"
fi
echo ""

# Set environment variables for show_result.py
export MY_PATH="$INPUT_DIR"
export MY_JUDGE="$JUDGE_NAME"
export MY_FACTOR_PATH="$TABLES_DIR"

echo "Environment variables set:"
echo "MY_PATH=$MY_PATH"
echo "MY_JUDGE=$MY_JUDGE"
echo "MY_FACTOR_PATH=$MY_FACTOR_PATH"

# Create leaderboard directory
mkdir -p leaderboard

# Determine output suffix for files
FILE_SUFFIX=""
if [[ -n "$SUFFIX" ]]; then
    FILE_SUFFIX="_$SUFFIX"
fi

echo "----- Step 1: Running show_result.py for each metric -----"

# Check if required Python packages are available
python -c "import plotly" 2>/dev/null || {
    echo "Installing plotly..."
    pip install plotly >/dev/null 2>&1 || echo "Warning: Could not install plotly, continuing anyway..."
}

# Run show_result.py for each metric (only generating rankings, no expensive computations)
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
    echo "Checking what files exist in leaderboard directory:"
    ls -la leaderboard/ || echo "Leaderboard directory is empty or doesn't exist"
    
    echo "Checking input data format..."
    echo "First JSONL file sample:"
    head -1 "$INPUT_DIR"/*.jsonl | head -5
else
    # Move the result files to the appropriate directory
    echo "Moving result files to $TABLES_DIR/factor_scores_updated_cis"
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
    
    echo "✅ Rankings generated successfully!"
fi

echo "----- Analysis completed -----"
echo "Results are available in $TABLES_DIR/"

# Provide a summary of what was created
echo ""
echo "SUMMARY OF RESULTS:"
echo "==================="

# Check updated CIs (rankings)
updated_ci_count=$(find "$TABLES_DIR/factor_scores_updated_cis" -name "*.csv" | wc -l)
echo "Factor score rankings: $updated_ci_count files"

if [[ $updated_ci_count -gt 0 ]]; then
    echo "Generated ranking files:"
    ls -1 "$TABLES_DIR/factor_scores_updated_cis"/*.csv
else
    echo "❌ No ranking files were generated"
    echo ""
    echo "Debugging information:"
    echo "Input directory contents:"
    ls -la "$INPUT_DIR"
    echo ""
    echo "Sample of first JSONL file:"
    if [[ -n "$(find "$INPUT_DIR" -name "*.jsonl" | head -1)" ]]; then
        head -3 "$(find "$INPUT_DIR" -name "*.jsonl" | head -1)"
    fi
fi

if [[ -n "$SUFFIX" ]]; then
    echo "All files have been suffixed with: $SUFFIX"
fi