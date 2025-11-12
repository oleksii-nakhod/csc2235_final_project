#!/bin/bash

# --- Fair Benchmarking Harness ---
#
# USAGE:
#   ./run.sh <pipeline_name> [framework_name]
#
# NOTE: Run this script *without* sudo. It will call sudo itself
#       only for the one command that needs it (clearing cache).

# 1. Argument Validation
if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    echo "Usage: $0 <pipeline_name> [framework_name]"
    echo "Example: $0 nyc_taxi"
    exit 1
fi

PIPELINE_NAME="$1"
FRAMEWORK_NAME="$2"

# 2. Setup Directories
PIPELINE_DIR="pipelines/$PIPELINE_NAME"
if [ ! -d "$PIPELINE_DIR" ]; then
    echo "Error: Pipeline directory not found: $PIPELINE_DIR"
    exit 1
fi

TIMESTAMP=$(date -u +"%Y-%m-%d_%H-%M-%S")
RESULTS_DIR="results/$TIMESTAMP"
RESULTS_PIPELINE_DIR="$RESULTS_DIR/$PIPELINE_NAME"

mkdir -p "$RESULTS_PIPELINE_DIR"
echo "Results will be saved to: $RESULTS_DIR"

VENV_PYTHON="/local/repository/venv/bin/python3"

# 3. Find and Run Scripts
TARGET_SCRIPTS=()
if [ -z "$FRAMEWORK_NAME" ]; then
    echo "Running all frameworks for pipeline: $PIPELINE_NAME"
    # --- 1. MODIFIED: Look for *_pipeline.py ---
    for script in $(find "$PIPELINE_DIR" -maxdepth 1 -name "*_pipeline.py" ! -name "__init__.py"); do
        TARGET_SCRIPTS+=("$script")
    done
else
    echo "Running framework '$FRAMEWORK_NAME' for pipeline: $PIPELINE_NAME"
    # --- 2. MODIFIED: Look for <framework>_pipeline.py ---
    SCRIPT_PATH="$PIPELINE_DIR/${FRAMEWORK_NAME}_pipeline.py"
    if [ ! -f "$SCRIPT_PATH" ]; then
        echo "Error: Framework script not found: $SCRIPT_PATH"
        exit 1
    fi
    TARGET_SCRIPTS=("$SCRIPT_PATH")
fi

echo "Found ${#TARGET_SCRIPTS[@]} script(s) to run."

# 4. Execute Benchmark(s)
for script in "${TARGET_SCRIPTS[@]}"; do
    # --- 3. MODIFIED: Extract framework name correctly ---
    FRAMEWORK=$(basename "$script" _pipeline.py)
    FRAMEWORK_RESULTS_DIR="$RESULTS_PIPELINE_DIR/$FRAMEWORK"
    
    mkdir -p "$FRAMEWORK_RESULTS_DIR"

    echo ""
    echo "--------------------------------------------------------"
    echo "--- Running: $script ---"
    echo "--- Results Dir: $FRAMEWORK_RESULTS_DIR ---"
    echo "--------------------------------------------------------"

    echo "Step 1: Clearing OS Page Cache (requires sudo)..."
    sudo sync
    sudo sh -c "echo 3 > /proc/sys/vm/drop_caches"
    if [ $? -ne 0 ]; then
        echo "!!! Failed to clear OS cache. Ensure user $(whoami) has passwordless sudo."
        exit 1
    fi
    echo "  Caches cleared."

    echo "Step 2: Running benchmark as user '$(whoami)'..."
    
    export SCRIPT_RESULTS_DIR="$FRAMEWORK_RESULTS_DIR"
    
    "$VENV_PYTHON" "$script"
    
    if [ $? -ne 0 ]; then
        echo "!!! Benchmark script failed: $script"
    else
        echo "--- Finished: $script ---"
    fi
    
    unset SCRIPT_RESULTS_DIR
done

# 5. Generate Comparison Visualizations
echo ""
echo "--------------------------------------------------------"
echo "--- Generating Comparison Visualizations ---"
echo "--------------------------------------------------------"

"$VENV_PYTHON" common/visualize.py "$RESULTS_DIR"

echo "--- All complete. Results are in $RESULTS_DIR ---"