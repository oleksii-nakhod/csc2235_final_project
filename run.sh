#!/bin/bash

# --- Fair Benchmarking Harness ---
#
# NEW USAGE:
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

# --- REMOVED SUDO CHECK ---
# We now run this script as the normal user.

PIPELINE_NAME="$1"
FRAMEWORK_NAME="$2" # This might be empty

# 2. Setup Directories (now run as you, not root)
PIPELINE_DIR="pipelines/$PIPELINE_NAME"
if [ ! -d "$PIPELINE_DIR" ]; then
    echo "Error: Pipeline directory not found: $PIPELINE_DIR"
    exit 1
fi

TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
RESULTS_DIR="results/$TIMESTAMP"
RESULTS_PIPELINE_DIR="$RESULTS_DIR/$PIPELINE_NAME"

# This is now created by the user, so permissions are correct
mkdir -p "$RESULTS_PIPELINE_DIR"
echo "Results will be saved to: $RESULTS_DIR"

VENV_PYTHON="/local/repository/venv/bin/python3"

# 3. Find and Run Scripts
TARGET_SCRIPTS=()
if [ -z "$FRAMEWORK_NAME" ]; then
    echo "Running all frameworks for pipeline: $PIPELINE_NAME"
    for script in $(find "$PIPELINE_DIR" -maxdepth 1 -name "*.py" ! -name "__init__.py"); do
        TARGET_SCRIPTS+=("$script")
    done
else
    echo "Running framework '$FRAMEWORK_NAME' for pipeline: $PIPELINE_NAME"
    SCRIPT_PATH="$PIPELINE_DIR/$FRAMEWORK_NAME.py"
    if [ ! -f "$SCRIPT_PATH" ]; then
        echo "Error: Framework script not found: $SCRIPT_PATH"
        exit 1
    fi
    TARGET_SCRIPTS=("$SCRIPT_PATH")
fi

echo "Found ${#TARGET_SCRIPTS[@]} script(s) to run."

# 4. Execute Benchmark(s)
for script in "${TARGET_SCRIPTS[@]}"; do
    FRAMEWORK=$(basename "$script" .py)
    FRAMEWORK_RESULTS_DIR="$RESULTS_PIPELINE_DIR/$FRAMEWORK"
    
    # This is also created by the user, so permissions are correct
    mkdir -p "$FRAMEWORK_RESULTS_DIR"

    echo ""
    echo "--------------------------------------------------------"
    echo "--- Running: $script ---"
    echo "--- Results Dir: $FRAMEWORK_RESULTS_DIR ---"
    echo "--------------------------------------------------------"

    # --- THIS BLOCK IS THE NEW LOGIC ---
    echo "Step 1: Clearing OS Page Cache (requires sudo)..."
    # Call sudo *only* for the commands that need it.
    # We use sudo sh -c "..." for the redirection to work.
    sudo sync
    sudo sh -c "echo 3 > /proc/sys/vm/drop_caches"
    if [ $? -ne 0 ]; then
        echo "!!! Failed to clear OS cache. Ensure user $(whoami) has passwordless sudo."
        exit 1
    fi
    echo "  Caches cleared."
    # --- END NEW LOGIC BLOCK ---

    echo "Step 2: Running benchmark as user '$(whoami)'..."
    export RESULTS_DIR="$FRAMEWORK_RESULTS_DIR"
    
    # --- THIS LINE IS FIXED ---
    # No 'su' needed. Just run the command directly.
    "$VENV_PYTHON" "$script"
    
    if [ $? -ne 0 ]; then
        echo "!!! Benchmark script failed: $script"
    else
        echo "--- Finished: $script ---"
    fi
    
    unset RESULTS_DIR
done

# 5. Generate Comparison Visualizations
echo ""
echo "--------------------------------------------------------"
echo "--- Generating Comparison Visualizations ---"
echo "--------------------------------------------------------"

# --- THIS LINE IS ALSO FIXED ---
# No 'su' needed. The $RESULTS_DIR argument will now pass correctly.
"$VENV_PYTHON" common/visualize.py "$RESULTS_DIR"

echo "--- All complete. Results are in $RESULTS_DIR ---"