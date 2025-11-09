#!/bin/bash

# --- Fair Benchmarking Harness ---
#
# USAGE:
#   sudo ./run.sh <pipeline_name> [framework_name]
#
# EXAMPLES:
#   sudo ./run.sh nyc_taxi          (Runs all frameworks in pipelines/nyc_taxi/)
#   sudo ./run.sh nyc_taxi duckdb   (Runs only pipelines/nyc_taxi/duckdb.py)
#

# 1. Argument Validation
if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    echo "Usage: $0 <pipeline_name> [framework_name]"
    echo "Example: $0 nyc_taxi"
    echo "Example: $0 nyc_taxi duckdb"
    exit 1
fi

if [ "$EUID" -ne 0 ]; then
    echo "!!! Please run this script with sudo !!!"
    echo "Sudo is required to clear the OS page cache for a fair 'cold run'."
    exit 1
fi

PIPELINE_NAME="$1"
FRAMEWORK_NAME="$2" # This might be empty

# 2. Setup Directories
PIPELINE_DIR="pipelines/$PIPELINE_NAME"
if [ ! -d "$PIPELINE_DIR" ]; then
    echo "Error: Pipeline directory not found: $PIPELINE_DIR"
    exit 1
fi

TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
RESULTS_DIR="results/$TIMESTAMP"
RESULTS_PIPELINE_DIR="$RESULTS_DIR/$PIPELINE_NAME"

mkdir -p "$RESULTS_PIPELINE_DIR"
echo "Results will be saved to: $RESULTS_DIR"

# 3. Find and Run Scripts
TARGET_SCRIPTS=()
if [ -z "$FRAMEWORK_NAME" ]; then
    echo "Running all frameworks for pipeline: $PIPELINE_NAME"
    # Find all .py files except __init__.py
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
    mkdir -p "$FRAMEWORK_RESULTS_DIR"

    echo ""
    echo "--------------------------------------------------------"
    echo "--- Running: $script ---"
    echo "--- Results Dir: $FRAMEWORK_RESULTS_DIR ---"
    echo "--------------------------------------------------------"

    echo "Step 1: Clearing OS Page Cache..."
    sync
    echo 3 > /proc/sys/vm/drop_caches
    echo "  Caches cleared."

    echo "Step 2: Running benchmark as user '$SUDO_USER'..."
    # Export the results directory so the Python script knows where to save
    export RESULTS_DIR="$FRAMEWORK_RESULTS_DIR"
    
    # Run the command as the original user, not as root
    if [ -n "$SUDO_USER" ]; then
        su "$SUDO_USER" -c "python3 $script"
    else
        python3 "$script"
    fi
    
    if [ $? -ne 0 ]; then
        echo "!!! Benchmark script failed: $script"
    else
        echo "--- Finished: $script ---"
    fi
    
    # Unset for the next loop
    unset RESULTS_DIR
done

# 5. Generate Comparison Visualizations
echo ""
echo "--------------------------------------------------------"
echo "--- Generating Comparison Visualizations ---"
echo "--------------------------------------------------------"
if [ -n "$SUDO_USER" ]; then
    su "$SUDO_USER" -c "python3 common/visualize.py $RESULTS_DIR"
else
    python3 common/visualize.py "$RESULTS_DIR"
fi

echo "--- All complete. Results are in $RESULTS_DIR ---"