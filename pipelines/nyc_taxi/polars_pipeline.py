import pandas as pd
import time
import os
import sys
import warnings

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_ROOT)

from common.download_utils import get_year_month_list, download_taxi_data

def run_polars_pipeline(config, global_results_df):
    """
    A STUB for the Polars pipeline.
    It should mimic the DuckDB steps: load, impute, feature_eng, encode, scale.
    """
    print(f"\n[BEGIN] Running STUB pipeline: {config['name']}...")
    
    # --- Simulate work ---
    time.sleep(2) 
    print("  Polars 'loaded' data.")
    time.sleep(3)
    print("  Polars 'imputed' data.")
    
    # --- Create dummy results ---
    # This is just to make the visualizer work
    steps = ['load_data', 'handle_missing_values', 'feature_engineering']
    run_results_list = []
    
    for i, step in enumerate(steps):
        run_results_list.append({
            'config_name': config.get('name', 'N/A'), 'step': step,
            'start_time': i * 2, 'end_time': (i + 1) * 2,
            'execution_time_s': 2.0, 'cpu_time_s': 1.8, 'peak_memory_mib': 1024.0,
            'data_size_pct': config.get('data_size_pct', 'N/A'),
            'system_size_pct': config.get('system_size_pct', 'N/A')
        })
        
    run_results_df = pd.DataFrame(run_results_list)
    return pd.concat([global_results_df, run_results_df], ignore_index=True)


if __name__ == "__main__":
    print("--- Polars NYC Taxi STUB Benchmark ---")
    
    RESULTS_DIR = os.environ.get('SCRIPT_RESULTS_DIR')
    if not RESULTS_DIR:
        RESULTS_DIR = "results/local_test/nyc_taxi/polars"
        os.makedirs(RESULTS_DIR, exist_ok=True)

    START_YEAR, START_MONTH = 2009, 1
    END_YEAR, END_MONTH = 2009, 3 # Polars test only on 3 months
    LOCAL_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "nyc_taxi")
    
    year_month_list = get_year_month_list(START_YEAR, START_MONTH, END_YEAR, END_MONTH)
    all_files_sorted = download_taxi_data(year_month_list, LOCAL_DATA_DIR)
    
    # --- Just run one dummy config ---
    CONFIGURATIONS = [{
        "name": "Data_10%_Sys_10%", "data_files": all_files_sorted,
        "data_size_pct": "10%", "system_size_pct": "10%"
    }, {
        "name": "Data_100%_Sys_100%", "data_files": all_files_sorted,
        "data_size_pct": "100%", "system_size_pct": "100%"
    }]
    
    all_results_df = pd.DataFrame()
    for config in CONFIGURATIONS:
        all_results_df = run_polars_pipeline(config, all_results_df)

    output_csv = os.path.join(RESULTS_DIR, 'polars_results.csv')
    all_results_df.to_csv(output_csv, index=False)
    
    print(f"\n--- Polars STUB benchmarks complete ---")
    print(f"Full results saved to {output_csv}")