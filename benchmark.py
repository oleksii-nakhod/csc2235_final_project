import duckdb
import psutil
import pandas as pd
import time
import os
import sys
import glob
import urllib.request
import plotly.express as px
import plotly.io as pio
from memory_profiler import memory_usage
from datetime import datetime
import math
import warnings

# --- New: Helper to get date range ---
def get_year_month_list(start_year, start_month, end_year, end_month):
    """
    Generates a list of (year, month) tuples within the specified range.
    """
    ym_list = []
    year = start_year
    month = start_month
    
    while True:
        ym_list.append((year, month))
        
        if year == end_year and month == end_month:
            break
            
        month += 1
        if month > 12:
            month = 1
            year += 1
            
    return ym_list

# --- New: Helper function to download data ---
def download_taxi_data(year_month_list, local_dir="data"):
    """
    Downloads NYC Taxi data for the given list of (year, month) tuples.
    Returns a sorted list of all downloaded file paths.
    """
    os.makedirs(local_dir, exist_ok=True)
    BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_"
    
    print(f"--- Checking for local data in '{local_dir}' ---")
    
    downloaded_files = []
    
    for year, month in year_month_list:
        month_str = f"{month:02d}"
        file_name = f"yellow_tripdata_{year}-{month_str}.parquet"
        local_path = os.path.join(local_dir, file_name)
        downloaded_files.append(local_path)
        
        if os.path.exists(local_path):
            continue
            
        print(f"  Downloading: {file_name} to {local_dir}/")
        url = f"{BASE_URL}{year}-{month_str}.parquet"
        try:
            urllib.request.urlretrieve(url, local_path)
            print(f"  Success: Downloaded {file_name}")
        except Exception as e:
            print(f"!!! Failed to download {url}: {e}")
            if os.path.exists(local_path):
                os.remove(local_path)
            downloaded_files.pop()
            
    print(f"--- Data download check complete. Found {len(downloaded_files)} files. ---")
    return sorted(downloaded_files)

# --- Database and Pipeline Steps ---
def connect_to_db(config):
    """
    Connect to DuckDB and apply thread/memory settings.
    """
    db_file = config.get('db_file', ':memory:')
    con = duckdb.connect(database=db_file, read_only=False)
    
    if config.get('memory_limit'):
        con.execute(f"SET memory_limit='{config['memory_limit']}'")
    if config.get('num_threads'):
        con.execute(f"SET threads={config['num_threads']}")
        
    print(f"  Connected to DB. Threads: {config.get('num_threads')}, Memory Limit: {config.get('memory_limit')}")
    return con

def load_data(con, config):
    data_files = config['data_files']
    
    if not data_files:
        print("  No data files to load. Skipping.")
        raise ValueError("No data files specified in config['data_files']")

    if config['file_type'] == 'parquet':
        con.execute(f"CREATE TABLE taxi_data AS SELECT * FROM read_parquet({data_files}, union_by_name=True)")
    else:
        print(f"Unsupported file type: {config['file_type']}")
        raise ValueError("Only parquet is configured")
    
    count = con.execute("SELECT COUNT(*) FROM taxi_data").fetchone()[0]
    print(f"  Loaded {count} rows from {len(data_files)} files.")
    return con

def handle_missing_values(con, config):
    mean_fare_result = con.execute("SELECT AVG(fare_amount) FROM taxi_data WHERE fare_amount > 0").fetchone()
    mean_fare = mean_fare_result[0] if mean_fare_result and mean_fare_result[0] is not None else 0.0
    
    mean_distance_result = con.execute("SELECT AVG(trip_distance) FROM taxi_data WHERE trip_distance > 0").fetchone()
    mean_distance = mean_distance_result[0] if mean_distance_result and mean_distance_result[0] is not None else 0.0
    
    con.execute(f"""
    CREATE TABLE taxi_clean AS
    SELECT
        *,
        CASE 
            WHEN fare_amount IS NULL OR fare_amount <= 0 THEN {mean_fare}
            ELSE fare_amount 
        END AS fare_amount_imputed,
        CASE 
            WHEN trip_distance IS NULL OR trip_distance <= 0 THEN {mean_distance}
            ELSE trip_distance 
        END AS trip_distance_imputed,
        COALESCE(payment_type, 0) AS payment_type_imputed
    FROM taxi_data
    """)
    print("  Imputed missing values for fare, distance, and payment_type.")
    return con

def feature_engineering(con, config):
    con.execute("""
    ALTER TABLE taxi_clean ADD COLUMN trip_duration_mins DOUBLE;
    ALTER TABLE taxi_clean ADD COLUMN speed_mph DOUBLE;
    ALTER TABLE taxi_clean ADD COLUMN is_weekend INTEGER;
    """)
    
    cols = [c[0] for c in con.execute("DESCRIBE taxi_clean").fetchall()]
    pickup_col = 'tpep_pickup_datetime' if 'tpep_pickup_datetime' in cols else 'pickup_datetime'
    dropoff_col = 'tpep_dropoff_datetime' if 'tpep_dropoff_datetime' in cols else 'dropoff_datetime'

    if pickup_col not in cols or dropoff_col not in cols:
        raise ValueError(f"Could not find valid pickup/dropoff timestamp columns. Found: {cols}")

    print(f"  Using columns: {pickup_col}, {dropoff_col} for feature engineering.")

    con.execute(f"""
    UPDATE taxi_clean SET 
        trip_duration_mins = CAST(epoch({dropoff_col} - {pickup_col}) AS DOUBLE) / 60.0;
    """)
    con.execute("""
    UPDATE taxi_clean SET
        speed_mph = CASE 
            WHEN trip_duration_mins > 0 THEN trip_distance_imputed / (trip_duration_mins / 60.0)
            ELSE 0
        END;
    """)
    con.execute(f"""
    UPDATE taxi_clean SET
        is_weekend = CASE 
            WHEN dayofweek({pickup_col}) IN (0, 6) THEN 1
            ELSE 0
        END;
    """)
    con.execute("UPDATE taxi_clean SET speed_mph = 0 WHERE speed_mph IS NULL OR isinf(speed_mph) OR isnan(speed_mph)")
    con.execute("UPDATE taxi_clean SET trip_duration_mins = 0 WHERE trip_duration_mins < 0")
    print("  Engineered features: duration, speed, is_weekend.")
    return con

def categorical_encoding(con, config):
    payment_types = [r[0] for r in con.execute("SELECT DISTINCT payment_type_imputed FROM taxi_clean").fetchall()]
    print(f"  One-hot encoding for payment types: {payment_types}")
    select_clauses = []
    for pt in payment_types:
        pt_col_name = str(pt).replace('.', '_')
        clause = f"CASE WHEN payment_type_imputed = {pt} THEN 1 ELSE 0 END AS payment_type_{pt_col_name}"
        select_clauses.append(clause)
        
    if not select_clauses:
        print("  No payment types found to encode. Skipping.")
        con.execute("CREATE TABLE taxi_encoded AS SELECT * FROM taxi_clean")
        return con

    con.execute(f"""
    CREATE TABLE taxi_encoded AS
    SELECT
        *,
        {', '.join(select_clauses)}
    FROM taxi_clean
    """)
    print("  Performed one-hot encoding for payment_type.")
    return con

def numerical_scaling(con, config):
    con.execute("""
    CREATE OR REPLACE MACRO min_max_scaler(val, min_val, max_val) AS
        (val - min_val) / nullif(max_val - min_val, 0)
    """);
    
    params = con.execute("""
        SELECT 
            min(fare_amount_imputed),
            max(fare_amount_imputed),
            min(trip_distance_imputed),
            max(trip_distance_imputed),
            min(speed_mph),
            max(speed_mph)
        FROM taxi_encoded
    """).fetchone()
    
    min_fare, max_fare, min_dist, max_dist, min_speed, max_speed = (val if val is not None else 0 for val in params)
    
    print(f"  Scaling params: Fare({min_fare}, {max_fare}), Dist({min_dist}, {max_dist}), Speed({min_speed}, {max_speed})")
    
    con.execute(f"""
    CREATE TABLE taxi_final AS
    SELECT
        *,
        min_max_scaler(fare_amount_imputed, {min_fare}, {max_fare}) AS fare_scaled,
        min_max_scaler(trip_distance_imputed, {min_dist}, {max_dist}) AS dist_scaled,
        min_max_scaler(speed_mph, {min_speed}, {max_speed}) AS speed_scaled
    FROM taxi_encoded
    """)
    print("  Applied Min-Max scaling to fare, distance, and speed.")
    return con

# --- NEW: Plotting Function ---
def visualize_results(results_df, output_file='benchmark_timeline.html'):
    """
    Generates an interactive Plotly timeline from the benchmark results.
    """
    if results_df.empty:
        print("No results to visualize.")
        return

    print(f"\n--- Generating visualization ---")
    
    if not all(col in results_df.columns for col in ['start_time', 'end_time', 'config_name', 'step']):
        print("Result DataFrame is missing required columns for visualization.")
        print(f"Columns found: {results_df.columns}")
        return

    pio.templates.default = "plotly_dark"
    
    fig = px.timeline(
        results_df,
        x_start="start_time",
        x_end="end_time",
        y="config_name",
        color="step",
        title="DuckDB Pipeline Benchmark Results",
        hover_data=['execution_time_s', 'peak_memory_mib', 'cpu_time_s']
    )
    
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(
        xaxis_title="Time (seconds)",
        yaxis_title="Benchmark Configuration",
        legend_title="Pipeline Step"
    )
    
    fig.write_html(output_file)
    print(f"Interactive timeline saved to {output_file}")

# --- Main Execution ---
def run_full_pipeline(config, global_results_df):
    print(f"\n[BEGIN] Running pipeline: {config['name']}...")
    
    run_results_list = []
    pipeline_start_time = time.perf_counter()
    con = None
    
    pipeline_steps = [
        (connect_to_db, (config,), {}),
        (load_data, (None, config), {}),
        (handle_missing_values, (None, config), {}),
        (feature_engineering, (None, config), {}),
        (categorical_encoding, (None, config), {}),
        (numerical_scaling, (None, config), {})
    ]

    try:
        for i, (func, args, kwargs) in enumerate(pipeline_steps):
            step_name = func.__name__
            
            if i > 0:
                if con is None:
                    print(f"  Connection is missing. Stopping pipeline.")
                    break
                args = (con, config)
            
            print(f"  [START] Step: {step_name}")
            process = psutil.Process(os.getpid())
            cpu_times_before = process.cpu_times()
            step_start_time = time.perf_counter()

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                mem_usage = memory_usage((func, args, kwargs), max_usage=True, retval=True, interval=0.1)
            
            peak_mem_mib = mem_usage[0]
            result = mem_usage[1]
            
            step_end_time = time.perf_counter()
            cpu_times_after = process.cpu_times()
            
            con = result

            elapsed_time = step_end_time - step_start_time
            cpu_time_used = (cpu_times_after.user - cpu_times_before.user) + \
                            (cpu_times_after.system - cpu_times_before.system)
            
            abs_start_time = step_start_time - pipeline_start_time
            abs_end_time = step_end_time - pipeline_start_time

            print(f"  [END] Step: {step_name}")
            print(f"    Exec Time: {elapsed_time:.4f} s")
            print(f"    CPU Time: {cpu_time_used:.4f} s")
            print(f"    Peak Mem: {peak_mem_mib:.2f} MiB")

            run_results_list.append({
                'config_name': config.get('name', 'N/A'),
                'step': step_name,
                'start_time': abs_start_time,
                'end_time': abs_end_time,
                'execution_time_s': elapsed_time,
                'cpu_time_s': cpu_time_used,
                'peak_memory_mib': peak_mem_mib,
                'file_type': config.get('file_type', 'N/A'),
                'memory_limit': config.get('memory_limit', 'None'),
                'num_threads': config.get('num_threads', 'N/A'),
                'data_size_pct': config.get('data_size_pct', 'N/A'),
                'system_size_pct': config.get('system_size_pct', 'N/A')
            })

    except Exception as e:
        print(f"!!! ERROR in pipeline config '{config['name']}' at step '{step_name}': {e}", file=sys.stderr)
    
    finally:
        if con:
            try:
                con.close()
                print("  Connection closed.")
            except Exception as ce:
                print(f"  Error while closing connection: {ce}", file=sys.stderr)

    print(f"[END] Pipeline: {config['name']} finished.")
    
    run_results_df = pd.DataFrame(run_results_list)
    return pd.concat([global_results_df, run_results_df], ignore_index=True)


if __name__ == "__main__":
    START_YEAR, START_MONTH = 2009, 1
    END_YEAR, END_MONTH = 2025, 9
    LOCAL_DATA_DIR = "data"
    
    year_month_list = get_year_month_list(START_YEAR, START_MONTH, END_YEAR, END_MONTH)
    all_files_sorted = download_taxi_data(year_month_list, LOCAL_DATA_DIR)
    
    if not all_files_sorted:
        print("No data files found or downloaded. Exiting benchmark.")
        sys.exit(1)

    try:
        total_mem_bytes = psutil.virtual_memory().total
        total_mem_gb = total_mem_bytes / (1024**3)
        total_cores = psutil.cpu_count(logical=True)
        print(f"System detected: {total_cores} logical cores, {total_mem_gb:.2f} GB RAM")
    except Exception:
        print("Could not detect system info, using defaults (4 cores, 8GB RAM).")
        total_mem_gb = 8.0
        total_cores = 4
    
    safe_max_mem_gb = total_mem_gb * 0.75

    total_file_count = len(all_files_sorted)
    data_size_configs = {
        '10%': all_files_sorted[:max(1, int(total_file_count * 0.10))],
        '30%': all_files_sorted[:max(1, int(total_file_count * 0.30))],
        '100%': all_files_sorted,
    }

    system_size_configs = {
        '10%': {
            'threads': max(1, int(total_cores * 0.10)),
            'memory': f"{max(1, int(safe_max_mem_gb * 0.10))}GB"
        },
        '30%': {
            'threads': max(1, int(total_cores * 0.30)),
            'memory': f"{max(1, int(safe_max_mem_gb * 0.30))}GB"
        },
        '100%': {
            'threads': total_cores,
            'memory': f"{int(safe_max_mem_gb)}GB"
        }
    }
    
    CONFIGURATIONS = []
    for data_pct, file_list in data_size_configs.items():
        for sys_pct, sys_config in system_size_configs.items():
            config_name = f"Data_{data_pct}_Sys_{sys_pct}"
            CONFIGURATIONS.append({
                "name": config_name,
                "file_type": "parquet",
                "data_files": file_list,
                "memory_limit": sys_config['memory'],
                "num_threads": sys_config['threads'],
                "data_size_pct": data_pct,
                "system_size_pct": sys_pct,
            })

    print(f"\n--- Generated {len(CONFIGURATIONS)} test configurations ---")
    for cfg in CONFIGURATIONS:
        print(f"  - {cfg['name']}: {len(cfg['data_files'])} files, {cfg['num_threads']} threads, {cfg['memory_limit']} RAM")

    all_results_df = pd.DataFrame()

    for config in CONFIGURATIONS:
        all_results_df = run_full_pipeline(config, all_results_df)

    output_csv = 'benchmark_results.csv'
    all_results_df.to_csv(output_csv, index=False)
    
    print(f"\n--- All benchmarks complete ---")
    print(all_results_df[['config_name', 'step', 'execution_time_s', 'peak_memory_mib']])
    print(f"Full results saved to {output_csv}")
    
    visualize_results(all_results_df, 'benchmark_timeline.html')