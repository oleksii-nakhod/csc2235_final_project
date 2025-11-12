import pandas as pd
import plotly.express as px
import plotly.io as pio
import sys
import os
import glob

def visualize_config_comparison(results_df, output_file):
    """
    Generates an interactive Plotly timeline for a single framework's
    configuration comparison.
    """
    if results_df.empty:
        print(f"  No data for config comparison plot: {output_file}")
        return

    req_cols = ['start_time', 'end_time', 'execution_time_s', 'config_name', 'step']
    if not all(col in results_df.columns for col in req_cols):
        print(f"  Result DataFrame is missing required columns for timeline.")
        return

    print(f"  Generating config comparison: {output_file}")
    pio.templates.default = "plotly_dark"

    results_df['start_time'] = pd.to_numeric(results_df['start_time'])
    results_df['execution_time_s'] = pd.to_numeric(results_df['execution_time_s'])
    results_df['end_time'] = pd.to_numeric(results_df['end_time'])
    
    fig = px.bar(
        results_df,
        base="start_time",
        x="execution_time_s",
        y="config_name",
        color="step",
        orientation='h',
        title=f"Framework Config Comparison",
        hover_data=['execution_time_s', 'peak_memory_mib', 'cpu_time_s', 'start_time', 'end_time']
    )

    fig.update_yaxes(autorange="reversed")
    
    fig.update_layout(
        xaxis_title="Time (seconds)",
        yaxis_title="Benchmark Configuration",
        legend_title="Pipeline Step",
        xaxis=dict(range=[0, results_df['end_time'].max() * 1.05])
    )
    
    fig.write_html(output_file)

def make_safe_filename(pct_str):
    """Converts '1%' to '1pct' or '100%' to '100pct' for clean filenames."""
    return str(pct_str).replace('%', 'pct')

def visualize_framework_comparison(all_results_df, output_dir):
    """
    Generates framework comparison bar charts for *each*
    data/system combination found in the results.
    
    1. A detailed Gantt chart (timeline) for pipeline step comparison.
    2. A simple bar chart for peak memory comparison.
    """
    if all_results_df.empty:
        print(f"  No data for framework comparison plot.")
        return
        
    print(f"  Generating framework comparison plots in: {output_dir}")
    
    combinations = all_results_df[['data_size_pct', 'system_size_pct']].drop_duplicates()
    
    if combinations.empty:
        print("  No benchmark combinations found to compare.")
        return
        
    print(f"  Found {len(combinations)} benchmark combinations to plot:")

    for index, row in combinations.iterrows():
        data_pct = row['data_size_pct']
        sys_pct = row['system_size_pct']
        
        data_pct_safe = make_safe_filename(data_pct)
        sys_pct_safe = make_safe_filename(sys_pct)
        
        print(f"    - Generating plots for: {data_pct} Data, {sys_pct} System")
        
        current_combo_df = all_results_df.loc[
            (all_results_df['data_size_pct'] == data_pct) &
            (all_results_df['system_size_pct'] == sys_pct)
        ]
        
        if current_combo_df.empty:
            print("      (No data for this combo, skipping)")
            continue

        try:
            if not current_combo_df.empty:
                title = f'Framework Pipeline Timeline ({data_pct} Data, {sys_pct} System)'
                filename = f"framework_comparison_timeline_{data_pct_safe}_data_{sys_pct_safe}_sys.html"
                
                current_combo_df['start_time'] = pd.to_numeric(current_combo_df['start_time'])
                current_combo_df['execution_time_s'] = pd.to_numeric(current_combo_df['execution_time_s'])
                current_combo_df['end_time'] = pd.to_numeric(current_combo_df['end_time'])

                fig_time = px.bar(
                    current_combo_df,
                    base="start_time",
                    x="execution_time_s",
                    y="framework",
                    color="step",
                    orientation='h',
                    title=title,
                    hover_data=['execution_time_s', 'peak_memory_mib', 'cpu_time_s', 'start_time', 'end_time']
                )
                
                fig_time.update_yaxes(autorange="reversed")
                fig_time.update_layout(
                    xaxis_title="Time (seconds)",
                    yaxis_title="Framework",
                    legend_title="Pipeline Step",
                    xaxis=dict(range=[0, current_combo_df['end_time'].max() * 1.05])
                )
                fig_time.write_html(os.path.join(output_dir, filename))
            else:
                print(f"      (No data for {data_pct}/{sys_pct} timeline, skipping)")

        except Exception as e:
            print(f"      Could not generate timeline plot: {e}")

        try:
            peak_mem_df = current_combo_df.groupby('framework')['peak_memory_mib'].max().reset_index()
            
            if not peak_mem_df.empty:
                title = f'Peak Pipeline Memory ({data_pct} Data, {sys_pct} System)'
                filename = f"framework_comparison_peak_mem_{data_pct_safe}_data_{sys_pct_safe}_sys.html"

                fig_mem = px.bar(
                    peak_mem_df,
                    x='framework',
                    y='peak_memory_mib',
                    color='framework',
                    title=title
                )
                fig_mem.write_html(os.path.join(output_dir, filename))
        except Exception as e:
            print(f"      Could not generate peak memory plot: {e}")


if __name__ == "__main__":
    """
    This script is called by run_fair.sh to generate all visualizations.
    """
    if len(sys.argv) < 2:
        print(f"Usage: python3 {sys.argv[0]} <timestamped_results_dir>")
        sys.exit(1)
        
    TOP_LEVEL_RESULTS_DIR = sys.argv[1]
    
    if not os.path.isdir(TOP_LEVEL_RESULTS_DIR):
        print(f"Error: Results directory not found: {TOP_LEVEL_RESULTS_DIR}")
        sys.exit(1)

    for pipeline_dir in glob.glob(os.path.join(TOP_LEVEL_RESULTS_DIR, '*')):
        if not os.path.isdir(pipeline_dir):
            continue
            
        pipeline_name = os.path.basename(pipeline_dir)
        print(f"\nProcessing pipeline: {pipeline_name}")
        
        all_framework_dfs = []
        
        for framework_dir in glob.glob(os.path.join(pipeline_dir, '*')):
            if not os.path.isdir(framework_dir):
                continue
                
            framework_name = os.path.basename(framework_dir)
            
            csv_path = os.path.join(framework_dir, f"{framework_name}_results.csv")
            if not os.path.exists(csv_path):
                print(f"  No CSV found at {csv_path}")
                continue
                
            print(f"  Found results for framework: {framework_name}")
            try:
                df = pd.read_csv(csv_path)
                df['framework'] = framework_name
                all_framework_dfs.append(df)
                
                plot_path = os.path.join(framework_dir, f"config_comparison_timeline.html")
                visualize_config_comparison(df, plot_path)
                
            except Exception as e:
                print(f"  Failed to process {csv_path}: {e}")

        if all_framework_dfs:
            pipeline_results_df = pd.concat(all_framework_dfs, ignore_index=True)
            visualize_framework_comparison(pipeline_results_df, pipeline_dir)

    print("\nVisualization generation complete.")