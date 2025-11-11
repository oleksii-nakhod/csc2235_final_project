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

def visualize_framework_comparison(all_results_df, output_dir):
    """
    Generates framework comparison bar charts for key metrics.
    """
    if all_results_df.empty:
        print(f"  No data for framework comparison plot.")
        return
        
    print(f"  Generating framework comparison plots in: {output_dir}")
    
    try:
        total_time_df = all_results_df.loc[
            (all_results_df['data_size_pct'] == '100%') &
            (all_results_df['system_size_pct'] == '100%')
        ].groupby('framework')['execution_time_s'].sum().reset_index()
        
        if not total_time_df.empty:
            fig_time = px.bar(
                total_time_df,
                x='framework',
                y='execution_time_s',
                color='framework',
                title='Total Pipeline Time (100% Data, 100% System)'
            )
            fig_time.write_html(os.path.join(output_dir, 'framework_comparison_total_time.html'))
    except Exception as e:
        print(f"  Could not generate total time plot: {e}")

    try:
        peak_mem_df = all_results_df.loc[
            (all_results_df['data_size_pct'] == '100%') &
            (all_results_df['system_size_pct'] == '100%')
        ].groupby('framework')['peak_memory_mib'].max().reset_index()
        
        if not peak_mem_df.empty:
            fig_mem = px.bar(
                peak_mem_df,
                x='framework',
                y='peak_memory_mib',
                color='framework',
                title='Peak Pipeline Memory (100% Data, 100% System)'
            )
            fig_mem.write_html(os.path.join(output_dir, 'framework_comparison_peak_mem.html'))
    except Exception as e:
        print(f"  Could not generate peak memory plot: {e}")

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