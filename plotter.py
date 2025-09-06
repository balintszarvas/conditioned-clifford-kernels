import wandb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def fetch_wandb_runs(project_name, run_ids=None, entity=None):
    """
    Fetch specified runs from wandb project
    """
    # Initialize wandb API
    api = wandb.Api()
    
    # Get runs from project
    runs = api.runs(f"{entity}/{project_name}" if entity else project_name)
    
    # Print available runs for debugging
    print("\nAvailable runs:")
    for run in runs:
        print(f"Run name: {run.name}")
        print(f"Run ID: {run.id}")
        print(f"Run tags: {run.tags}")
        print("-" * 50)
    
    run_histories = {}
    for run in runs:
        # If run_names specified, only get those runs
        if run_ids and run.id not in run_ids:
            continue
            
        # Convert run history to pandas dataframe
        history = run.history()
        
        # Print available metrics for this run
        print(f"\nAvailable metrics for run {run.name}:")
        print(history.columns.tolist())
        
        # Store in dictionary with run name as key
        run_histories[run.name] = history
        
    return run_histories

def plot_loss_curves(run_histories, loss_key='loss', smoothing=0.0):
    """
    Plot loss curves for multiple runs
    """
    if not run_histories:
        print("No run histories were fetched. Check if run names are correct.")
        return
   
    plt.rcParams['axes.labelsize'] = 24
    plt.rcParams['axes.titlesize'] = 24
    plt.rcParams['legend.fontsize'] = 24
    plt.rcParams['xtick.labelsize'] = 24
    plt.rcParams['ytick.labelsize'] = 24

    
    plt.figure(figsize=(12, 8))
    sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
    sns.set_style("darkgrid", {"axes.facecolor": "0.95"})
    
    for run_name, history in run_histories.items():
        if loss_key not in history.columns:
            print(f"Warning: '{loss_key}' not found in run {run_name}")
            print(f"Available columns for {run_name}:")
            print(history.columns.tolist())
            continue
            
        loss_values = history[loss_key]
        if smoothing > 0:
            loss_values = loss_values.ewm(alpha=(1 - smoothing)).mean()
            
        plt.plot(history.index, loss_values, label=run_name)
    
    if plt.gca().get_lines():
        plt.xlabel('Epoch', fontsize=24)
        plt.ylabel(loss_key.capitalize(), fontsize=24)
        plt.title(f'{loss_key.capitalize()} vs Epoch', fontsize=24)
        plt.legend(fontsize=24)
        plt.yscale('log')
        plt.tight_layout()
        plt.show()
    else:
        print("No valid data to plot. Check if loss key matches available metrics.")

def main():
    # Configure these parameters
    PROJECT_NAME = "clifford-equivariant-cnns"
    ENTITY = "balintszarvas-university-of-amsterdam"
    RUN_NAMES = [  #pecify the runs you want to plot
        "7cwhvuw1",
        "bcf87sod"
    ]
    LOSS_KEY = "valid.loss_total"  # metric name
    SMOOTHING = 0.1  # smoothing factor (0-1)
    
    # Fetch and plot runs
    run_histories = fetch_wandb_runs(PROJECT_NAME, RUN_NAMES, ENTITY)
    plot_loss_curves(run_histories, LOSS_KEY, SMOOTHING)

if __name__ == "__main__":
    main()
