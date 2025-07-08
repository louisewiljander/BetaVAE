import os
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from dotenv import load_dotenv
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties
from collections import defaultdict
import numpy as np

# Load environment variables for wandb entity and project
load_dotenv()
entity = os.getenv("WANDB_ENTITY")
project = os.getenv("WANDB_PROJECT")

api = wandb.Api()
runs = api.runs(f"{entity}/{project}")

# Set this to filter runs by dataset (e.g., 'mnist', 'dsprites', 'cifar10')
DATASET_FILTER = 'dsprites'  # Set to None to disable filtering

# Metrics to fetch from each run
metrics = ['standard_elbo', 'standard_kl_loss', 'standard_recon_loss', 'beta', 
           'beta_start', 'beta_end','val_kl_loss', 'val_recon_loss', 'val_loss', 'epoch', '_step']
plot_metrics = [
    ("val_loss", "Total Loss", "Loss"),
    ("val_kl_loss", "Distribution (KL) Loss", "Loss"),
    ("val_recon_loss", "Reconstruction Loss ", "Loss"),
]

# Ensure plots directory exists
plots_dir = os.path.join("results", "plots")
os.makedirs(plots_dir, exist_ok=True)

run_data = []
#print("Fetched runs from wandb:") # Uncomment for debugging
for run in runs:
    if run.state != "finished":
        continue
    config = run.config
    if DATASET_FILTER is not None and config.get("dataset", None) != DATASET_FILTER:
        continue
    # print(f"- {run.name} (ID: {run.id}, State: {run.state})") # Uncomment for debugging
    beta_start_val = config.get('beta_start', config.get('beta', 1.0))
    beta_end_val = config.get('beta_end', beta_start_val)
    # Format beta as float if between 0 and 1, else as int
    def beta_fmt(val):
        return f"{val:.1f}" if 0 < val < 1 else f"{int(round(val))}"
    beta_start = beta_fmt(beta_start_val)
    beta_end = beta_fmt(beta_end_val)
    if float(beta_start_val) == 1 and float(beta_end_val) == 1:
        label = "Standard VAE"
    elif float(beta_start_val) == float(beta_end_val):
        label = f"β={beta_start}"
    else:
        label = f"Annealed β={beta_start}→{beta_end}"
    try:
        history = run.history(pandas=True)
    except Exception as e:
        print(f"  [ERROR] Could not fetch history for run {run.id}: {e}")
        continue
    # Options for debugging:
    # print(history.head())  # Uncomment to see first few rows of history
    # print(f"  Available metrics for this run: {list(history.columns)}") # Uncomment to see available metrics
    # print(f"  run.summary: {dict(run.summary)}")
    # print(f"  run.config: {dict(run.config)}") 
    run_data.append({
        "label": label,
        "history": history,
        "run_id": run.id,
        "beta_start_val": beta_start_val
    })

# Sort run_data by beta_start_val descending (highest first)
run_data.sort(key=lambda r: r["beta_start_val"], reverse=True)

# Build color mapping for beta_start
unique_beta_starts = sorted(set([r['label'].split('=')[1].split('→')[0] if 'β=' in r['label'] else r['label'] for r in run_data]))
color_maps = [plt.cm.Blues, plt.cm.Greens, plt.cm.Reds, plt.cm.Purples, plt.cm.Oranges, plt.cm.Greys]
base_colors = {b: color_maps[i % len(color_maps)] for i, b in enumerate(unique_beta_starts)}
beta_start_to_runs = defaultdict(list)
for i, r in enumerate(run_data):
    b = r['label'].split('=')[1].split('→')[0] if 'β=' in r['label'] else r['label']
    beta_start_to_runs[b].append(i)

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": "Times New Roman",
    "axes.labelsize": 13,
    "axes.titlesize": 15,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 9,
    "lines.linewidth": 2,
    "lines.markersize": 7,
    "axes.grid": True,
    "grid.alpha": 0.2,
})

sns.set(style="white")
font_prop = FontProperties(family='serif')
font_prop.set_name('Times New Roman') 
for metric, title, ylabel in plot_metrics:
    plt.figure(figsize=(7, 4.2))
    plotted = False
    for run_idx, run in enumerate(run_data):
        hist = run["history"]
        x_axis = "epoch" if "epoch" in hist else "_step"
        col = metric if metric in hist else None
        if col is None:
            matches = [c for c in hist.columns if metric.lower() == c.lower()]
            if not matches:
                matches = [c for c in hist.columns if metric.lower() in c.lower()]
            if matches:
                col = matches[0]
                print(f"[INFO] Using column '{col}' for metric '{metric}' in run '{run['label']}'")
        if col:
            valid = hist[[x_axis, col]].dropna()
            if not valid.empty:
                # Assign color based on beta_start and shade based on run index
                b = run['label'].split('=')[1].split('→')[0] if 'β=' in run['label'] else run['label']
                cmap = base_colors[b]
                shade_idx = beta_start_to_runs[b].index(run_idx)
                n_shades = len(beta_start_to_runs[b])
                color = cmap(0.4 + 0.5 * shade_idx / max(n_shades-1,1)) if n_shades > 1 else cmap(0.7)
                plt.plot(valid[x_axis], valid[col], label=run["label"], color=color)
                plt.plot(valid[x_axis].iloc[-1], valid[col].iloc[-1], marker='o', color=color, markersize=4)
                # Annotate final value next to the last point
                final_x = valid[x_axis].iloc[-1]
                final_y = valid[col].iloc[-1]
                plt.text(final_x + 0.2, final_y, f"{final_y:.2f}", color=color, fontsize=7, fontname='Times New Roman', va='center', ha='left', weight='bold', bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=0.5))
                plotted = True
            else:
                print(f"[WARNING] All values for '{col}' are NaN in run '{run['label']}'")
        else:
            print(f"[WARNING] Metric '{metric}' not found in run '{run['label']}'. Available columns: {list(hist.columns)}")
    plt.title(title, fontname='Times New Roman', fontweight='bold')
    plt.xlabel("Epoch", fontname='Times New Roman', fontsize=11)
    plt.ylabel(ylabel, fontname='Times New Roman', fontsize=11)
    plt.xlim(0, 10)
    plt.xticks(range(0, 11), fontname="Times New Roman", fontsize=10)
    plt.yticks(fontname="Times New Roman", fontsize=10)
    plt.grid(axis='y', color='gray', alpha=0.08, linewidth=1, linestyle='-')
    handles, labels = plt.gca().get_legend_handles_labels()
    main_handles = [h for h, l in zip(handles, labels) if l and not l.startswith("Final ELBO")]
    main_labels = [l for l in labels if l and not l.startswith("Final ELBO")]
    if main_handles:
        fig = plt.gcf()
        # Adjust figure to be wider for legend
        fig.set_size_inches(10, 4.2)
        leg = plt.legend(main_handles, main_labels, loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False, prop=font_prop, borderaxespad=0.1, handlelength=1.0, handletextpad=0.2, borderpad=0.2, labelspacing=0.2, columnspacing=0.5)
        for text in leg.get_texts():
            text.set_fontsize(9)
        leg._legend_box.align = "left"
        plt.subplots_adjust(right=0.7)
    else:
        plt.legend().remove()
    plt.tight_layout(rect=[0, 0, 0.7, 1])
    out_path_png = os.path.join(plots_dir, f"{metric}_{DATASET_FILTER}.png")
    plt.savefig(out_path_png, dpi=300, bbox_inches='tight')
    #if plotted:
        #plt.show()
    #else:
        #print(f"[WARNING] No data to plot for metric: {metric}")

print(f"Plots saved as PNG files in {plots_dir}.")