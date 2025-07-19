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
font_prop = FontProperties(family='Times New Roman', weight='bold', size=10)

# Load environment variables for wandb entity and project
load_dotenv()
entity = os.getenv("WANDB_ENTITY")
project = os.getenv("WANDB_PROJECT")

api = wandb.Api()
runs = api.runs(f"{entity}/{project}")

# --- Tunable parameters --- #
# Set this to filter runs by dataset (e.g., 'mnist' or 'dsprites')
DATASET_FILTER = 'mnist'  # Set to None to disable filtering
PLOT_AGGREGATED_ONLY = True  # Set True to plot only aggregated results, False to plot all runs

# Metrics to fetch from each run
metrics = ['standard_elbo', 'standard_kl_loss', 'standard_recon_loss', 'beta', 
           'beta_start', 'beta_end','val_kl_loss', 'val_recon_loss', 'val_loss', 'epoch', '_step']
plot_metrics = [
    ("val_loss", f"β-loss ({DATASET_FILTER})", "β-loss"),
    ("val_kl_loss", f"Distribution (KL) Loss ({DATASET_FILTER})", "Loss"),
    ("val_recon_loss", f"Reconstruction Loss ({DATASET_FILTER})", "Loss"),
    ("elbo", f"ELBO ({DATASET_FILTER})", "Negative ELBO = Reconstruction loss + KL loss (β = 1)"),
]

# Ensure plots directory exists
plots_dir = os.path.join("results", "plots")
os.makedirs(plots_dir, exist_ok=True)

# --- Aggregate results from the last 3 runs
from collections import defaultdict

# Group runs by (dataset, beta_start, beta_end)
grouped_runs = defaultdict(list)
for run in runs:
    config = run.config
    dataset = config.get('dataset', None)
    beta_start = config.get('beta_start', config.get('beta', 1.0))
    beta_end = config.get('beta_end', beta_start)
    key = (dataset, beta_start, beta_end)
    grouped_runs[key].append(run)

# For each group, sort by creation time and keep the last 3 runs
aggregated_results = defaultdict(dict)  # {(dataset, beta_start, beta_end): {metric: (mean, std, n)}}
for key, run_list in grouped_runs.items():
    run_list = sorted(run_list, key=lambda r: r.created_at, reverse=True)[:3]
    for metric, _, _ in plot_metrics:
        epoch_vals = defaultdict(list)  # {epoch: [values]}
        for run in run_list:
            try:
                hist = run.history(pandas=True)
                x_axis = "epoch" if "epoch" in hist else "_step"
                col = metric if metric in hist else None
                if col is None:
                    matches = [c for c in hist.columns if metric.lower() == c.lower()]
                    if not matches:
                        matches = [c for c in hist.columns if metric.lower() in c.lower()]
                    if matches:
                        col = matches[0]
                if col:
                    valid = hist[[x_axis, col]].dropna()
                    for _, row in valid.iterrows():
                        epoch = int(row[x_axis])
                        epoch_vals[epoch].append(row[col])
            except Exception as e:
                print(f"[ERROR] Could not fetch history for run {run.id}: {e}")
        metric_epoch_stats = {}
        for epoch, vals in epoch_vals.items():
            if vals:
                mean_val = np.mean(vals)
                std_val = np.std(vals)
                metric_epoch_stats[epoch] = (mean_val, std_val, len(vals))
        if metric_epoch_stats:
            aggregated_results[key][metric] = metric_epoch_stats


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
        # Compute ELBO if both columns exist
        if 'val_recon_loss' in history and 'val_kl_loss' in history:
            history['elbo'] = history['val_recon_loss'] + history['val_kl_loss']
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

# Global color mapping for beta_start
all_beta_starts = set()
for run in runs:
    config = run.config
    beta_start_val = config.get('beta_start', config.get('beta', 1.0))
    # Format beta as float if between 0 and 1, else as int
    def beta_fmt(val):
        return f"{val:.1f}" if 0 < val < 1 else f"{int(round(val))}"
    beta_start = beta_fmt(beta_start_val)
    all_beta_starts.add(beta_start)
# Sort in descending order for color assignment
unique_beta_starts = sorted(all_beta_starts, reverse=True)
color_maps = [plt.cm.Blues, plt.cm.Greens, plt.cm.Reds, plt.cm.Greys]
base_colors = {b: color_maps[i % len(color_maps)] for i, b in enumerate(unique_beta_starts)}
# Explicit RGBA color mapping for each beta_start value
explicit_colors = {
    "8": (0.1791464821222607, 0.49287197231833907, 0.7354248366013072, 1.0),
    "4": (0.18246828143021915, 0.5933256439830834, 0.3067589388696655, 1.0),
    "10": (0.8503344867358708, 0.14686658977316416, 0.13633217993079583, 1.0),
    "1": (0.3713033448673587, 0.3713033448673587, 0.3713033448673587, 1.0),
    "0.1": (0.8871510957324106, 0.3320876585928489, 0.03104959630911188, 1.0)
}
base_colors = {b: explicit_colors.get(str(b), (0.5, 0.5, 0.5, 1.0)) for b in unique_beta_starts}
print("Base color mapping for beta_start values:")
for k, v in base_colors.items():
    print(f"  beta_start: {k}, color: {v}")

# Add a marker for each epoch
marker_styles = ["o", "s", "D", "^", "*"]
marker_map = {b: marker_styles[i % len(marker_styles)] for i, b in enumerate(unique_beta_starts)}
# Assign star marker to beta_start = 1 and triangle marker to beta_start = 0.1
if "1" in marker_map:
    marker_map["1"] = "*"
if "0.1" in marker_map:
    marker_map["0.1"] = "^"

# Sort run_data by beta_start_val descending (highest first)
run_data.sort(key=lambda r: r["beta_start_val"], reverse=True)

beta_start_to_runs = defaultdict(list)
for i, r in enumerate(run_data):
    b = r['label'].split('=')[1].split('→')[0] if 'β=' in r['label'] else r['label']
    beta_start_to_runs[b].append(i)



# ===================== PLOTTING SECTION =====================

# --- 1. Aggregated Mean Beta-Loss Plots (mean over last 3 runs per group) ---
for metric, title, ylabel in plot_metrics:
    # Aggregated mean plots for each metric, grouped by (dataset, beta_start, beta_end)
    if metric not in ["val_recon_loss", "val_kl_loss"]:
        continue
    plt.figure(figsize=(8, 4.2), constrained_layout=True)
    plotted = False
    plot_lines = []
    legend_labels = []
    legend_betas = []
    for (dataset, beta_start, beta_end), metrics_dict in aggregated_results.items():
        if dataset != DATASET_FILTER:
            continue
        if metric not in metrics_dict:
            continue
        epoch_stats = metrics_dict[metric]  # {epoch: (mean, std, n)}
        epochs = sorted(epoch_stats.keys())
        means = [epoch_stats[e][0] for e in epochs]
        def beta_fmt(val):
            return f"{val:.1f}" if 0 < val < 1 else f"{int(round(val))}"
        beta_start_fmt = beta_fmt(beta_start)
        if float(beta_start) == 1 and float(beta_end) == 1:
            label = "Standard VAE"
        elif float(beta_start) == float(beta_end):
            label = f"β={beta_start_fmt}"
        else:
            label = f"Annealed β={beta_start_fmt}→{beta_fmt(beta_end)}"
        color = base_colors.get(beta_start_fmt, (0.5, 0.5, 0.5, 1.0))
        marker = marker_map.get(beta_start_fmt, 'o')
        linestyle = ':' if float(beta_start) != float(beta_end) else '-'
        line, = plt.plot(epochs, means, label=label, color=color, marker=marker, linestyle=linestyle, linewidth=1.3, markersize=4.5)
        plot_lines.append(line)
        legend_labels.append(label)
        legend_betas.append(float(beta_start))
        plotted = True
    # Sort legend entries: highest to lowest beta_start, Standard VAE at the bottom
    if plot_lines:
        # Prepare tuples for sorting
        legend_tuples = []
        for beta, line, label in zip(legend_betas, plot_lines, legend_labels):
            if label == "Standard VAE":
                sort_key = float('-inf')  # Always last
            else:
                sort_key = beta
            legend_tuples.append((sort_key, line, label))
        # Sort by sort_key descending
        legend_tuples_sorted = sorted(legend_tuples, key=lambda x: x[0], reverse=True)
        # Unpack sorted
        _, plot_lines_sorted, legend_labels_sorted = zip(*legend_tuples_sorted)
        plt.cla()  # Clear current axes
        # Use the same color scheme for all plots, based on beta_start_fmt
        for i, line in enumerate(plot_lines_sorted):
            label = legend_labels_sorted[i]
            if label == "Standard VAE":
                beta_key = "1"
            elif label.startswith("β="):
                beta_key = label.split('=')[1].split('→')[0]
            elif label.startswith("Annealed β="):
                beta_key = label.split('=')[1].split('→')[0]
            else:
                beta_key = str(i)
            color = base_colors.get(beta_key, (0.5, 0.5, 0.5, 1.0))
            plt.plot(line.get_xdata(), line.get_ydata(), label=label, color=color, marker=line.get_marker(), linestyle=line.get_linestyle(), linewidth=1.3, markersize=4.5)
            # Add value label for final value (last epoch) with offset
            xdata = line.get_xdata()
            ydata = line.get_ydata()
            if len(xdata) > 0 and len(ydata) > 0:
                final_x = xdata[-1]
                final_y = ydata[-1]
                plt.text(final_x + 0.3, final_y, f"{final_y:.2f}", color=color, fontsize=8, fontname='Times New Roman', va='center', ha='left', weight='bold', bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=0.5))

        fig = plt.gcf()
        fig.set_size_inches(8, 4.2)
        leg = plt.legend(plot_lines_sorted, legend_labels_sorted, loc='lower center', bbox_to_anchor=(0.5, -0.38), ncol=4, frameon=False, prop=font_prop)
        for text in leg.get_texts():
            text.set_fontsize(10)
        leg._legend_box.align = "left"
    else:
        plt.legend().remove()
    plt.tight_layout(rect=[0, 0, 0.7, 1])
    out_path_png = os.path.join(plots_dir, f"{metric}_{DATASET_FILTER}.png")
    plt.savefig(out_path_png, dpi=300, bbox_inches='tight')

# --- Mean Beta-Loss Plots ---
for metric, title, ylabel in plot_metrics:
    plt.figure(figsize=(7, 4.2))
    plotted = False
    if PLOT_AGGREGATED_ONLY:
        # Plot only aggregated results (mean for each group, per epoch)
        # Collect all lines to sort legend later
    plt.xlabel("Epoch", fontname='Times New Roman', fontsize=12)
    plt.ylabel(ylabel, fontname='Times New Roman', fontsize=12)
    if DATASET_FILTER == "dsprites":
        plt.xlim(1, 10.5)
        plt.xticks([x for x in range(1, 11)], [str(x) for x in range(1, 11)], fontname="Times New Roman", fontsize=10)
        plt.ylim(40, 200)
        plt.yticks(np.arange(40, 201, 20), fontname="Times New Roman", fontsize=10)
    else:
        plt.xlim(1, 20.5)
        plt.xticks([x for x in range(0, 20)], [str(x+1) for x in range(0, 20)], fontname="Times New Roman", fontsize=10)
        if metric == "val_kl_loss":
            plt.ylim(0, 55)
            plt.yticks(np.arange(0, 56, 5), fontname="Times New Roman", fontsize=10)
        elif metric == "val_recon_loss":
            # Expand y-axis range to include all recon loss values up to 1000 for visibility
            plt.ylim(550, 1000)
            plt.yticks(np.arange(550, 1001, 50), fontname="Times New Roman", fontsize=10)
    plt.grid(axis='y', color='gray', alpha=0.08, linewidth=1, linestyle='-')
    out_path_png = os.path.join(plots_dir, f"{metric}_{DATASET_FILTER}.png")
    plt.savefig(out_path_png, dpi=300, bbox_inches='tight')
    print(f"Plot saved as {out_path_png}")
print(f"Plots saved as PNG files in {plots_dir}.")

# --- 2. Aggregated ELBO per-epoch Plot (mean over last 3 runs, all beta grouped) ---
elbo_grouped = defaultdict(list)  # {dataset: list of (beta_start, beta_end, metrics_dict)}
for (dataset, beta_start, beta_end), metrics_dict in aggregated_results.items():
    elbo_grouped[dataset].append((beta_start, beta_end, metrics_dict))

for dataset, group in elbo_grouped.items():
    if dataset != DATASET_FILTER:
        continue
    plt.figure(figsize=(8, 4.2), constrained_layout=True)
    plotted = False
    all_epochs_combined = set()
    for beta_start, beta_end, metrics_dict in group:

        if 'val_recon_loss' not in metrics_dict or 'val_kl_loss' not in metrics_dict:
            continue
        all_epochs = set(metrics_dict['val_recon_loss'].keys()) | set(metrics_dict['val_kl_loss'].keys())
        all_epochs = sorted(int(e) for e in all_epochs)
        all_epochs_combined.update(all_epochs)
        means = []
        for e in all_epochs:
            vrl = metrics_dict['val_recon_loss'].get(e, (np.nan,))[0]
            vkl = metrics_dict['val_kl_loss'].get(e, (np.nan,))[0]
            means.append(vrl + vkl if not np.isnan(vrl) and not np.isnan(vkl) else np.nan)
        # Compose label for line
        def beta_fmt(val):
            return f"{val:.1f}" if 0 < val < 1 else f"{int(round(val))}"
        beta_start_fmt = beta_fmt(beta_start)
        if float(beta_start) == 1 and float(beta_end) == 1:
            label = "Standard VAE"
        elif float(beta_start) == float(beta_end):
            label = f"β={beta_start_fmt}"
        else:
            label = f"Annealed β={beta_start_fmt}→{beta_fmt(beta_end)}"
        color = base_colors.get(beta_start_fmt, (0.5, 0.5, 0.5, 1.0))
        marker = marker_map.get(beta_start_fmt, 'o')
        linestyle = ':' if float(beta_start) != float(beta_end) else '-'
        plt.plot(all_epochs, means, label=label, color=color, linestyle=linestyle, marker=marker, linewidth=1.3, markersize=4.5)
        if all_epochs and not all(np.isnan(means)):
            for i in range(len(all_epochs)-1, -1, -1):
                if not np.isnan(means[i]):
                    plt.text(all_epochs[i] + 0.2, means[i], f"{means[i]:.2f}", color=color, fontsize=8, fontname='Times New Roman', va='center', ha='left', weight='bold', bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=0.5))
                    break
        plotted = True
    # Set x-axis ticks based on epochs present in this dataset
    if all_epochs_combined:
        min_epoch = min(all_epochs_combined)
        max_epoch = max(all_epochs_combined)
        epoch_ticks = list(range(min_epoch, max_epoch + 2))  # +1 for "number of epochs + 1"
    else:
        min_epoch = 1
        max_epoch = 20
        epoch_ticks = list(range(min_epoch, max_epoch + 2))
    if plotted:
        plt.xlabel('Epoch', fontname='Times New Roman', fontsize=12)
        plt.ylabel('Negative ELBO = Reconstruction loss + KL loss)', fontname='Times New Roman', fontsize=14)
        if DATASET_FILTER == "dsprites":
            plt.xlim(1, 10.5)
            plt.xticks([x for x in range(1, 11)], [str(x) for x in range(1, 11)], fontname='Times New Roman', fontsize=10)
            plt.ylim(40, 200)
            plt.yticks(np.arange(40, 201, 20), fontname='Times New Roman', fontsize=10)
        else:
            plt.xlim(1, 20.5)
            plt.xticks([x for x in range(0, 20)], [str(x+1) for x in range(0, 20)], fontname='Times New Roman', fontsize=10)
            plt.ylim(600, 1000)
            plt.yticks(np.arange(600, 1001, 50), fontname='Times New Roman', fontsize=10)
        plt.grid(axis='y', color='gray', alpha=0.08, linewidth=1, linestyle='-')
        leg = plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.32), ncol=4, frameon=False, prop=font_prop)
        for text in leg.get_texts():
            text.set_fontsize(10)
        leg._legend_box.align = "left"
        out_path_elbo_line = os.path.join(plots_dir, f"val_elbo_per_epoch_{dataset}_all_beta_grouped.png")
        plt.savefig(out_path_elbo_line, dpi=300, bbox_inches='tight')
        print(f"Mean ELBO per-epoch line plot saved as {out_path_elbo_line}.")