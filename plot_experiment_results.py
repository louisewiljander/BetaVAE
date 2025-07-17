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

# Add a marker for each epoch
marker_styles = ["o", "s", "D", "^", "v", "<", ">", "p", "h", "H", "*", "P", "X"]
marker_map = {b: marker_styles[i % len(marker_styles)] for i, b in enumerate(unique_beta_starts)}

# Sort run_data by beta_start_val descending (highest first)
run_data.sort(key=lambda r: r["beta_start_val"], reverse=True)

beta_start_to_runs = defaultdict(list)
for i, r in enumerate(run_data):
    b = r['label'].split('=')[1].split('→')[0] if 'β=' in r['label'] else r['label']
    beta_start_to_runs[b].append(i)

# mpl.rcParams.update({
#     "font.family": "serif",
#     "font.serif": "Times New Roman",
#     "axes.labelsize": 13,
#     "axes.titlesize": 15,
#     "xtick.labelsize": 11,
#     "ytick.labelsize": 11,
#     "legend.fontsize": 9,
#     "lines.linewidth": 2,
#     "lines.markersize": 7,
#     "axes.grid": True,
#     "grid.alpha": 0.2,
# })

sns.set(style="white")
font_prop = FontProperties(family='serif')
font_prop.set_name('Times New Roman') 

for metric, title, ylabel in plot_metrics:
    plt.figure(figsize=(7, 4.2))
    plotted = False
    all_epochs_combined = set()
    for run_idx, run in enumerate(run_data):
        hist = run["history"]
        x_axis = "epoch" if "epoch" in hist else "_step"
        if metric == "elbo":
            if "val_recon_loss" in hist and "val_kl_loss" in hist:
                valid = hist[[x_axis, "val_recon_loss", "val_kl_loss"]].dropna()
                if not valid.empty:
                    valid["elbo"] = valid["val_recon_loss"] + valid["val_kl_loss"]
                    all_epochs_combined.update(valid[x_axis].astype(int).tolist())
                    b = run['label'].split('=')[1].split('→')[0] if 'β=' in run['label'] else run['label']
                    cmap = base_colors[b]
                    color = cmap(0.7)
                    marker = marker_map[b]
                    linestyle = ':' if run['label'].startswith('Annealed β=') else '-'
                    plt.plot(valid[x_axis], valid["elbo"], label=run["label"], color=color, linestyle=linestyle, marker=marker, markersize=5)
                    # Annotate final value next to the last point
                    final_x = valid[x_axis].iloc[-1]
                    final_y = valid["elbo"].iloc[-1]
                    plt.text(final_x + 0.2, final_y, f"{final_y:.2f}", color=color, fontsize=7, fontname='Times New Roman', va='center', ha='left', weight='bold', bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=0.5))
                    plotted = True
                else:
                    print(f"[WARNING] All values for 'elbo' are NaN in run '{run['label']}'")
            else:
                print(f"[WARNING] val_recon_loss or val_kl_loss not found in run '{run['label']}' for elbo plot.")
        else:
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
                    all_epochs_combined.update(valid[x_axis].astype(int).tolist())
                    b = run['label'].split('=')[1].split('→')[0] if 'β=' in run['label'] else run['label']
                    cmap = base_colors[b]
                    color = cmap(0.7)
                    marker = marker_map[b]
                    linestyle = ':' if run['label'].startswith('Annealed β=') else '-'
                    plt.plot(valid[x_axis], valid[col], label=run["label"], color=color, linestyle=linestyle, marker=marker, markersize=5)
                    final_x = valid[x_axis].iloc[-1]
                    final_y = valid[col].iloc[-1]
                    plt.text(final_x + 0.2, final_y, f"{final_y:.2f}", color=color, fontsize=7, fontname='Times New Roman', va='center', ha='left', weight='bold', bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=0.5))
                    plotted = True
                else:
                    print(f"[WARNING] All values for '{col}' are NaN in run '{run['label']}'")
            else:
                print(f"[WARNING] Metric '{metric}' not found in run '{run['label']}'. Available columns: {list(hist.columns)}")
    # Set x-axis ticks based on epochs present in this dataset
    if all_epochs_combined:
        min_epoch = min(all_epochs_combined)
        max_epoch = max(all_epochs_combined)
        epoch_ticks = list(range(min_epoch, max_epoch + 2))  # +1 for "number of epochs + 1"
    else:
        min_epoch = 1
        max_epoch = 20
        epoch_ticks = list(range(min_epoch, max_epoch + 2))
    plt.title(title, fontname='Times New Roman', fontweight='bold')
    plt.xlabel("Epoch", fontname='Times New Roman', fontsize=11)
    plt.ylabel(ylabel, fontname='Times New Roman', fontsize=11)
    plt.xlim(min_epoch, max_epoch)
    plt.xticks(epoch_ticks, fontname="Times New Roman", fontsize=10)
    plt.yticks(fontname="Times New Roman", fontsize=10)
    plt.grid(axis='y', color='gray', alpha=0.08, linewidth=1, linestyle='-')
    if metric == "elbo":
        plt.ylim(auto=True)
    handles, labels = plt.gca().get_legend_handles_labels()
    main_handles = [h for h, l in zip(handles, labels) if l and not l.startswith("Final ELBO")]
    main_labels = [l for l in labels if l and not l.startswith("Final ELBO")]
    if main_handles:
        fig = plt.gcf()
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

# --- Mean Beta-Loss Plots ---
for metric, title, ylabel in plot_metrics:
    plt.figure(figsize=(7, 4.2))
    plotted = False
    if PLOT_AGGREGATED_ONLY:
        # Plot only aggregated results (mean for each group, per epoch)
        # Collect all lines to sort legend later
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
            # Use the same color/marker scheme as individual runs
            def beta_fmt(val):
                return f"{val:.1f}" if 0 < val < 1 else f"{int(round(val))}"
            beta_start_fmt = beta_fmt(beta_start)
            # Compose label as in run_data
            if float(beta_start) == 1 and float(beta_end) == 1:
                label = "Standard VAE"
            elif float(beta_start) == float(beta_end):
                label = f"β={beta_start_fmt}"
            else:
                label = f"Annealed β={beta_start_fmt}→{beta_fmt(beta_end)}"
            cmap = base_colors.get(beta_start_fmt, plt.cm.Blues)
            color = cmap(0.7)
            marker = marker_map.get(beta_start_fmt, 'o')
            # Slimmer lines
            # Use dotted lines for annealed, solid for fixed
            if float(beta_start) != float(beta_end):
                linestyle = ':'
            else:
                linestyle = '-'
            # Plot mean only (no std)
            line, = plt.plot(epochs, means, label=label, color=color, marker=marker, linestyle=linestyle, linewidth=1.2, markersize=4)
            # Add more room to the right for labels
            xlim_right = max(epochs) + 2 if epochs else 12
            plt.xlim(1, xlim_right)
            if epochs:
                plt.text(epochs[-1] + 1.0, means[-1], f"{means[-1]:.2f}", color=color, fontsize=8, fontname='Times New Roman', va='center', ha='left', weight='bold', bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=0.5))
            plot_lines.append(line)
            legend_labels.append(label)
            # For sorting: use beta_start as float (descending)
            legend_betas.append(float(beta_start))
            plotted = True
        # Sort lines, labels, and betas by beta_start descending, and plot in that order
        if plot_lines:
            sorted_tuples = sorted(zip(legend_betas, plot_lines, legend_labels), key=lambda x: x[0], reverse=True)
            legend_betas[:], plot_lines[:], legend_labels[:] = zip(*sorted_tuples)
            # Re-plot the lines in sorted order
            plt.cla()  # Clear current axes
            for i, line in enumerate(plot_lines):
                plt.plot(line.get_xdata(), line.get_ydata(), label=legend_labels[i], color=line.get_color(), marker=line.get_marker(), linestyle=line.get_linestyle(), linewidth=line.get_linewidth(), markersize=line.get_markersize())
                # Re-annotate the final value
                if len(line.get_xdata()) > 0:
                    final_x = line.get_xdata()[-1]
                    final_y = line.get_ydata()[-1]
                    plt.text(final_x + 1.0, final_y, f"{final_y:.2f}", color=line.get_color(), fontsize=8, fontname='Times New Roman', va='center', ha='left', weight='bold', bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=0.5))
            fig = plt.gcf()
            fig.set_size_inches(10, 4.2)
            leg = plt.legend(plot_lines, legend_labels, loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False, prop=font_prop, borderaxespad=0.1, handlelength=1.0, handletextpad=0.2, borderpad=0.2, labelspacing=0.2, columnspacing=0.5)
            for text in leg.get_texts():
                text.set_fontsize(9)
            leg._legend_box.align = "left"
            plt.subplots_adjust(right=0.7)
        else:
            plt.legend().remove()
    else:
        # Plot every training run as before
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
                    # Assign color and marker based on beta_start (no shade variation)
                    b = run['label'].split('=')[1].split('→')[0] if 'β=' in run['label'] else run['label']
                    cmap = base_colors[b]
                    color = cmap(0.7)
                    marker = marker_map[b]
                    # Determine linestyle: dashed for annealed, solid for fixed
                    linestyle = ':' if run['label'].startswith('Annealed β=') else '-'
                    plt.plot(valid[x_axis], valid[col], label=run["label"], color=color, linestyle=linestyle, marker=marker, markersize=5)
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
    plt.xlim(1, 11)
    plt.xticks(range(0, 23), fontname="Times New Roman", fontsize=10)
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
print(f"Plots saved as PNG files in {plots_dir}.")

# --- Aggregated line plot: Mean ELBO (val_recon_loss + val_kl_loss) per epoch, grouped by beta_start, for each dataset (last 3 runs) ---
from collections import defaultdict
elbo_grouped = defaultdict(list)  # {(dataset, beta_start): list of (beta_end, metrics_dict)}
for (dataset, beta_start, beta_end), metrics_dict in aggregated_results.items():
    elbo_grouped[(dataset, beta_start)].append((beta_end, metrics_dict))

for (dataset, beta_start), group in elbo_grouped.items():
    def beta_fmt(val):
        return f"{val:.1f}" if 0 < val < 1 else f"{int(round(val))}"
    beta_start_fmt = beta_fmt(beta_start)
    if dataset != DATASET_FILTER:
        continue
    plt.figure(figsize=(8, 4.2))
    plotted = False
    for beta_end, metrics_dict in group:
        # Only plot if both losses are present
        if 'val_recon_loss' not in metrics_dict or 'val_kl_loss' not in metrics_dict:
            continue
        all_epochs = set(metrics_dict['val_recon_loss'].keys()) | set(metrics_dict['val_kl_loss'].keys())
        all_epochs = sorted(int(e) for e in all_epochs)
        means = []
        for e in all_epochs:
            vrl = metrics_dict['val_recon_loss'].get(e, (np.nan,))[0]
            vkl = metrics_dict['val_kl_loss'].get(e, (np.nan,))[0]
            means.append(vrl + vkl if not np.isnan(vrl) and not np.isnan(vkl) else np.nan)
        # Compose label for line
        if float(beta_start) == 1 and float(beta_end) == 1:
            label = "Standard VAE"
        elif float(beta_start) == float(beta_end):
            label = f"β={beta_start_fmt}"
        else:
            label = f"Annealed β={beta_start_fmt}→{beta_fmt(beta_end)}"
        cmap = base_colors.get(beta_start_fmt, plt.cm.Blues)
        color = cmap(0.7)
        marker = marker_map.get(beta_start_fmt, 'o')
        linestyle = ':' if float(beta_start) != float(beta_end) else '-'
        plt.plot(all_epochs, means, label=label, color=color, linestyle=linestyle, marker=marker, markersize=5)
        if all_epochs and not all(np.isnan(means)):
            for i in range(len(all_epochs)-1, -1, -1):
                if not np.isnan(means[i]):
                    plt.text(all_epochs[i] + 0.2, means[i], f"{means[i]:.2f}", color=color, fontsize=7, fontname='Times New Roman', va='center', ha='left', weight='bold', bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=0.5))
                    break
        plotted = True
    if plotted:
        plt.title(f'Mean Negative Validation ELBO (Recon + KL, β = 1) per Epoch\nβ={beta_start_fmt} (fixed & annealed) | Dataset: {dataset}', fontname='Times New Roman', fontweight='bold')
        plt.xlabel('Epoch', fontname='Times New Roman', fontsize=11)
        plt.ylabel('Negative ELBO = Reconstruction loss + KL loss (β = 1)', fontname='Times New Roman', fontsize=11)
        plt.xlim(min_epoch, max_epoch)
        plt.xticks(epoch_ticks, fontname='Times New Roman', fontsize=10)
        plt.yticks(fontname='Times New Roman', fontsize=10)
        plt.grid(axis='y', color='gray', alpha=0.08, linewidth=1, linestyle='-')
        plt.legend()
        plt.tight_layout(rect=[0, 0, 0.95, 1])
        out_path_elbo_line = os.path.join(plots_dir, f"val_elbo_per_epoch_{dataset}_beta{beta_start_fmt}_grouped.png")
        plt.savefig(out_path_elbo_line, dpi=300, bbox_inches='tight')
        print(f"Mean ELBO per-epoch line plot saved as {out_path_elbo_line}.")