"""
Plotting functions for neural network simulation
All visualization and figure generation
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for faster rendering
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from brian2 import ms

from config import THESIS_STYLE, OUTPUT_DIR_RUNS

# Apply thesis style once at import
plt.rcParams.update(THESIS_STYLE)

# Reusable color constants
EXC_COLOR = "crimson"
INH_COLOR = "royalblue"


def _save_and_close(fig, filepath):
    """Save figure and close to free memory."""
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_basic_activity(SpikeMonexc, SpikeMoninh, StateMonexc, StateMoninh,
                        RateMonexc, RateMoninh, Nexc, Ninh,
                        initialsettletimeval, runtimeval, currentImidnAval,
                        overallmeancvexcval, cvwindowsize, cvstepsize,
                        groupname="Model", rcaccuracyinfo=None):
    """
    Create comprehensive activity plot showing raster, voltage, adaptation, CV, and rate.
    """
    from analysis import calculate_live_cv
    from brian2 import ms, nA, mV

    outputdir = f"{OUTPUT_DIR_RUNS}/{groupname}/"
    os.makedirs(outputdir, exist_ok=True)

    fig = plt.figure(figsize=(12, 18))
    gs = plt.GridSpec(6, 1, figure=fig, height_ratios=[0.5, 2, 1, 1, 1, 1.5])

    # Info panel
    axinfo = fig.add_subplot(gs[0, 0])
    axinfo.axis("off")
    cvtext = f"Overall Mean Exc CV (Delayed Analysis): {overallmeancvexcval:.3f}" if not np.isnan(overallmeancvexcval) else "Overall Mean Exc CV (Delayed Analysis): NA"

    rctext = ""
    if rcaccuracyinfo:
        bestacc = rcaccuracyinfo.get("best_accuracy", np.nan)
        numsamplesforbestacc = rcaccuracyinfo.get("num_samples_for_best_accuracy", "NA")
        rctext = f"\nRC MNIST Accuracy: {bestacc:.4f} with {numsamplesforbestacc} training samples"

    axinfo.text(0.5, 0.5, f"Network {groupname}\n{cvtext}{rctext}",
                ha="center", va="center", fontsize=10, wrap=True)

    # Raster plot
    ax1 = fig.add_subplot(gs[1, 0])
    if SpikeMonexc and hasattr(SpikeMonexc, "t") and len(SpikeMonexc.t) > 0:
        ax1.plot(SpikeMonexc.t/ms, SpikeMonexc.i, ".", color=EXC_COLOR, markersize=0.1, label="Excitatory")
    if SpikeMoninh and hasattr(SpikeMoninh, "t") and len(SpikeMoninh.t) > 0:
        ax1.plot(SpikeMoninh.t/ms, SpikeMoninh.i + Nexc, ".", color=INH_COLOR, markersize=0.1, label="Inhibitory")
    ax1.set_ylabel("Neuron index")
    ax1.legend(markerscale=20, loc="upper right")
    ax1.grid(True, linestyle=":", alpha=0.5)
    ax1.tick_params(labelbottom=False)

    # Voltage traces
    ax2 = fig.add_subplot(gs[2, 0], sharex=ax1)
    if StateMonexc and hasattr(StateMonexc, "V") and len(StateMonexc.t) > 0 and StateMonexc.V.shape[1] > 0:
        ax2.plot(StateMonexc.t/ms, StateMonexc.V[0]/mV, color=EXC_COLOR, linewidth=0.8, label="Exc N0 Vm")
    if StateMoninh and hasattr(StateMoninh, "V") and len(StateMoninh.t) > 0 and StateMoninh.V.shape[1] > 0:
        ax2.plot(StateMoninh.t/ms, StateMoninh.V[0]/mV, color=INH_COLOR, linewidth=0.8, label="Inh N0 Vm")
    ax2.set_ylabel("Voltage (mV)")
    ax2.grid(True, linestyle=":", alpha=0.5)
    ax2.tick_params(labelbottom=False)
    ax2.legend(loc="upper right")

    # Adaptation traces
    ax3 = fig.add_subplot(gs[3, 0], sharex=ax1)
    if StateMonexc and hasattr(StateMonexc, "A") and len(StateMonexc.t) > 0 and StateMonexc.A.shape[1] > 0:
        ax3.plot(StateMonexc.t/ms, StateMonexc.A[0]/nA, color="black", linewidth=0.8, label="Exc N0 Adapt")
    if StateMoninh and hasattr(StateMoninh, "A") and len(StateMoninh.t) > 0 and StateMoninh.A.shape[1] > 0:
        ax3.plot(StateMoninh.t/ms, StateMoninh.A[0]/nA, color="dimgray", linewidth=0.8, label="Inh N0 Adapt")
    ax3.set_ylabel("Adaptation (nA)")
    ax3.grid(True, linestyle=":", alpha=0.5)
    ax3.tick_params(labelbottom=False)
    ax3.legend(loc="upper right")

    # Live CV plot
    ax4 = fig.add_subplot(gs[4, 0], sharex=ax1)
    if SpikeMonexc:
        livecvtimesms, livecvvalues = calculate_live_cv(
            SpikeMonexc, Nexc,
            analysis_start_time=initialsettletimeval,
            analysis_duration=runtimeval,
            window_size=cvwindowsize,
            step_size=cvstepsize
        )
        if len(livecvtimesms) > 0 and not np.all(np.isnan(livecvvalues)):
            ax4.plot(livecvtimesms, livecvvalues, "m-", linewidth=1.5, label="Live CV Exc")
            ax4.legend(loc="upper right")
            ax4.grid(True, linestyle=":", alpha=0.5)
        else:
            ax4.text(0.5, 0.5, "No Live CV data", ha="center", va="center", transform=ax4.transAxes)
    else:
        ax4.text(0.5, 0.5, "No Exc Spikes for CV", ha="center", va="center", transform=ax4.transAxes)
    ax4.set_ylabel("Live CV Exc")
    ax4.tick_params(labelbottom=False)

    # Rate and input current plot
    ax5 = fig.add_subplot(gs[5, 0], sharex=ax1)
    lines_ax5, labels_ax5 = [], []

    if RateMonexc and hasattr(RateMonexc, "t") and len(RateMonexc.t) > 0:
        try:
            from brian2 import Hz
            smooth_rate_exc = RateMonexc.smooth_rate(window="gaussian", width=10*ms) / Hz
            line, = ax5.plot(RateMonexc.t/ms, smooth_rate_exc, color=EXC_COLOR, linewidth=1.2)
            lines_ax5.append(line)
            labels_ax5.append("Exc Rate (Hz)")
        except Exception:
            pass

    if RateMoninh and hasattr(RateMoninh, "t") and len(RateMoninh.t) > 0:
        try:
            from brian2 import Hz
            smooth_rate_inh = RateMoninh.smooth_rate(window="gaussian", width=10*ms) / Hz
            line, = ax5.plot(RateMoninh.t/ms, smooth_rate_inh, color=INH_COLOR, linewidth=1.2)
            lines_ax5.append(line)
            labels_ax5.append("Inh Rate (Hz)")
        except Exception:
            pass

    ax5.set_ylabel("Rate (Hz)")

    ax5b = ax5.twinx()
    total_duration_ms_plot_base = (initialsettletimeval + runtimeval) / ms
    max_mon_time_ms = total_duration_ms_plot_base

    if SpikeMonexc and hasattr(SpikeMonexc, "t") and len(SpikeMonexc.t) > 0:
        max_mon_time_ms = max(max_mon_time_ms, np.max(SpikeMonexc.t/ms))
    if RateMonexc and hasattr(RateMonexc, "t") and len(RateMonexc.t) > 0:
        max_mon_time_ms = max(max_mon_time_ms, np.max(RateMonexc.t/ms))

    ax1.set_xlim(0, max_mon_time_ms)

    time_points_plot_ms = np.linspace(0, max_mon_time_ms, 500)
    imid_signal_plot = np.zeros_like(time_points_plot_ms)
    settle_time_plot_ms = initialsettletimeval / ms
    imid_signal_plot[time_points_plot_ms >= settle_time_plot_ms] = currentImidnAval

    line, = ax5b.plot(time_points_plot_ms, imid_signal_plot, "g--", linewidth=1.5)
    lines_ax5.append(line)
    labels_ax5.append("Imid (nA)")

    ax5b.set_ylabel("Imid (nA)", color="g")
    ax5b.tick_params(axis="y", labelcolor="g")

    abs_imid = abs(currentImidnAval) + 1e-9
    min_imid = min(0, currentImidnAval * 1.1) - 0.05 * abs_imid
    max_imid = max(0.1, currentImidnAval * 1.1) + 0.05 * abs_imid
    ax5b.set_ylim(min_imid, max_imid)

    ax5.set_xlabel("Time (ms)")
    ax5.grid(True, linestyle=":", alpha=0.5)
    ax5.legend(lines_ax5, labels_ax5, loc="upper right")

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.suptitle(f"Network Activity {groupname}", y=0.99)

    _save_and_close(fig, f"{outputdir}/basic_activity_plot.png")


def plot_initial_raster(spikemonexc, spikemoninh, nexc, ninh, starttime, duration,
                        outputdir, groupname):
    """Plots a raster of the first few seconds of the main simulation run."""
    from brian2 import second

    fig, ax = plt.subplots(figsize=(12, 7))

    start_time_ms = starttime / second * 1000
    end_time_ms = start_time_ms + duration / second * 1000

    # Pre-compute time arrays once
    exc_times_ms = spikemonexc.t / second * 1000
    exc_mask = (exc_times_ms >= start_time_ms) & (exc_times_ms <= end_time_ms)
    ax.plot(exc_times_ms[exc_mask], spikemonexc.i[exc_mask], ".",
            color=EXC_COLOR, markersize=1.5, label="Excitatory")

    inh_times_ms = spikemoninh.t / second * 1000
    inh_mask = (inh_times_ms >= start_time_ms) & (inh_times_ms <= end_time_ms)
    ax.plot(inh_times_ms[inh_mask], spikemoninh.i[inh_mask] + nexc, ".",
            color=INH_COLOR, markersize=1.5, label="Inhibitory")

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Neuron Index")
    ax.set_title(f"Initial {duration/second}s Raster Plot - {groupname}")
    ax.set_xlim(start_time_ms, end_time_ms)
    ax.set_ylim(-10, nexc + ninh + 10)
    ax.axhline(y=nexc, color="black", linestyle="--", linewidth=1.0)
    ax.legend(markerscale=10, loc="upper right")
    ax.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    _save_and_close(fig, os.path.join(outputdir, "initial_5s_raster.png"))


def plot_detailed_stimulus_raster(spikemonexc, spikemoninh, trial_details, nexc, ninh,
                                   outputdir, groupname):
    """Plot detailed raster showing stimulus presentation trials with digit labels."""
    fig, ax = plt.subplots(figsize=(15, 8))

    exc_color_muted = "#b2df8a"
    inh_color_muted = "#a1c9f4"
    highlight_color = "crimson"

    # Pre-compute time arrays
    exc_times_ms = np.asarray(spikemonexc.t / ms)
    inh_times_ms = np.asarray(spikemoninh.t / ms)
    exc_indices = np.asarray(spikemonexc.i)
    inh_indices = np.asarray(spikemoninh.i)

    # Plot all spikes as background
    ax.plot(exc_times_ms, exc_indices, ".", color=exc_color_muted,
            markersize=0.7, label="Exc. Spike (Unstimulated)", zorder=1)
    ax.plot(inh_times_ms, inh_indices + nexc, ".", color=inh_color_muted,
            markersize=0.7, label="Inh. Spike (Unstimulated)", zorder=1)

    for trial in trial_details:
        stim_start = trial["stim_start_ms"]
        stim_end = trial["stim_end_ms"]
        trial_end = trial["trial_end_ms"]
        digit = trial["digit"]
        stimulated_exc = trial["stimulated_exc_indices"]
        stimulated_inh = trial["stimulated_inh_indices"]

        ax.axvspan(stim_start, stim_end, facecolor="mistyrose", alpha=0.7, edgecolor="none", zorder=0)
        ax.axvspan(stim_end, trial_end, facecolor="aliceblue", alpha=0.7, edgecolor="none", zorder=0)

        ax.text((stim_start + stim_end) / 2, ax.get_ylim()[1] * 1.02,
                f"Digit {digit}", ha="center", va="bottom", fontsize=12, fontweight="bold")

        # Highlight stimulated neuron spikes
        exc_mask = ((exc_times_ms >= stim_start) & (exc_times_ms <= stim_end) &
                    np.isin(exc_indices, stimulated_exc))
        ax.plot(exc_times_ms[exc_mask], exc_indices[exc_mask], ".",
                color=highlight_color, markersize=2.5, zorder=2)

        if stimulated_inh:
            inh_mask = ((inh_times_ms >= stim_start) & (inh_times_ms <= stim_end) &
                        np.isin(inh_indices, stimulated_inh))
            ax.plot(inh_times_ms[inh_mask], inh_indices[inh_mask] + nexc, ".",
                    color=highlight_color, markersize=2.5, zorder=2)

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Neuron Index")
    ax.set_title(f"Detailed Stimulus Presentation Raster: {groupname}")
    ax.grid(True, linestyle=":", alpha=0.6)

    if trial_details:
        ax.set_xlim(trial_details[0]["trial_start_ms"] - 50, trial_details[-1]["trial_end_ms"] + 50)
    ax.set_ylim(-10, nexc + ninh + 10)

    legend_elements = [
        Line2D([0], [0], marker=".", color=exc_color_muted, label="Exc. Spike", linestyle="None", markersize=8),
        Line2D([0], [0], marker=".", color=inh_color_muted, label="Inh. Spike", linestyle="None", markersize=8),
        Line2D([0], [0], marker=".", color=highlight_color, label="Stimulated Neuron Spike", linestyle="None", markersize=8),
        mpatches.Patch(color="mistyrose", alpha=0.7, label="Stimulus Period"),
        mpatches.Patch(color="aliceblue", alpha=0.7, label="Delay Period")
    ]
    ax.legend(handles=legend_elements, loc="upper right", frameon=True, facecolor="white", framealpha=1.0)
    ax.axhline(y=nexc, color="black", linestyle="--", linewidth=1.2)

    plt.tight_layout()
    _save_and_close(fig, os.path.join(outputdir, "detailed_stimulus_raster.png"))
    print(f"Saved detailed stimulus raster plot to {outputdir}")


def plot_neural_manifold(reservoir_states, labels, title="Neural Manifold of Digit Representations",
                        outputdir="."):
    """Generate 3D PCA visualization of reservoir states colored by digit class."""
    if reservoir_states.size == 0:
        print("Warning: Cannot plot manifold, reservoir states are empty.")
        return

    print("--- Generating Neural Manifold Plot ---")

    scaler = StandardScaler()
    scaled_states = scaler.fit_transform(reservoir_states)

    pca = PCA(n_components=3)
    states_3d = pca.fit_transform(scaled_states)

    explained_variance = pca.explained_variance_ratio_.sum()
    print(f"Variance explained by 3 PCs: {explained_variance:.2%}")

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    cmap = cm.get_cmap("tab10", 10)

    for i in range(10):
        mask = labels == i
        if mask.any():
            ax.scatter(states_3d[mask, 0], states_3d[mask, 1], states_3d[mask, 2],
                       color=cmap(i), label=str(i), s=60, alpha=0.8,
                       edgecolor="k", linewidth=0.5)

    ax.set_title(title, pad=20)
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_zlabel("PC 3")

    legend = ax.legend(title="Digits", markerscale=1.5, bbox_to_anchor=(1.1, 0.8))
    legend.get_frame().set_alpha(0.0)

    ax.grid(True)
    ax.view_init(elev=20., azim=-65)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor("w")

    output_path = os.path.join(outputdir, "neural_manifold_pca.png")
    _save_and_close(fig, output_path)
    print(f"Saved neural manifold plot to {output_path}")


def create_combined_plots(results_dict_keys_seconds, bin_widths_labels_us, groupname):
    """Create combined avalanche distribution plots for multiple bin widths."""
    outputdir = f"{OUTPUT_DIR_RUNS}/{groupname}/"
    os.makedirs(outputdir, exist_ok=True)

    valid_bw_labels_us = [
        bw for bw in bin_widths_labels_us
        if (bw * 1e-6) in results_dict_keys_seconds
        and results_dict_keys_seconds[bw * 1e-6] is not None
    ]

    if not valid_bw_labels_us:
        return

    cmap_func = plt.cm.viridis
    n = len(valid_bw_labels_us)
    colors = cmap_func(np.linspace(0, 1, n)) if n > 1 else [cmap_func(0.5)]

    # Size and duration distributions
    fig_dist, (ax_size, ax_dur) = plt.subplots(1, 2, figsize=(15, 6))

    for i, bw_us in enumerate(valid_bw_labels_us):
        res = results_dict_keys_seconds[bw_us * 1e-6]
        color = colors[i]
        label_suffix = f"{bw_us:.0f}us"

        if res["size_fit"] and hasattr(res["size_fit"], "data") and len(res["size_fit"].data) > 1:
            try:
                fit_label = label_suffix + (f" a={res['size_alpha']:.2f}" if not np.isnan(res['size_alpha']) else "")
                res["size_fit"].plot_ccdf(ax=ax_size, color=color, linewidth=1.5,
                                         marker="o", markersize=4, label=fit_label)
            except Exception:
                pass

        if res["duration_fit"] and hasattr(res["duration_fit"], "data") and len(res["duration_fit"].data) > 1:
            try:
                fit_label = label_suffix + (f" a={res['duration_alpha']:.2f}" if not np.isnan(res['duration_alpha']) else "")
                res["duration_fit"].plot_ccdf(ax=ax_dur, color=color, linewidth=1.5,
                                            marker="o", markersize=4, label=fit_label)
            except Exception:
                pass

    for ax, xlabel, ylabel, title in [
        (ax_size, "Avalanche Size (spikes)", "CCDF P(S>=s)", "Size Distribution"),
        (ax_dur, "Avalanche Duration (bins)", "CCDF P(T>=t)", "Duration Distribution"),
    ]:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xscale("log")
        ax.set_yscale("log")
        if valid_bw_labels_us:
            ax.legend(fontsize=9)
        ax.grid(True, which="both", linestyle="--", alpha=0.3)

    fig_dist.suptitle(f"Avalanche Distributions {groupname}", fontsize=16, y=1.02)
    fig_dist.tight_layout(rect=[0, 0, 1, 0.98])
    _save_and_close(fig_dist, f"{outputdir}/avalanche_distributions_combined.png")

    # Scaling relation
    fig_scaling, ax_scaling = plt.subplots(figsize=(8, 6))

    for i, bw_us in enumerate(valid_bw_labels_us):
        res = results_dict_keys_seconds[bw_us * 1e-6]
        color = colors[i]
        label_suffix = f"{bw_us:.0f}us"

        duration_groups = res.get("duration_groups")
        if duration_groups is not None and isinstance(duration_groups, pd.DataFrame) and not duration_groups.empty:
            if "duration" in duration_groups.columns and "size" in duration_groups.columns and len(duration_groups) > 1:
                try:
                    plot_data = duration_groups[
                        (duration_groups["duration"] > 0) & (duration_groups["size"] > 0)
                    ]
                    if not plot_data.empty:
                        gamma_str = f" g={res['gamma']:.2f}" if not np.isnan(res.get('gamma', np.nan)) else ""
                        ax_scaling.scatter(plot_data["duration"], plot_data["size"],
                                          color=color, alpha=0.6, s=40, label=label_suffix + gamma_str)

                        if not np.isnan(res.get('gamma', np.nan)) and not np.isnan(res.get('intercept', np.nan)):
                            log_durs = res.get("log_durations", np.array([]))
                            if len(log_durs) >= 2:
                                x_fit = np.linspace(log_durs.min(), log_durs.max(), 50)
                                y_fit = res["intercept"] + res["gamma"] * x_fit
                                ax_scaling.plot(10**x_fit, 10**y_fit, color=color, linewidth=2, linestyle="--")
                except Exception:
                    pass

    ax_scaling.set_xscale("log")
    ax_scaling.set_yscale("log")
    ax_scaling.set_xlabel("Duration (bins)")
    ax_scaling.set_ylabel("Size (spikes)")
    ax_scaling.set_title(f"Size-Duration Scaling {groupname}")
    if valid_bw_labels_us:
        ax_scaling.legend(fontsize=9)
    ax_scaling.grid(True, which="both", linestyle="--", alpha=0.3)

    fig_scaling.tight_layout()
    _save_and_close(fig_scaling, f"{outputdir}/avalanche_scaling_relation.png")


def create_individual_plots(results_dict_keys_seconds, bin_widths_labels_us, groupname):
    """Create individual avalanche distribution plots for each bin width."""
    outputdir = f"{OUTPUT_DIR_RUNS}/{groupname}/individual_avalanche_plots/"
    os.makedirs(outputdir, exist_ok=True)

    valid_bw_labels_us = [
        bw for bw in bin_widths_labels_us
        if (bw * 1e-6) in results_dict_keys_seconds
        and results_dict_keys_seconds[bw * 1e-6] is not None
    ]

    if not valid_bw_labels_us:
        return

    cmap_func = plt.cm.viridis
    n = len(valid_bw_labels_us)
    colors = cmap_func(np.linspace(0, 1, n)) if n > 1 else [cmap_func(0.5)]

    for i, bw_us in enumerate(valid_bw_labels_us):
        res = results_dict_keys_seconds[bw_us * 1e-6]
        color = colors[i]

        for dist_type in ["size", "duration"]:
            fit = res.get(f"{dist_type}_fit")
            alpha = res.get(f"{dist_type}_alpha", np.nan)

            if fit and hasattr(fit, "power_law") and fit.power_law:
                if hasattr(fit, "data") and len(fit.data) > 1:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    try:
                        fit.plot_ccdf(color=color, linewidth=1.5, marker="o", markersize=4, label="Empirical", ax=ax)
                        fit.power_law.plot_ccdf(color="black", linestyle="--", linewidth=2,
                                                label=f"Fit a={alpha:.2f}, xmin={fit.xmin:.1f}", ax=ax)
                        unit = "s" if dist_type == "size" else "t, bins"
                        ax.set_title(f"{dist_type.capitalize()} Distribution - {bw_us:.0f}us {groupname}")
                        ax.set_xlabel(f"Avalanche {dist_type.capitalize()} ({unit})")
                        ax.set_ylabel(f"CCDF")
                        ax.legend()
                        ax.grid(True, which="both", ls="--", alpha=0.3)
                        ax.set_xscale("log")
                        ax.set_yscale("log")
                        plt.tight_layout()
                        _save_and_close(fig, f"{outputdir}/{dist_type}_dist_{bw_us:.0f}us.png")
                    except Exception:
                        plt.close(fig)


def plot_all_learning_accuracy_curves(all_learning_data, imid_param_values, ei_ratio_param_values,
                                       condition_map, output_dir_base):
    """Plot comparative learning curves across conditions with mean and SEM."""
    os.makedirs(output_dir_base, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    colormap = {"subcritical": "blue", "critical": "green", "supercritical": "red"}
    plot_count = 0

    for i_idx, imid_val in enumerate(imid_param_values):
        for j_idx, ei_val in enumerate(ei_ratio_param_values):
            aggregated = {}
            for rep_idx in range(len(all_learning_data[i_idx][j_idx])):
                lc = all_learning_data[i_idx][j_idx][rep_idx]
                if lc:
                    for n_samples, acc in lc.items():
                        aggregated.setdefault(n_samples, []).append(acc)

            if aggregated:
                sorted_samples = sorted(aggregated.keys())
                means = np.array([np.mean(aggregated[s]) for s in sorted_samples])
                sems = np.array([np.std(aggregated[s], ddof=1) / np.sqrt(len(aggregated[s]))
                                 for s in sorted_samples])

                condition_name = condition_map.get(ei_val, f"EI_{ei_val:.3f}")
                color = colormap.get(condition_name, "black")

                ax.plot(sorted_samples, means, marker="o", linestyle="-",
                        color=color, label=condition_name.capitalize(), markersize=5)
                ax.fill_between(sorted_samples, means - sems, means + sems,
                                color=color, alpha=0.2)
                plot_count += 1

    if plot_count == 0:
        ax.text(0.5, 0.5, "No learning curve data to plot.", ha="center", va="center", transform=ax.transAxes)
    else:
        ax.set_xlabel("Number of Training Samples")
        ax.set_ylabel("Accuracy")
        ax.set_title("Classification Accuracy Across Conditions (Mean +/- SEM)")
        ax.set_ylim(0, 1.05)
        ax.legend(loc="best", title="Condition")
        ax.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()
    plot_filename = f"{output_dir_base}/comparative_learning_accuracy_curves.png"
    _save_and_close(fig, plot_filename)
    print(f"Saved comparative learning accuracy curves plot to {plot_filename}")


def plot_separate_aggregated_avalanche_ccdfs(all_avalanche_distributions, output_dir_base):
    """
    Plot aggregated avalanche size and duration CCDFs across conditions.

    Parameters
    ----------
    all_avalanche_distributions : list of dict
        Each dict has 'condition', 'repetition', 'sizes', 'durations'
    output_dir_base : str
        Output directory
    """
    os.makedirs(output_dir_base, exist_ok=True)

    if not all_avalanche_distributions:
        return

    # Group by condition
    conditions = {}
    for entry in all_avalanche_distributions:
        cond = entry['condition']
        if cond not in conditions:
            conditions[cond] = {'sizes': [], 'durations': []}
        conditions[cond]['sizes'].extend(entry['sizes'])
        conditions[cond]['durations'].extend(entry['durations'])

    colormap = {"subcritical": "blue", "critical": "green", "supercritical": "red"}

    # Size CCDF
    fig_size, ax_size = plt.subplots(figsize=(8, 6))
    for cond_name, data in conditions.items():
        sizes = np.array(data['sizes'])
        if len(sizes) > 0:
            sorted_sizes = np.sort(sizes)[::-1]
            ccdf = np.arange(1, len(sorted_sizes) + 1) / len(sorted_sizes)
            color = colormap.get(cond_name, "black")
            ax_size.loglog(sorted_sizes, ccdf, '.', color=color, alpha=0.5,
                          markersize=3, label=cond_name.capitalize())

    ax_size.set_xlabel("Avalanche Size (spikes)")
    ax_size.set_ylabel("CCDF P(S >= s)")
    ax_size.set_title("Aggregated Avalanche Size Distributions")
    ax_size.legend()
    ax_size.grid(True, which="both", linestyle="--", alpha=0.3)
    plt.tight_layout()
    _save_and_close(fig_size, os.path.join(output_dir_base, "aggregated_avalanche_ccdf_sizes.png"))

    # Duration CCDF
    fig_dur, ax_dur = plt.subplots(figsize=(8, 6))
    for cond_name, data in conditions.items():
        durations = np.array(data['durations'])
        if len(durations) > 0:
            sorted_durs = np.sort(durations)[::-1]
            ccdf = np.arange(1, len(sorted_durs) + 1) / len(sorted_durs)
            color = colormap.get(cond_name, "black")
            ax_dur.loglog(sorted_durs, ccdf, '.', color=color, alpha=0.5,
                         markersize=3, label=cond_name.capitalize())

    ax_dur.set_xlabel("Avalanche Duration (bins)")
    ax_dur.set_ylabel("CCDF P(T >= t)")
    ax_dur.set_title("Aggregated Avalanche Duration Distributions")
    ax_dur.legend()
    ax_dur.grid(True, which="both", linestyle="--", alpha=0.3)
    plt.tight_layout()
    _save_and_close(fig_dur, os.path.join(output_dir_base, "aggregated_avalanche_ccdf_durations.png"))

    print("Saved aggregated avalanche CCDF plots")


def generate_summary_plots(imid_vals, ei_ratio_vals, metrics_data, output_dir_base):
    """
    Generate summary plots (phase diagrams) for all metrics.
    Delegates to 1D or 2D plotting based on parameter sweep type.
    """
    os.makedirs(output_dir_base, exist_ok=True)

    n_imid = len(imid_vals)
    n_ei = len(ei_ratio_vals)

    if n_imid > 1 and n_ei > 1:
        print("--- Generating 2D summary plots (heatmaps)... ---")
        _plot_2d_heatmaps(imid_vals, ei_ratio_vals, metrics_data, output_dir_base)
    elif n_imid == 1 and n_ei > 1:
        print(f"--- Generating 1D summary plots vs. E/I Ratio (at Imid = {imid_vals[0]} nA)... ---")
        _plot_1d_graphs(xdata=ei_ratio_vals, xlabel="E/I Conductance Ratio (gE,max/gI,max)",
                        metrics_data=metrics_data, output_dir_base=output_dir_base,
                        param_str=f"Imid_{imid_vals[0]:.3f}nA")
    elif n_imid > 1 and n_ei == 1:
        print(f"--- Generating 1D summary plots vs. Imid (at E/I Ratio = {ei_ratio_vals[0]})... ---")
        _plot_1d_graphs(xdata=imid_vals, xlabel="Input Current Imid (nA)",
                        metrics_data=metrics_data, output_dir_base=output_dir_base,
                        param_str=f"EIRatio_{ei_ratio_vals[0]:.3f}")
    else:
        print("--- Skipping summary plot generation (not a 1D or 2D sweep). ---")


def _plot_1d_graphs(xdata, xlabel, metrics_data, output_dir_base, param_str=""):
    """Generate 1D line plots with error bars for each metric."""
    for key, (data, ylabel, error_data) in metrics_data.items():
        fig, ax = plt.subplots(figsize=(8, 6))
        plot_data = data.flatten()

        if np.all(np.isnan(plot_data)):
            ax.text(0.5, 0.5, f"No valid data for {key}", ha="center", va="center", transform=ax.transAxes)
        else:
            ax.errorbar(xdata, plot_data, yerr=error_data.flatten(),
                        marker="o", linestyle="None", color="royalblue",
                        capsize=5, ecolor="lightgray", elinewidth=3)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.grid(True, linestyle="--", alpha=0.6)

        ax.set_title(f"{ylabel} vs. {xlabel.split()[0]}")
        plt.tight_layout()

        safe_key = key.replace(" ", "_").replace("/", "_")
        _save_and_close(fig, f"{output_dir_base}/phase_diagram_{safe_key}_{param_str}.png")


def _plot_2d_heatmaps(imid_vals, ei_ratio_vals, metrics_data, output_dir_base):
    """Generate 2D heatmap phase diagrams for each metric."""
    X, Y = np.meshgrid(ei_ratio_vals, imid_vals)

    # Define all heatmap configurations
    heatmap_configs = [
        ("firing_rate", "viridis", None, None),
        ("cv", "coolwarm", mcolors.TwoSlopeNorm(vcenter=1.0, vmin=0.0, vmax=2.0), [1.0]),
        ("sigma", "coolwarm", "auto_sigma", [1.0]),
        ("rc_accuracy", "magma", None, None),
        ("samples_to_threshold", "viridis_r", None, None),
        ("accuracy_at_fixed_samples", "magma", None, None),
    ]

    for metric_key, cmap_name, norm, contour_levels in heatmap_configs:
        if metric_key not in metrics_data:
            continue

        data, label, _ = metrics_data[metric_key]
        fig, ax = plt.subplots(figsize=(8, 6))
        masked = np.ma.masked_invalid(data)

        if not masked.mask.all():
            actual_norm = norm
            if norm == "auto_sigma":
                actual_norm = mcolors.TwoSlopeNorm(
                    vcenter=1.0, vmin=np.nanmin(masked), vmax=np.nanmax(masked)
                )

            contour = ax.contourf(X, Y, masked, levels=50, cmap=cmap_name,
                                  norm=actual_norm, extend="both")
            fig.colorbar(contour, ax=ax, label=label)

            if contour_levels:
                for level in contour_levels:
                    if np.any(masked < level) and np.any(masked > level):
                        ax.contour(X, Y, masked, levels=[level], colors="black",
                                   linestyles="--", linewidths=1.5)
        else:
            ax.text(0.5, 0.5, f"No {metric_key} data", ha="center", va="center", transform=ax.transAxes)

        ax.set_xlabel("E/I Conductance Ratio (gE,max/gI,max)")
        ax.set_ylabel("Input Current Imid (nA)")
        ax.set_title(f"Phase Diagram: {label}")
        plt.tight_layout()

        safe_key = metric_key.replace(" ", "_")
        _save_and_close(fig, f"{output_dir_base}/phase_diagram_{safe_key}.png")
