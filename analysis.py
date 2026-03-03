"""
Analysis functions for neural activity
Includes firing rate, CV, avalanche, and criticality measures
"""

import numpy as np
from scipy import stats
import powerlaw as pl
from brian2 import second


def calculate_average_iei(spike_monitor, analysis_start_time):
    """
    Calculate average inter-event interval from spike monitor.

    Parameters
    ----------
    spike_monitor : SpikeMonitor
        Brian2 spike monitor
    analysis_start_time : Quantity
        Time to start analysis (with Brian2 units)

    Returns
    -------
    float or None
        Average IEI in seconds, or None if insufficient data
    """
    if not hasattr(spike_monitor, 't') or len(spike_monitor.t) == 0:
        return None

    all_times = np.asarray(spike_monitor.t / second)
    start_s = float(analysis_start_time / second)
    relevant = all_times[all_times >= start_s]

    if len(relevant) < 2:
        return None

    relevant.sort()
    ieis = np.diff(relevant)

    if len(ieis) == 0:
        return None

    mean_iei = ieis.mean()
    return mean_iei if mean_iei > 1e-6 else None


def calculate_cv(spike_monitor, num_neurons, start_time=0*second):
    """
    Calculate coefficient of variation (CV) of inter-spike intervals.
    Vectorized: groups spikes by neuron index using np.unique for O(N log N) performance.

    Parameters
    ----------
    spike_monitor : SpikeMonitor
        Brian2 spike monitor
    num_neurons : int
        Total number of neurons
    start_time : Quantity
        Time to start analysis

    Returns
    -------
    float
        Mean CV across all neurons
    """
    if not hasattr(spike_monitor, 't') or len(spike_monitor.t) == 0:
        return np.nan

    spike_times_s = np.asarray(spike_monitor.t / second)
    spike_indices = np.asarray(spike_monitor.i)
    start_s = float(start_time / second)

    # Filter by start time
    mask = spike_times_s >= start_s
    spike_times_s = spike_times_s[mask]
    spike_indices = spike_indices[mask]

    if len(spike_times_s) == 0:
        return np.nan

    # Sort by neuron index, then by time within each neuron
    sort_order = np.lexsort((spike_times_s, spike_indices))
    spike_times_s = spike_times_s[sort_order]
    spike_indices = spike_indices[sort_order]

    # Find boundaries between neurons using np.unique
    unique_neurons, neuron_starts, neuron_counts = np.unique(
        spike_indices, return_index=True, return_counts=True
    )

    all_cvs = []
    for idx, (start, count) in enumerate(zip(neuron_starts, neuron_counts)):
        if count < 2:
            continue
        neuron_times = spike_times_s[start:start + count]
        isis = np.diff(neuron_times)
        if len(isis) == 0:
            continue
        mean_isi = isis.mean()
        if mean_isi <= 1e-12:
            continue
        cv = isis.std() / mean_isi
        if np.isfinite(cv):
            all_cvs.append(cv)

    return np.nanmean(all_cvs) if all_cvs else np.nan


def calculate_live_cv(spike_monitor, num_neurons, analysis_start_time,
                      analysis_duration, window_size, step_size,
                      min_spikes_per_neuron_window=3):
    """
    Calculate CV over sliding windows for temporal dynamics.
    Optimized: pre-sorts spikes once, uses binary search for window extraction.

    Parameters
    ----------
    spike_monitor : SpikeMonitor
        Brian2 spike monitor
    num_neurons : int
        Number of neurons
    analysis_start_time : Quantity
        Start time for analysis
    analysis_duration : Quantity
        Duration of analysis period
    window_size : Quantity
        Size of sliding window
    step_size : Quantity
        Step size for sliding window
    min_spikes_per_neuron_window : int
        Minimum spikes required per neuron in window

    Returns
    -------
    tuple
        (window_centers_ms, mean_cv_values)
    """
    if not hasattr(spike_monitor, 't') or len(spike_monitor.t) == 0:
        return np.array([]), np.array([])

    spike_times_s = np.asarray(spike_monitor.t / second)
    spike_indices = np.asarray(spike_monitor.i)

    analysis_start_s = float(analysis_start_time / second)
    analysis_end_s = analysis_start_s + float(analysis_duration / second)
    window_size_s = float(window_size / second)
    step_size_s = float(step_size / second)

    if analysis_end_s < analysis_start_s + window_size_s:
        return np.array([]), np.array([])

    window_starts = np.arange(
        analysis_start_s,
        analysis_end_s - window_size_s + step_size_s,
        step_size_s
    )

    if len(window_starts) == 0 and analysis_end_s >= analysis_start_s + window_size_s:
        window_starts = np.array([analysis_start_s])

    # Pre-sort spikes by neuron, then by time
    sort_order = np.lexsort((spike_times_s, spike_indices))
    sorted_times = spike_times_s[sort_order]
    sorted_indices = spike_indices[sort_order]

    # Build per-neuron spike time arrays once
    unique_neurons, neuron_starts, neuron_counts = np.unique(
        sorted_indices, return_index=True, return_counts=True
    )
    neuron_time_slices = {}
    for neuron_id, start, count in zip(unique_neurons, neuron_starts, neuron_counts):
        neuron_time_slices[neuron_id] = sorted_times[start:start + count]

    mean_cv_values = np.empty(len(window_starts))
    window_centers_s = np.empty(len(window_starts))

    for w_idx, win_start_s in enumerate(window_starts):
        win_end_s = win_start_s + window_size_s
        window_centers_s[w_idx] = win_start_s + window_size_s / 2.0

        cvs_in_window = []
        for neuron_id, times in neuron_time_slices.items():
            # Binary search for window bounds
            left = np.searchsorted(times, win_start_s, side='left')
            right = np.searchsorted(times, win_end_s, side='left')
            n_spikes = right - left

            if n_spikes >= min_spikes_per_neuron_window:
                window_times = times[left:right]
                isis = np.diff(window_times)
                if len(isis) > 0:
                    mean_isi = isis.mean()
                    if mean_isi > 1e-12:
                        cv = isis.std() / mean_isi
                        if np.isfinite(cv):
                            cvs_in_window.append(cv)

        mean_cv_values[w_idx] = np.nanmean(cvs_in_window) if cvs_in_window else np.nan

    return window_centers_s * 1000, mean_cv_values


def calculate_branching_parameter(binned_activity):
    """
    Calculate branching parameter (sigma) from binned activity.
    Vectorized: uses array operations instead of Python loop.

    Parameters
    ----------
    binned_activity : array
        Array of activity counts in each time bin

    Returns
    -------
    float
        Branching parameter (sigma)
    """
    if not isinstance(binned_activity, np.ndarray) or len(binned_activity) < 2:
        return np.nan

    ancestors = binned_activity[:-1]
    descendants = binned_activity[1:]

    # Only consider bins where ancestor > 0
    valid = ancestors > 0
    if not np.any(valid):
        return np.nan

    ratios = descendants[valid].astype(np.float64) / ancestors[valid].astype(np.float64)
    finite_ratios = ratios[np.isfinite(ratios)]

    return finite_ratios.mean() if len(finite_ratios) > 0 else np.nan


def _detect_avalanches(hist, min_duration=2):
    """
    Detect avalanches as contiguous periods of non-zero activity.

    Parameters
    ----------
    hist : array
        Binned spike counts
    min_duration : int
        Minimum number of bins for a valid avalanche

    Returns
    -------
    list of list
        Each inner list contains spike counts per bin for one avalanche
    """
    if hist.size == 0:
        return []

    # Find transitions: active/inactive boundaries
    active = hist > 0
    # Pad with False to detect boundaries at edges
    padded = np.concatenate(([False], active, [False]))
    diffs = np.diff(padded.astype(np.int8))

    # Starts where diff == 1, ends where diff == -1
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]

    avalanches = []
    for s, e in zip(starts, ends):
        duration = e - s
        if duration >= min_duration:
            avalanches.append(hist[s:e].tolist())

    return avalanches


def analyze_bin_width(timestamps, bin_width_seconds, max_time_seconds):
    """
    Analyze avalanche statistics for a given bin width.

    Parameters
    ----------
    timestamps : array
        Array of spike times in seconds
    bin_width_seconds : float
        Width of time bins in seconds
    max_time_seconds : float
        Maximum time for analysis

    Returns
    -------
    dict
        Dictionary containing avalanche statistics
    """
    empty_result = {
        'num_avalanches': 0,
        'mean_size': np.nan,
        'mean_duration': np.nan,
        'size_alpha': np.nan,
        'duration_alpha': np.nan,
        'gamma': np.nan,
        'intercept': np.nan,
        'r_value': np.nan,
        'p_value': np.nan,
        'std_err': np.nan,
        'size_fit': None,
        'duration_fit': None,
        'duration_groups': None,
        'log_durations': np.array([]),
        'log_sizes': np.array([]),
        'avalanches': [],
        'branching_parameter': np.nan
    }

    try:
        if bin_width_seconds <= 0 or max_time_seconds < 0:
            return empty_result

        if len(timestamps) == 0 or not np.all(np.isfinite(timestamps)):
            return empty_result

        # Build histogram
        actual_max = max(max_time_seconds, timestamps.max())
        bins = np.arange(0, actual_max + bin_width_seconds, bin_width_seconds)

        if bins.size < 2:
            return empty_result

        hist, _ = np.histogram(timestamps, bins=bins)

        # Calculate branching parameter from full histogram
        branching_param_val = calculate_branching_parameter(hist)

        # Detect avalanches
        avalanches = _detect_avalanches(hist)

        if not avalanches:
            empty_result['branching_parameter'] = branching_param_val
            return empty_result

        # Calculate avalanche properties (vectorized)
        sizes = np.array([sum(av) for av in avalanches])
        durations = np.array([len(av) for av in avalanches])

        # Fit power laws
        size_alpha, duration_alpha = np.nan, np.nan
        size_fit, duration_fit = None, None

        if len(sizes) > 10:
            valid_sizes = sizes[sizes > 0]
            if len(valid_sizes) > 10:
                try:
                    size_fit = pl.Fit(valid_sizes, discrete=True, xmin=None, verbose=False)
                    if size_fit and hasattr(size_fit, 'power_law') and size_fit.power_law:
                        size_alpha = size_fit.power_law.alpha
                except Exception:
                    size_fit = None

        if len(durations) > 10:
            valid_durations = durations[durations > 0]
            if len(valid_durations) > 10:
                try:
                    duration_fit = pl.Fit(valid_durations, discrete=True, xmin=None, verbose=False)
                    if duration_fit and hasattr(duration_fit, 'power_law') and duration_fit.power_law:
                        duration_alpha = duration_fit.power_law.alpha
                except Exception:
                    duration_fit = None

        # Calculate scaling relationship (gamma)
        slope, intercept, r_value, p_value, std_err = np.nan, np.nan, np.nan, np.nan, np.nan
        log_durations, log_sizes = np.array([]), np.array([])
        duration_groups = None

        # Group by duration and compute mean size per duration
        unique_durs, dur_inverse = np.unique(durations, return_inverse=True)
        if len(unique_durs) >= 2:
            mean_sizes_per_dur = np.zeros(len(unique_durs))
            for k, _ in enumerate(unique_durs):
                mean_sizes_per_dur[k] = sizes[dur_inverse == k].mean()

            valid_mask = (unique_durs > 0) & (mean_sizes_per_dur > 0)
            if valid_mask.sum() >= 2:
                log_durations = np.log10(unique_durs[valid_mask].astype(float))
                log_sizes = np.log10(mean_sizes_per_dur[valid_mask])

                if not (np.isnan(log_durations).any() or np.isnan(log_sizes).any()):
                    slope, intercept, r_value, p_value, std_err = stats.linregress(
                        log_durations, log_sizes
                    )

            # Build duration_groups for plotting compatibility
            import pandas as pd
            duration_groups = pd.DataFrame({
                'duration': unique_durs,
                'size': mean_sizes_per_dur
            })

        return {
            'num_avalanches': len(avalanches),
            'mean_size': sizes.mean(),
            'mean_duration': durations.mean(),
            'size_alpha': size_alpha,
            'duration_alpha': duration_alpha,
            'gamma': slope,
            'intercept': intercept,
            'r_value': r_value,
            'p_value': p_value,
            'std_err': std_err,
            'size_fit': size_fit,
            'duration_fit': duration_fit,
            'duration_groups': duration_groups,
            'log_durations': log_durations,
            'log_sizes': log_sizes,
            'avalanches': avalanches,
            'branching_parameter': branching_param_val
        }

    except Exception as e:
        print(f"Error analyzing bin_width={bin_width_seconds}: {str(e)}")
        return empty_result


def analyze_model_spikes(spike_times_seconds, bin_widths_to_analyze_seconds, group_name="Model"):
    """
    Analyze spikes from model simulation for avalanche statistics.

    Parameters
    ----------
    spike_times_seconds : array
        Array of spike times in seconds
    bin_widths_to_analyze_seconds : list
        List of bin widths to test
    group_name : str
        Name for this analysis group

    Returns
    -------
    dict
        Dictionary mapping bin widths to analysis results
    """
    spike_times_seconds = np.sort(spike_times_seconds)

    if len(spike_times_seconds) == 0:
        return {bw_s: analyze_bin_width(np.array([]), bw_s, 0)
                for bw_s in bin_widths_to_analyze_seconds if bw_s > 0}

    max_spike_time_s = spike_times_seconds[-1]

    # Deduplicate and sort bin widths
    unique_bws = sorted(set(bw for bw in bin_widths_to_analyze_seconds if bw > 0))

    return {bw_s: analyze_bin_width(spike_times_seconds, bw_s, max_spike_time_s)
            for bw_s in unique_bws}
