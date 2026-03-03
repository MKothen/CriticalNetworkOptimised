"""
Reservoir Computing (RC) specific functions
Handles MNIST classification task using the spiking network as a reservoir
"""

import numpy as np
from brian2 import *
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score

from config import (
    RC_STATE_SMOOTHING_WIDTH_STD_DEV,
    RC_STATE_RATE_CALC_WINDOW_DURATION,
    FEED_INPUT_TO_INHIBITORY,
    PIXEL_BINARIZATION_THRESHOLD,
    BRIAN2_RUN_NAMESPACE,
)


def create_input_projection_map(n_input_pixels, n_exc_neurons, n_inh_neurons,
                                neurons_per_pixel=5, feed_to_inhibitory=True):
    """
    Create a random projection map from input pixels to neurons.
    Returns numpy arrays for fast vectorized indexing.

    Parameters
    ----------
    n_input_pixels : int
        Number of input channels (e.g., 784 for MNIST)
    n_exc_neurons : int
        Number of excitatory neurons
    n_inh_neurons : int
        Number of inhibitory neurons
    neurons_per_pixel : int
        How many neurons each pixel projects to
    feed_to_inhibitory : bool
        Whether to project to inhibitory neurons

    Returns
    -------
    dict
        'exc_targets': array of shape (n_input_pixels, neurons_per_pixel)
        'inh_targets': array of shape (n_input_pixels, neurons_per_pixel) or None
    """
    exc_targets = np.zeros((n_input_pixels, neurons_per_pixel), dtype=np.intp)
    inh_targets = None
    if feed_to_inhibitory:
        inh_targets = np.zeros((n_input_pixels, neurons_per_pixel), dtype=np.intp)

    # Interleave exc/inh generation per pixel (preserves original RNG sequence)
    for pixel_idx in range(n_input_pixels):
        exc_targets[pixel_idx] = np.random.choice(n_exc_neurons, neurons_per_pixel, replace=False)
        if feed_to_inhibitory:
            inh_targets[pixel_idx] = np.random.choice(n_inh_neurons, neurons_per_pixel, replace=False)

    return {'exc_targets': exc_targets, 'inh_targets': inh_targets}


def _precompute_gaussian_kernel(total_window_width_s, sim_dt_s, kernel_std_dev_s):
    """
    Pre-compute Gaussian smoothing kernel (called once, reused across all trials).

    Returns
    -------
    tuple
        (gaussian_kernel, kernel_half_pts, num_hist_bins_for_window)
    """
    kernel_half_pts = int(np.ceil(total_window_width_s / (2 * sim_dt_s)))
    if kernel_half_pts <= 0:
        return None, 0, 0

    kernel_time_rel = np.arange(-kernel_half_pts, kernel_half_pts + 1) * sim_dt_s
    gaussian_kernel = (1.0 / (kernel_std_dev_s * np.sqrt(2 * np.pi))) * \
                      np.exp(-kernel_time_rel**2 / (2 * kernel_std_dev_s**2))
    gaussian_kernel *= sim_dt_s

    return gaussian_kernel, kernel_half_pts, len(kernel_time_rel)


# Module-level kernel cache
_kernel_cache = {}


def _get_cached_kernel(total_window_width_s, sim_dt_s, kernel_std_dev_s):
    """Get or compute the Gaussian kernel (cached)."""
    key = (total_window_width_s, sim_dt_s, kernel_std_dev_s)
    if key not in _kernel_cache:
        _kernel_cache[key] = _precompute_gaussian_kernel(
            total_window_width_s, sim_dt_s, kernel_std_dev_s
        )
    return _kernel_cache[key]


def calculate_per_neuron_smoothed_rates(spike_monitor_t_arr, spike_monitor_i_arr,
                                        num_neurons, target_time_s, total_window_width_s,
                                        sim_dt_s, kernel_std_dev_s):
    """
    Calculate smoothed firing rates for all neurons at a specific time point.
    Optimized: pre-computes kernel once, uses vectorized spike grouping.

    Parameters
    ----------
    spike_monitor_t_arr : array
        Spike times in seconds
    spike_monitor_i_arr : array
        Neuron indices for each spike
    num_neurons : int
        Total number of neurons
    target_time_s : float
        Target time point for rate calculation
    total_window_width_s : float
        Total width of analysis window
    sim_dt_s : float
        Simulation timestep
    kernel_std_dev_s : float
        Standard deviation of Gaussian kernel

    Returns
    -------
    array
        Smoothed rates for each neuron at target time
    """
    rates = np.zeros(num_neurons)

    if sim_dt_s <= 0 or kernel_std_dev_s <= 0 or total_window_width_s <= 0:
        return rates
    if len(spike_monitor_t_arr) == 0:
        return rates

    # Get cached kernel
    gaussian_kernel, kernel_half_pts, _ = _get_cached_kernel(
        total_window_width_s, sim_dt_s, kernel_std_dev_s
    )
    if gaussian_kernel is None:
        return rates

    # Define histogram window
    hist_start_time_s = target_time_s - total_window_width_s / 2.0
    hist_end_time_s = target_time_s + total_window_width_s / 2.0 + sim_dt_s
    num_hist_bins = int(round((hist_end_time_s - hist_start_time_s) / sim_dt_s))

    if num_hist_bins <= 0:
        return rates

    hist_bin_edges = np.linspace(hist_start_time_s, hist_end_time_s, num_hist_bins + 1)

    # Filter to relevant time window (with padding for convolution)
    analysis_window_start = hist_start_time_s - total_window_width_s
    analysis_window_end = hist_end_time_s + total_window_width_s

    relevant_mask = (spike_monitor_t_arr >= analysis_window_start) & \
                    (spike_monitor_t_arr <= analysis_window_end)
    rel_times = spike_monitor_t_arr[relevant_mask]
    rel_indices = spike_monitor_i_arr[relevant_mask]

    if len(rel_times) == 0:
        return rates

    # Sort by neuron index for grouped processing
    sort_order = np.argsort(rel_indices)
    rel_times = rel_times[sort_order]
    rel_indices = rel_indices[sort_order]

    # Find neuron boundaries
    unique_neurons, neuron_starts, neuron_counts = np.unique(
        rel_indices, return_index=True, return_counts=True
    )

    center_bin_index = num_hist_bins // 2

    for idx in range(len(unique_neurons)):
        neuron_id = unique_neurons[idx]
        start = neuron_starts[idx]
        count = neuron_counts[idx]
        neuron_spike_times = rel_times[start:start + count]

        spike_counts, _ = np.histogram(neuron_spike_times, bins=hist_bin_edges)
        binned_rates_hz = spike_counts / sim_dt_s
        smoothed = np.convolve(binned_rates_hz, gaussian_kernel, mode='same')

        if 0 <= center_bin_index < len(smoothed):
            rates[neuron_id] = smoothed[center_bin_index]

    return rates


def _apply_stimulus_current(input_img_flat, projection_map, n_exc, n_inh,
                            input_current_amp):
    """
    Vectorized stimulus current application from input image to neuron populations.

    Parameters
    ----------
    input_img_flat : array
        Flattened input image (784 pixels)
    projection_map : dict
        Projection map with 'exc_targets' and 'inh_targets' arrays
    n_exc : int
        Number of excitatory neurons
    n_inh : int
        Number of inhibitory neurons
    input_current_amp : Quantity
        Input current amplitude

    Returns
    -------
    tuple
        (stim_current_exc, stim_current_inh) with Brian2 units
    """
    # Find active pixels (above threshold)
    active_pixels = np.where(input_img_flat > PIXEL_BINARIZATION_THRESHOLD)[0]

    stim_exc = np.zeros(n_exc)
    stim_inh = np.zeros(n_inh)

    amp_val = float(input_current_amp / nA)

    if len(active_pixels) > 0:
        # Get all target neurons for active pixels and accumulate
        exc_targets = projection_map['exc_targets'][active_pixels].ravel()
        np.add.at(stim_exc, exc_targets, amp_val)

        if FEED_INPUT_TO_INHIBITORY and projection_map['inh_targets'] is not None:
            inh_targets = projection_map['inh_targets'][active_pixels].ravel()
            np.add.at(stim_inh, inh_targets, amp_val)

    return stim_exc * nA, stim_inh * nA


def run_rc_simulation_for_input(network_sim_object, input_img_pattern_flat, projection_map,
                                pop_exc_group, pop_inh_group, n_exc_val, n_inh_val,
                                stim_duration, post_stim_total_duration,
                                mnist_input_current_amp, sim_dt_brian,
                                spike_mon_exc_obj, spike_mon_inh_obj,
                                trial_internal_settle_time, readout_snapshot_time_offset):
    """
    Run simulation for a single input pattern and extract reservoir state.

    Parameters
    ----------
    network_sim_object : Network
        Brian2 network object
    input_img_pattern_flat : array
        Flattened input image (784 pixels for MNIST)
    projection_map : dict
        Mapping from pixels to neurons (optimized format)
    pop_exc_group : NeuronGroup
        Excitatory neurons
    pop_inh_group : NeuronGroup
        Inhibitory neurons
    n_exc_val : int
        Number of excitatory neurons
    n_inh_val : int
        Number of inhibitory neurons
    stim_duration : Quantity
        Duration to present stimulus
    post_stim_total_duration : Quantity
        Duration after stimulus
    mnist_input_current_amp : Quantity
        Input current amplitude
    sim_dt_brian : Quantity
        Simulation timestep
    spike_mon_exc_obj : SpikeMonitor
        Excitatory spike monitor
    spike_mon_inh_obj : SpikeMonitor
        Inhibitory spike monitor
    trial_internal_settle_time : Quantity
        Settling time before stimulus
    readout_snapshot_time_offset : Quantity
        When to read out state after stimulus

    Returns
    -------
    array
        Reservoir state vector (concatenated firing rates)
    """
    ns = BRIAN2_RUN_NAMESPACE

    # Reset stimulus and settle
    pop_exc_group.I_stim = 0 * nA
    pop_inh_group.I_stim = 0 * nA

    if trial_internal_settle_time > 0 * ms:
        network_sim_object.run(trial_internal_settle_time, report=None, namespace=ns)

    time_before_stim = network_sim_object.t

    # Apply stimulus current (vectorized)
    stim_exc, stim_inh = _apply_stimulus_current(
        input_img_pattern_flat, projection_map,
        n_exc_val, n_inh_val, mnist_input_current_amp
    )

    pop_exc_group.I_stim = stim_exc
    pop_inh_group.I_stim = stim_inh
    network_sim_object.run(stim_duration, report=None, namespace=ns)

    # Turn off stimulus
    pop_exc_group.I_stim = 0 * nA
    pop_inh_group.I_stim = 0 * nA
    network_sim_object.run(post_stim_total_duration, report=None, namespace=ns)

    # Extract reservoir state at readout time
    readout_target_time_s = float(
        (time_before_stim + stim_duration + readout_snapshot_time_offset) / second
    )
    sim_dt_s = float(sim_dt_brian / second)
    window_s = float(RC_STATE_RATE_CALC_WINDOW_DURATION / second)
    kernel_std_s = float(RC_STATE_SMOOTHING_WIDTH_STD_DEV / second)

    # Get spike data (convert once)
    all_spike_times_exc = np.asarray(spike_mon_exc_obj.t / second)
    all_spike_indices_exc = np.asarray(spike_mon_exc_obj.i)
    all_spike_times_inh = np.asarray(spike_mon_inh_obj.t / second)
    all_spike_indices_inh = np.asarray(spike_mon_inh_obj.i)

    # Calculate rates at readout time
    rates_exc = calculate_per_neuron_smoothed_rates(
        all_spike_times_exc, all_spike_indices_exc, n_exc_val,
        readout_target_time_s, window_s, sim_dt_s, kernel_std_s
    )

    rates_inh = calculate_per_neuron_smoothed_rates(
        all_spike_times_inh, all_spike_indices_inh, n_inh_val,
        readout_target_time_s, window_s, sim_dt_s, kernel_std_s
    )

    return np.concatenate((rates_exc, rates_inh))


def train_readout_weights(reservoir_states_matrix_train, target_outputs_onehot_train,
                          ridge_alpha_val):
    """
    Train linear readout using Ridge regression.

    Parameters
    ----------
    reservoir_states_matrix_train : array
        Training reservoir states (n_samples x n_neurons)
    target_outputs_onehot_train : array
        One-hot encoded target labels
    ridge_alpha_val : float
        Ridge regularization parameter

    Returns
    -------
    array
        Trained readout weight matrix
    """
    n_samples = reservoir_states_matrix_train.shape[0]
    n_features = reservoir_states_matrix_train.shape[1]
    print(f"Training readout with {n_samples} samples, {n_features}D states.")

    # Add bias term
    X_train_with_bias = np.hstack([
        reservoir_states_matrix_train,
        np.ones((n_samples, 1))
    ])

    ridge_model = Ridge(alpha=ridge_alpha_val, fit_intercept=False)
    ridge_model.fit(X_train_with_bias, target_outputs_onehot_train)

    W_out = ridge_model.coef_.T
    print(f"Readout weights W_out shape: {W_out.shape}")

    return W_out


def evaluate_readout_performance(reservoir_states_matrix_test, trained_weights_W_out,
                                 original_labels_test):
    """
    Evaluate readout performance on test data.

    Parameters
    ----------
    reservoir_states_matrix_test : array
        Test reservoir states
    trained_weights_W_out : array
        Trained readout weights
    original_labels_test : array
        True labels for test data

    Returns
    -------
    tuple
        (accuracy, predicted_labels)
    """
    X_test_with_bias = np.hstack([
        reservoir_states_matrix_test,
        np.ones((reservoir_states_matrix_test.shape[0], 1))
    ])

    predicted_outputs = X_test_with_bias @ trained_weights_W_out
    predicted_labels = np.argmax(predicted_outputs, axis=1)
    accuracy = accuracy_score(original_labels_test, predicted_labels)

    return accuracy, predicted_labels
