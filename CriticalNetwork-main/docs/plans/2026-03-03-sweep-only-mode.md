# Sweep-Only Mode Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a config-toggled sweep-only mode that runs dense Imid x E/I parameter sweeps without the RC task, extracting only intrinsic dynamics metrics (branching parameter, CV, firing rate, avalanche stats).

**Architecture:** A `SWEEP_ONLY_MODE` flag in `config.py` controls whether sweep-specific linspace parameters are used and whether the RC task, MNIST loading, and heavy monitors are skipped. All conditional logic lives in `main_simulation.py` with minimal branching.

**Tech Stack:** Brian2 (neural simulation), NumPy, pandas, matplotlib, powerlaw

---

### Task 1: Add sweep mode parameters to config.py

**Files:**
- Modify: `config.py:108-120` (after `EXC_FACTOR_FIXED`, before condition naming)

**Step 1: Add sweep mode config block**

Insert the following after line 109 (`EXC_FACTOR_FIXED = 1.0`) and before line 111 (`# Condition naming`):

```python
# ============================================================================
# SWEEP-ONLY MODE
# ============================================================================
# When True: skip RC task, use linspace parameters below, minimal monitors
# When False: standard RC mode with original parameters above
SWEEP_ONLY_MODE = False

# Sweep-specific parameter grid (used only when SWEEP_ONLY_MODE = True)
SWEEP_Imid_values_nA = np.linspace(0.1, 0.5, 5)
SWEEP_EI_ratio_values = np.linspace(0.001, 1.0, 15)
SWEEP_NUM_REPETITIONS = 1
```

**Step 2: Update condition naming to support sweep override**

Replace lines 111-120 (the condition naming block) with:

```python
# Apply sweep overrides if enabled
if SWEEP_ONLY_MODE:
    Imid_values_nA = SWEEP_Imid_values_nA
    EI_ratio_values = SWEEP_EI_ratio_values
    NUM_REPETITIONS = SWEEP_NUM_REPETITIONS

# Condition naming
sorted_ei_ratios = sorted(EI_ratio_values)
if len(sorted_ei_ratios) == 3 and not SWEEP_ONLY_MODE:
    condition_map = {
        sorted_ei_ratios[0]: 'subcritical',
        sorted_ei_ratios[1]: 'critical',
        sorted_ei_ratios[2]: 'supercritical'
    }
else:
    condition_map = {val: f"EI_{val:.3f}" for val in sorted_ei_ratios}
```

**Step 3: Verify config loads without errors**

Run: `python -c "from config import *; print(f'SWEEP_ONLY_MODE={SWEEP_ONLY_MODE}'); print(f'Imid={Imid_values_nA}'); print(f'EI={EI_ratio_values}')"`

Expected output:
```
SWEEP_ONLY_MODE=False
Imid=[0.3333]
EI=[0.001 0.385 1.  ]
```

**Step 4: Verify sweep mode override works**

Run: `python -c "import config; config.SWEEP_ONLY_MODE=True; exec(open('config.py').read()); print(f'Imid has {len(Imid_values_nA)} values'); print(f'EI has {len(EI_ratio_values)} values')"`

Note: This is a quick sanity check. The real test is that when `SWEEP_ONLY_MODE=True` in config.py, the linspace arrays are used.

**Step 5: Commit**

```bash
git add config.py
git commit -m "feat: add SWEEP_ONLY_MODE config with linspace parameter grid"
```

---

### Task 2: Conditionally skip MNIST loading and RC imports

**Files:**
- Modify: `main_simulation.py:13-34` (imports section)
- Modify: `main_simulation.py:78-91` (MNIST loading section)

**Step 1: Make RC and data_utils imports conditional**

Replace lines 13-34 with:

```python
# Import all our custom modules
from config import *
from network_model import create_network
from analysis import (
    calculate_cv,
    calculate_average_iei,
    calculate_live_cv,
    analyze_model_spikes
)
from plotting import *
from statistics import *

if not SWEEP_ONLY_MODE:
    from data_utils import (
        load_and_preprocess_mnist,
        calculate_samples_to_reach_threshold,
        get_accuracy_at_fixed_samples
    )
    from reservoir import (
        create_input_projection_map,
        run_rc_simulation_for_input,
        train_readout_weights,
        evaluate_readout_performance
    )
```

**Step 2: Update header print to show mode**

Replace lines 40-45 with:

```python
print("=" * 80)
mode_str = "SWEEP-ONLY (No RC)" if SWEEP_ONLY_MODE else "FULL (With RC)"
print(f"  NEURAL NETWORK SIMULATION - PARAMETER SWEEP [{mode_str}]")
print("=" * 80)
print(f"Configuration: {len(Imid_values_nA)} Imid values x {len(EI_ratio_values)} E/I ratios x {NUM_REPETITIONS} repetitions")
print(f"Total simulations: {len(Imid_values_nA) * len(EI_ratio_values) * NUM_REPETITIONS}")
print("=" * 80)
```

**Step 3: Wrap MNIST loading in conditional**

Replace lines 78-91 (the MNIST loading section) with:

```python
# ============================================================================
# LOAD MNIST DATA (RC mode only)
# ============================================================================

if not SWEEP_ONLY_MODE:
    print("\nLoading MNIST dataset...")

    X_train_global, y_train_onehot_global, y_train_labels_global, \
    X_test_global, y_test_onehot_global, y_test_labels_global = \
        load_and_preprocess_mnist(NUM_TRAIN_SAMPLES_MAX, NUM_TEST_SAMPLES_MAX, seed=SEED)

    NUM_TRAIN_SAMPLES_EFFECTIVE = len(X_train_global)
    NUM_TEST_SAMPLES_EFFECTIVE = len(X_test_global)

    print(f"  MNIST loaded: {NUM_TRAIN_SAMPLES_EFFECTIVE} train / {NUM_TEST_SAMPLES_EFFECTIVE} test samples")
else:
    print("\nSweep-only mode: Skipping MNIST loading")
```

**Step 4: Commit**

```bash
git add main_simulation.py
git commit -m "feat: conditionally skip MNIST loading and RC imports in sweep mode"
```

---

### Task 3: Conditionally skip RC data storage arrays

**Files:**
- Modify: `main_simulation.py:47-76` (data storage initialization section)

**Step 1: Make RC-specific arrays conditional**

Replace lines 47-76 with:

```python
# ============================================================================
# INITIALIZATION - Data Storage Arrays
# ============================================================================

print("\nInitializing data storage arrays...")

# Intrinsic dynamics arrays (always needed)
phase_diagram_results_firing_rate = np.full((len(Imid_values_nA), len(EI_ratio_values), NUM_REPETITIONS), np.nan)
phase_diagram_results_overall_cv = np.full((len(Imid_values_nA), len(EI_ratio_values), NUM_REPETITIONS), np.nan)
phase_diagram_results_sigma = np.full((len(Imid_values_nA), len(EI_ratio_values), NUM_REPETITIONS), np.nan)
phase_diagram_results_size_alpha = np.full((len(Imid_values_nA), len(EI_ratio_values), NUM_REPETITIONS), np.nan)
phase_diagram_results_duration_alpha = np.full((len(Imid_values_nA), len(EI_ratio_values), NUM_REPETITIONS), np.nan)
phase_diagram_results_gamma = np.full((len(Imid_values_nA), len(EI_ratio_values), NUM_REPETITIONS), np.nan)

# RC-specific arrays (only in full mode)
if not SWEEP_ONLY_MODE:
    phase_diagram_results_rc_accuracy = np.full((len(Imid_values_nA), len(EI_ratio_values), NUM_REPETITIONS), np.nan)
    phase_diagram_samples_to_threshold = np.full((len(Imid_values_nA), len(EI_ratio_values), NUM_REPETITIONS), np.nan)
    phase_diagram_accuracy_at_fixed_samples = np.full((len(Imid_values_nA), len(EI_ratio_values), NUM_REPETITIONS), np.nan)

    phase_diagram_learning_curves_data = [
        [[{} for _ in range(NUM_REPETITIONS)] for _ in range(len(EI_ratio_values))]
        for _ in range(len(Imid_values_nA))
    ]

# Storage for avalanche distributions
all_avalanche_distributions = []

# Storage for Excel export
all_runs_results_list = []

print("  Data structures initialized")
```

**Step 2: Commit**

```bash
git add main_simulation.py
git commit -m "feat: conditionally allocate RC data arrays only in full mode"
```

---

### Task 4: Optimize main loop — skip monitors and RC task

**Files:**
- Modify: `main_simulation.py:165-174` (monitors section)
- Modify: `main_simulation.py:176-195` (network object and simulation section)
- Modify: `main_simulation.py:262-398` (RC task section)

**Step 1: Make monitors conditional**

Replace lines 165-174 (the monitors section) with:

```python
                # ============================================================
                # MONITORS
                # ============================================================

                SpikeMon_exc = SpikeMonitor(Pop_exc)
                SpikeMon_inh = SpikeMonitor(Pop_inh)

                if not SWEEP_ONLY_MODE:
                    StateMon_exc = StateMonitor(Pop_exc, variables=['V', 'A'], record=0, dt=1*ms)
                    StateMon_inh = StateMonitor(Pop_inh, variables=['V', 'A'], record=0, dt=1*ms)
                    RateMon_exc = PopulationRateMonitor(Pop_exc)
                    RateMon_inh = PopulationRateMonitor(Pop_inh)
```

**Step 2: Update network object list**

Replace lines 176-195 (network object + simulation section) with:

```python
                # ============================================================
                # NETWORK OBJECT AND INTRINSIC DYNAMICS
                # ============================================================

                net_objects = [Pop_exc, Pop_inh, SpikeMon_exc, SpikeMon_inh] + all_synapses

                if not SWEEP_ONLY_MODE:
                    net_objects.extend([StateMon_exc, StateMon_inh, RateMon_exc, RateMon_inh])

                net = Network(net_objects)

                print(f"Running initial settling: {SIM_INITIAL_SETTLE_TIME/second:.1f} s")
                net.run(SIM_INITIAL_SETTLE_TIME, report='text', report_period=60*second)

                print(f"Running main simulation: {SIM_RUNTIME/second:.1f} s")
                net.run(SIM_RUNTIME, report='text', report_period=60*second)

                print("  Intrinsic dynamics simulation complete")
```

**Step 3: Wrap RC task in conditional**

Wrap lines 262-398 (the entire RC task section) with `if not SWEEP_ONLY_MODE:`. The block starts at the `# RESERVOIR COMPUTING TASK` comment and ends just before the `# PLOTTING` section:

```python
                # ============================================================
                # RESERVOIR COMPUTING TASK (RC mode only)
                # ============================================================

                rc_plot_info = None
                trial_details_for_plot = []

                if not SWEEP_ONLY_MODE:
                    print("Running reservoir computing task...")

                    # ... (all existing RC code from lines 268-398, indented one more level)

                else:
                    print("Sweep-only mode: Skipping RC task")
```

Important: All existing RC code (lines 268-398) goes inside the `if not SWEEP_ONLY_MODE:` block, indented one additional level. The `rc_plot_info = None` and `trial_details_for_plot = []` defaults are set before the conditional so the plotting section doesn't crash.

**Step 4: Commit**

```bash
git add main_simulation.py
git commit -m "feat: skip StateMonitor, RateMonitor, and RC task in sweep mode"
```

---

### Task 5: Update plotting section for sweep mode

**Files:**
- Modify: `main_simulation.py:400-440` (plotting section)

**Step 1: Simplify plotting in sweep mode**

Replace lines 400-440 with:

```python
                # ============================================================
                # PLOTTING (only first repetition)
                # ============================================================

                if rep_idx == 0:
                    print("Generating plots...")
                    output_dir_for_run = f"results_phase_diagram_runs/{param_combo_str}"
                    os.makedirs(output_dir_for_run, exist_ok=True)

                    if not SWEEP_ONLY_MODE:
                        # Full activity plot (needs StateMonitor, RateMonitor)
                        plot_basic_activity(
                            SpikeMon_exc, SpikeMon_inh,
                            StateMon_exc, StateMon_inh,
                            RateMon_exc, RateMon_inh,
                            N_exc, N_inh,
                            SIM_INITIAL_SETTLE_TIME, SIM_RUNTIME,
                            imid_val_nA, overall_mean_cv_val_exc,
                            CV_WINDOW_SIZE, CV_STEP_SIZE,
                            param_combo_str, rcaccuracyinfo=rc_plot_info
                        )

                    # Raster plot (only needs SpikeMonitors)
                    plot_initial_raster(
                        SpikeMon_exc, SpikeMon_inh, N_exc, N_inh,
                        starttime=SIM_INITIAL_SETTLE_TIME,
                        duration=5*second,
                        outputdir=output_dir_for_run,
                        groupname=param_combo_str
                    )

                    if not SWEEP_ONLY_MODE and trial_details_for_plot:
                        # Detailed stimulus raster (RC-specific)
                        plot_detailed_stimulus_raster(
                            SpikeMon_exc, SpikeMon_inh,
                            trial_details_for_plot,
                            N_exc, N_inh,
                            output_dir_for_run,
                            param_combo_str
                        )

                    print("  Plots saved")
```

**Step 2: Commit**

```bash
git add main_simulation.py
git commit -m "feat: simplify plotting in sweep mode to raster only"
```

---

### Task 6: Update result storage for sweep mode

**Files:**
- Modify: `main_simulation.py:442-460` (result storage section)

**Step 1: Store only intrinsic metrics in sweep mode**

Replace lines 442-460 with:

```python
                # ============================================================
                # STORE RESULTS
                # ============================================================

                run_data_dict = {
                    'Repetition': rep_idx,
                    'IMID (nA)': imid_val_nA,
                    'E/I Ratio': ei_ratio,
                    'Mean Firing Rate (Hz)': phase_diagram_results_firing_rate[i_idx, j_idx, rep_idx],
                    'Overall Mean CV': phase_diagram_results_overall_cv[i_idx, j_idx, rep_idx],
                    'Branching Parameter (sigma)': phase_diagram_results_sigma[i_idx, j_idx, rep_idx],
                    'Avalanche Size Alpha': phase_diagram_results_size_alpha[i_idx, j_idx, rep_idx],
                    'Avalanche Duration Alpha': phase_diagram_results_duration_alpha[i_idx, j_idx, rep_idx],
                    'Avalanche Gamma': phase_diagram_results_gamma[i_idx, j_idx, rep_idx],
                }

                if not SWEEP_ONLY_MODE:
                    run_data_dict['RC Accuracy (Best)'] = best_accuracy_this_run
                    run_data_dict['Samples to Threshold'] = np.nan
                    run_data_dict['RC Accuracy (Fixed Samples)'] = np.nan

                all_runs_results_list.append(run_data_dict)

                print(f"  Run {current_run_counter}/{total_runs} complete")
```

Note: `best_accuracy_this_run` is defined inside the RC block, so it's only available in full mode.

**Step 2: Commit**

```bash
git add main_simulation.py
git commit -m "feat: store only intrinsic metrics in sweep mode results"
```

---

### Task 7: Update post-processing for sweep mode

**Files:**
- Modify: `main_simulation.py:476-689` (entire post-processing section)

**Step 1: Wrap RC post-processing in conditionals**

Replace lines 476-512 (learning curve metrics) with:

```python
# ============================================================================
# POST-PROCESSING AND ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print(" " * 20 + "POST-PROCESSING RESULTS")
print("=" * 80)

if not SWEEP_ONLY_MODE:
    # Calculate learning curve metrics
    print("\nCalculating learning curve metrics...")

    for r_idx in range(len(Imid_values_nA)):
        for c_idx in range(len(EI_ratio_values)):
            for rep_idx in range(NUM_REPETITIONS):
                lc_data = phase_diagram_learning_curves_data[r_idx][c_idx][rep_idx]

                if lc_data and isinstance(lc_data, dict) and lc_data:
                    samples_to_thresh = calculate_samples_to_reach_threshold(
                        lc_data, QUICKNESS_TARGET_ACCURACY, READOUT_TRAINING_SUBSETS
                    )
                    phase_diagram_samples_to_threshold[r_idx, c_idx, rep_idx] = samples_to_thresh

                    acc_at_fixed = get_accuracy_at_fixed_samples(lc_data, QUICKNESS_FIXED_SAMPLE_SIZE)
                    phase_diagram_accuracy_at_fixed_samples[r_idx, c_idx, rep_idx] = acc_at_fixed

                    for run_dict in all_runs_results_list:
                        if (run_dict['IMID (nA)'] == Imid_values_nA[r_idx] and
                            run_dict['E/I Ratio'] == EI_ratio_values[c_idx] and
                            run_dict['Repetition'] == rep_idx):
                            run_dict['Samples to Threshold'] = samples_to_thresh
                            run_dict['RC Accuracy (Fixed Samples)'] = acc_at_fixed
                            break

    print("  Learning curve metrics calculated")
```

**Step 2: Update output directory and Excel export**

Replace lines 514-555 (Excel export section). Change the output directory based on mode:

```python
# ============================================================================
# SAVE RESULTS TO EXCEL
# ============================================================================

print("\nSaving results to Excel...")

output_directory = "results_sweep_summary" if SWEEP_ONLY_MODE else "results_phase_diagram_summary"
os.makedirs(output_directory, exist_ok=True)

if all_runs_results_list:
    output_path = os.path.join(output_directory, "simulation_summary.xlsx")
    df = pd.DataFrame(all_runs_results_list)

    # Add condition names
    df['Condition'] = df['E/I Ratio'].map(condition_map)

    # Define column order based on mode
    if SWEEP_ONLY_MODE:
        column_order = [
            'Repetition', 'IMID (nA)', 'E/I Ratio', 'Condition',
            'Mean Firing Rate (Hz)', 'Overall Mean CV',
            'Branching Parameter (sigma)',
            'Avalanche Size Alpha', 'Avalanche Duration Alpha', 'Avalanche Gamma'
        ]
    else:
        column_order = [
            'Repetition', 'IMID (nA)', 'E/I Ratio', 'Condition',
            'Mean Firing Rate (Hz)', 'Overall Mean CV',
            'Branching Parameter (sigma)', 'RC Accuracy (Best)',
            'RC Accuracy (Fixed Samples)', 'Samples to Threshold',
            'Avalanche Size Alpha', 'Avalanche Duration Alpha', 'Avalanche Gamma'
        ]

    existing_columns = [col for col in column_order if col in df.columns]
    df = df[existing_columns]

    df.sort_values(by=['IMID (nA)', 'E/I Ratio', 'Repetition'], inplace=True)

    try:
        df.to_excel(output_path, index=False, engine='openpyxl')
        print(f"  Results saved to {output_path}")
    except Exception as e:
        print(f"Warning: Could not save Excel file: {e}")
        csv_path = os.path.join(output_directory, "simulation_summary.csv")
        df.to_csv(csv_path, index=False)
        print(f"  Results saved to CSV: {csv_path}")
else:
    print("Warning: No results to save")
```

**Step 3: Update summary plots generation**

Replace lines 557-597 (summary plots section). Build the metrics dict based on mode:

```python
# ============================================================================
# GENERATE SUMMARY PLOTS
# ============================================================================

print("\nGenerating summary plots...")

summary_metrics = {
    'firing_rate': (
        np.nanmean(phase_diagram_results_firing_rate, axis=2),
        'Mean Firing Rate (Exc, Hz)',
        np.nanstd(phase_diagram_results_firing_rate, axis=2)
    ),
    'cv': (
        np.nanmean(phase_diagram_results_overall_cv, axis=2),
        'Overall Mean CV (Exc)',
        np.nanstd(phase_diagram_results_overall_cv, axis=2)
    ),
    'sigma': (
        np.nanmean(phase_diagram_results_sigma, axis=2),
        'Branching Parameter (sigma, bin=2xIEI)',
        np.nanstd(phase_diagram_results_sigma, axis=2)
    ),
}

if not SWEEP_ONLY_MODE:
    summary_metrics.update({
        'rc_accuracy': (
            np.nanmean(phase_diagram_results_rc_accuracy, axis=2),
            'MNIST Classification Accuracy',
            np.nanstd(phase_diagram_results_rc_accuracy, axis=2)
        ),
        'samples_to_threshold': (
            np.nanmean(phase_diagram_samples_to_threshold, axis=2),
            f'Samples to reach {QUICKNESS_TARGET_ACCURACY*100:.0f}% accuracy',
            np.nanstd(phase_diagram_samples_to_threshold, axis=2)
        ),
        'accuracy_at_fixed_samples': (
            np.nanmean(phase_diagram_accuracy_at_fixed_samples, axis=2),
            f'Accuracy with {QUICKNESS_FIXED_SAMPLE_SIZE} training samples',
            np.nanstd(phase_diagram_accuracy_at_fixed_samples, axis=2)
        ),
    })

generate_summary_plots(Imid_values_nA, EI_ratio_values, summary_metrics, output_directory)
print("  Phase diagrams generated")
```

**Step 4: Wrap RC-only post-processing sections in conditionals**

Replace lines 599-673 (learning curves, neural manifold, stats sections) with:

```python
if not SWEEP_ONLY_MODE:
    # ============================================================================
    # LEARNING CURVES
    # ============================================================================

    print("\nPlotting learning curves...")

    plot_all_learning_accuracy_curves(
        phase_diagram_learning_curves_data,
        Imid_values_nA,
        EI_ratio_values,
        condition_map,
        output_directory
    )
    print("  Learning curves plotted")

    # ============================================================================
    # NEURAL MANIFOLD
    # ============================================================================

    print("\nPlotting neural manifold...")

    if 'X_test_states_collected' in locals() and X_test_states_collected.size > 0:
        last_condition = condition_map.get(EI_ratio_values[-1], f"EI_{EI_ratio_values[-1]:.3f}")
        plot_neural_manifold(
            X_test_states_collected,
            y_test_labels_global,
            title=f"Neural Manifold of Test Set ({last_condition.capitalize()})",
            outputdir=output_directory
        )
        print("  Neural manifold plotted")
    else:
        print("  Skipped: No test states available")

# ============================================================================
# AGGREGATED AVALANCHE PLOTS (always run)
# ============================================================================

print("\nPlotting aggregated avalanche distributions...")

if all_avalanche_distributions:
    from plotting import plot_separate_aggregated_avalanche_ccdfs
    plot_separate_aggregated_avalanche_ccdfs(all_avalanche_distributions, output_directory)
    print("  Avalanche distributions plotted")
else:
    print("  Skipped: No avalanche data available")

# ============================================================================
# STATISTICAL TESTS
# ============================================================================

if not SWEEP_ONLY_MODE:
    print("\n" + "=" * 80)
    print(" " * 20 + "STATISTICAL ANALYSIS")
    print("=" * 80)

    statistical_results_to_test = {
        "Final Accuracy": phase_diagram_results_rc_accuracy,
        "Accuracy at Fixed Samples": phase_diagram_accuracy_at_fixed_samples,
        "Samples to Threshold": phase_diagram_samples_to_threshold
    }

    run_and_print_statistical_tests(
        statistical_results_to_test,
        EI_ratio_values,
        Imid_values_nA,
        NUM_REPETITIONS,
        condition_map
    )

    run_learning_curve_statistics(
        phase_diagram_learning_curves_data,
        Imid_values_nA,
        EI_ratio_values,
        condition_map
    )
```

**Step 5: Update completion message**

Replace lines 675-689 with:

```python
# ============================================================================
# COMPLETION
# ============================================================================

print("\n" + "=" * 80)
print(" " * 25 + "SIMULATION COMPLETE!")
print("=" * 80)
print(f"\nResults saved to: {output_directory}/")
print("  - simulation_summary.xlsx")
print("  - phase_diagram_*.png")
if not SWEEP_ONLY_MODE:
    print("  - comparative_learning_accuracy_curves.png")
    print("  - neural_manifold_pca.png")
print("  - aggregated_avalanche_ccdf_*.png")
print(f"\nIndividual run results in: results_phase_diagram_runs/")
print("=" * 80)
```

**Step 6: Commit**

```bash
git add main_simulation.py
git commit -m "feat: complete sweep-only mode post-processing and output"
```

---

### Task 8: Smoke test both modes

**Step 1: Verify sweep-only mode starts and prints correct header**

Temporarily set `SWEEP_ONLY_MODE = True` in config.py. Run:

```bash
python -c "from config import *; print(f'Mode: SWEEP_ONLY={SWEEP_ONLY_MODE}'); print(f'Grid: {len(Imid_values_nA)}x{len(EI_ratio_values)}x{NUM_REPETITIONS}={len(Imid_values_nA)*len(EI_ratio_values)*NUM_REPETITIONS} conditions')"
```

Expected:
```
Mode: SWEEP_ONLY=True
Grid: 5x15x1=75 conditions
```

**Step 2: Verify full mode is unchanged**

Set `SWEEP_ONLY_MODE = False` in config.py. Run same command.

Expected:
```
Mode: SWEEP_ONLY=False
Grid: 1x3x8=24 conditions
```

**Step 3: Set final default**

Ensure `SWEEP_ONLY_MODE = False` is the default in config.py (preserving existing behavior).

**Step 4: Commit**

```bash
git add config.py
git commit -m "chore: verify both modes, keep SWEEP_ONLY_MODE=False as default"
```

---

## Summary of Changes

| File | Lines Changed | What |
|------|--------------|------|
| `config.py` | +15 lines | Sweep mode flag + linspace parameters + updated condition naming |
| `main_simulation.py` | ~50 lines modified | Conditional imports, MNIST loading, monitors, RC task, plotting, storage, post-processing |

## Execution Notes

- Tasks 1-7 are sequential (each builds on the previous)
- Task 8 is validation
- No new files created
- No new dependencies
- Existing RC mode behavior is fully preserved when `SWEEP_ONLY_MODE = False`
