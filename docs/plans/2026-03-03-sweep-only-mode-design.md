# Sweep-Only Mode Design

## Problem

Running a parameter sweep currently always activates the reservoir computing (RC) task, which accounts for ~50% of total runtime (~200-300s per condition). For exploring how metrics like branching parameter vary across Imid and E/I ratio space, the RC task is unnecessary. Additionally, the current parameter arrays (`np.array([0.3333])` and `np.array([0.001, 0.385, 1.0])`) are designed for RC evaluation at specific conditions, not for dense parameter exploration.

## Solution

Add a `SWEEP_ONLY_MODE` config flag that:
1. Overrides parameter arrays with `np.linspace` ranges for dense sweeping
2. Skips the RC task entirely
3. Removes unnecessary monitors (StateMonitor, PopulationRateMonitor)
4. Outputs only intrinsic dynamics metrics

## Files Modified

- `config.py` — New sweep mode parameters
- `main_simulation.py` — Conditional logic for sweep mode

## Config Changes (`config.py`)

New section after experimental design parameters:

```python
SWEEP_ONLY_MODE = False

SWEEP_Imid_values_nA = np.linspace(0.1, 0.5, 5)
SWEEP_EI_ratio_values = np.linspace(0.001, 1.0, 15)
SWEEP_NUM_REPETITIONS = 1
```

## `main_simulation.py` Changes

### Parameter override (after imports)
When `SWEEP_ONLY_MODE=True`, override `Imid_values_nA`, `EI_ratio_values`, `NUM_REPETITIONS` with sweep versions. Regenerate `condition_map` for arbitrary E/I values.

### Skip MNIST loading
Wrap MNIST loading block in `if not SWEEP_ONLY_MODE`.

### Skip RC data arrays
Don't allocate RC-specific storage arrays in sweep mode.

### Main loop per condition
- **Network creation**: Same
- **Monitors**: SpikeMonitor only (skip StateMonitor, PopulationRateMonitor)
- **Intrinsic dynamics**: Same (20.3s simulation)
- **Basic analysis**: Same (firing rate, CV)
- **Avalanche analysis**: Same (branching parameter, size/duration alpha, gamma)
- **RC task**: Skipped entirely
- **Plotting**: Raster only for first repetition
- **Result storage**: 7 intrinsic metrics only

### Post-processing
- Skip learning curve calculation
- Skip RC-related statistical tests
- Skip neural manifold plot
- Skip learning curve plots
- Only include intrinsic metrics in summary plots and Excel
- Output to `results_sweep_summary/`

## Metrics in Sweep Mode

| Metric | Source |
|--------|--------|
| Mean Firing Rate (Hz) | Spike count / (N_exc * runtime) |
| Overall Mean CV | ISI coefficient of variation |
| Branching Parameter (sigma) | Binned activity ratio |
| Avalanche Size Alpha | Power-law exponent |
| Avalanche Duration Alpha | Power-law exponent |
| Gamma | Size-duration scaling exponent |

## Performance Estimate

Default sweep grid: 5 Imid x 15 EI x 1 rep = 75 conditions
- Per condition: ~25s sim + ~5s analysis = ~30s
- Total: ~37 minutes
- Compared to RC mode (1x3x8 = 24 conditions at ~5min each): ~2 hours
