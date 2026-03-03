"""
Simple example: Running a basic network simulation
This demonstrates the core functionality without the full parameter sweep
"""
import brian2
from brian2 import *
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import (
    SEED, N_TOTAL_NEURONS, N_EXC, N_INH,
    base_g_syn_max_exc_value, base_g_syn_max_inh_value,
    SIM_INITIAL_SETTLE_TIME, SIM_RUNTIME, ANALYSIS_DELAY_AFTER_SETTLE,
    THESIS_STYLE, BRIAN2_RUN_NAMESPACE,
)
from network_model import create_network
from analysis import calculate_cv, calculate_average_iei

# Set random seed
np.random.seed(SEED)
brian2.seed(SEED)

plt.rcParams.update(THESIS_STYLE)

print("=" * 60)
print("SIMPLE NETWORK SIMULATION EXAMPLE")
print("=" * 60)

# ============================================================================
# STEP 1: Define network parameters
# ============================================================================
print("\nStep 1: Defining network parameters...")

current_Imid = 0.2 * nA
ei_ratio = 0.385  # Critical condition

exc_factor = 1.0
inh_factor = (base_g_syn_max_exc_value * exc_factor) / (base_g_syn_max_inh_value * ei_ratio)

print(f"  Excitatory neurons: {N_EXC}")
print(f"  Inhibitory neurons: {N_INH}")
print(f"  Background current: {current_Imid/nA:.3f} nA")
print(f"  E/I ratio: {ei_ratio}")

# ============================================================================
# STEP 2: Create the network
# ============================================================================
print("\nStep 2: Creating network...")

network_components = create_network(N_EXC, N_INH, current_Imid, exc_factor, inh_factor)

Pop_exc = network_components['Pop_exc']
Pop_inh = network_components['Pop_inh']

print("  Network created successfully")

# ============================================================================
# STEP 3: Add monitors
# ============================================================================
print("\nStep 3: Setting up monitors...")

SpikeMon_exc = SpikeMonitor(Pop_exc)
SpikeMon_inh = SpikeMonitor(Pop_inh)
StateMon_exc = StateMonitor(Pop_exc, ['V', 'A'], record=[0])
RateMon_exc = PopulationRateMonitor(Pop_exc)

print("  Monitors added")

# ============================================================================
# STEP 4: Create network object and run simulation
# ============================================================================
print("\nStep 4: Running simulation...")
print(f"  Initial settling: {SIM_INITIAL_SETTLE_TIME/second:.1f} s")
print(f"  Main simulation: {SIM_RUNTIME/second:.1f} s")

net = Network(collect())
net.run(SIM_INITIAL_SETTLE_TIME, report='text', namespace=BRIAN2_RUN_NAMESPACE)
net.run(SIM_RUNTIME/3, report='text', namespace=BRIAN2_RUN_NAMESPACE)
Pop_exc.I_stim[1:150] = 50*nA
net.run(15*ms, namespace=BRIAN2_RUN_NAMESPACE)
Pop_exc.I_stim[1:150] = 0*nA
net.run(SIM_RUNTIME/2, namespace=BRIAN2_RUN_NAMESPACE)

print("  Simulation completed")

# ============================================================================
# STEP 5: Basic analysis
# ============================================================================
print("\nStep 5: Analyzing results...")

total_spikes_exc = len(SpikeMon_exc.t)
mean_firing_rate = total_spikes_exc / (N_EXC * SIM_RUNTIME / second)

analysis_start = SIM_INITIAL_SETTLE_TIME + ANALYSIS_DELAY_AFTER_SETTLE
cv_value = calculate_cv(SpikeMon_exc, N_EXC, start_time=analysis_start)
iei_value = calculate_average_iei(SpikeMon_exc, analysis_start)

print(f"  Mean firing rate: {mean_firing_rate:.2f} Hz")
print(f"  Coefficient of variation: {cv_value:.3f}")
print(f"  Average IEI: {iei_value*1000:.2f} ms" if iei_value else "  Average IEI: N/A")

# ============================================================================
# STEP 6: Create a simple plot
# ============================================================================
print("\nStep 6: Creating visualization...")

fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# Raster plot
axes[0].plot(SpikeMon_exc.t/ms, SpikeMon_exc.i, '.', color='crimson', markersize=0.5, label='Excitatory')
axes[0].plot(SpikeMon_inh.t/ms, SpikeMon_inh.i + N_EXC, '.', color='royalblue', markersize=0.5, label='Inhibitory')
axes[0].set_ylabel('Neuron Index')
axes[0].set_title('Network Raster Plot')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Voltage trace
axes[1].plot(StateMon_exc.t/ms, StateMon_exc.V[0]/mV, color='crimson', linewidth=0.8)
axes[1].set_ylabel('Membrane Potential (mV)')
axes[1].set_title('Example Neuron Voltage Trace')
axes[1].grid(True, alpha=0.3)

# Population rate
smooth_rate = RateMon_exc.smooth_rate(window='gaussian', width=10*ms)/Hz
axes[2].plot(RateMon_exc.t/ms, smooth_rate, color='crimson', linewidth=1.2)
axes[2].set_xlabel('Time (ms)')
axes[2].set_ylabel('Population Rate (Hz)')
axes[2].set_title('Excitatory Population Activity')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('simple_example_output.png', dpi=300)
plt.close(fig)
print("  Plot saved as 'simple_example_output.png'")

print("\n" + "=" * 60)
print("SIMULATION COMPLETE!")
print("=" * 60)
