"""
Neural network model definitions
Contains Brian2 equations and network setup functions
"""

from brian2 import *
import numpy as np
from config import *

# Pre-build equation strings once at module load (avoids repeated string operations)
_NEURON_EQS = '''
dV/dt = 1/C_mem * (I_leak + I_exp - (I_syn_exc + I_syn_inh) + I_noise - A + I_stim) : volt (unless refractory)
dA/dt = 1/tau_A*(g_A*(V-V_L)-A) : ampere
dI_noise/dt = -(I_noise-Imid_val_eq)/tau_noise + sigma_noise * sqrt(2/tau_noise) * xi : ampere
I_stim : ampere
Imid_val_eq : ampere
s_tot_exc : siemens
s_tot_inh : siemens
V_thresh_val_eq : volt (constant)
I_syn_exc = s_tot_exc * (V - V_syn_exc) : ampere
I_syn_inh = s_tot_inh * (V - V_syn_inh) : ampere
I_leak = -g_mem_val_eq*(V-V_L) : ampere
I_exp = g_mem_val_eq*D_T*exp((V-V_T_val_eq)/D_T) : ampere
g_mem_val_eq : siemens (constant)
b_val_eq : ampere (constant)
tau_A : second (constant)
V_T_val_eq : volt (constant)
'''

_NEURON_RESET = 'V = Vr; A += b_val_eq'

# Pre-build synapse equation templates
_SYN_DYNAMICS_TEMPLATE = '''
ds_syn_{tag}/dt = x_syn_{tag} : 1 (clock-driven)
dx_syn_{tag}/dt = 1/(tau_r_syn*tau_d_syn) *( -(tau_r_syn+tau_d_syn)*x_syn_{tag} - s_syn_{tag} ) : Hz (clock-driven)
peak_conductance : siemens
'''

_SYN_EXC_MODEL = _SYN_DYNAMICS_TEMPLATE.format(tag='exc') + \
    's_tot_exc_post = peak_conductance * s_syn_exc : siemens (summed)'
_SYN_INH_MODEL = _SYN_DYNAMICS_TEMPLATE.format(tag='inh') + \
    's_tot_inh_post = peak_conductance * s_syn_inh : siemens (summed)'

_SYN_EXC_ONPRE = 'x_syn_exc += 1*Hz'
_SYN_INH_ONPRE = 'x_syn_inh += 1*Hz'


def get_neuron_equations():
    """
    Returns the AdEx (Adaptive Exponential Integrate-and-Fire) neuron equations.

    Returns
    -------
    tuple
        (equations_string, reset_string)
    """
    return _NEURON_EQS, _NEURON_RESET


def get_synapse_equations():
    """
    Returns the synaptic dynamics equations.

    Returns
    -------
    tuple
        (exc_model, inh_model, exc_onpre, inh_onpre)
    """
    return _SYN_EXC_MODEL, _SYN_INH_MODEL, _SYN_EXC_ONPRE, _SYN_INH_ONPRE


def initialize_neuron_population(pop, current_Imid):
    """
    Initialize a neuron population with heterogeneous parameters.

    Parameters
    ----------
    pop : NeuronGroup
        The Brian2 neuron group to initialize
    current_Imid : Quantity
        The mid-point current value (with Brian2 units)
    """
    N = len(pop)

    # Generate heterogeneous parameters (same RNG call order as original)
    pop.V_thresh_val_eq = Vt_mean + np.random.randn(N) * Vt_std
    pop.g_mem_val_eq = np.maximum(0.1*nS, g_mem_mean + np.random.randn(N) * g_mem_std)
    pop.b_val_eq = np.maximum(0*nA, b_mean + np.random.randn(N) * b_std)
    pop.tau_A = np.maximum(50*ms, tau_A_mean + np.random.randn(N) * tau_A_std)

    pop.V_T_val_eq = V_T_val
    pop.A = 0.0 * nA
    pop.Imid_val_eq = current_Imid
    pop.I_noise = current_Imid
    pop.I_stim = 0 * nA
    pop.V = V_L + np.random.randn(N) * 5 * mV
    pop.s_tot_exc = 0 * siemens
    pop.s_tot_inh = 0 * siemens


def create_network(N_exc, N_inh, current_Imid, exc_factor, inh_factor, connection_prob=0.1):
    """
    Create a balanced excitatory-inhibitory network.

    Parameters
    ----------
    N_exc : int
        Number of excitatory neurons
    N_inh : int
        Number of inhibitory neurons
    current_Imid : Quantity
        Background input current
    exc_factor : float
        Scaling factor for excitatory synapses
    inh_factor : float
        Scaling factor for inhibitory synapses
    connection_prob : float
        Connection probability

    Returns
    -------
    dict
        Dictionary containing all network components
    """
    eqs_neurons, eqs_reset = get_neuron_equations()
    model_exc, model_inh, onpre_exc, onpre_inh = get_synapse_equations()

    # Create neuron populations
    Pop_exc = NeuronGroup(
        N_exc, eqs_neurons,
        threshold='V>V_thresh_val_eq',
        reset=eqs_reset,
        method='euler',
        dt=set_dt
    )

    Pop_inh = NeuronGroup(
        N_inh, eqs_neurons,
        threshold='V>V_thresh_val_eq',
        reset=eqs_reset,
        method='euler',
        dt=set_dt
    )

    # Initialize both populations
    for pop in (Pop_exc, Pop_inh):
        initialize_neuron_population(pop, current_Imid)

    # Create synapses - use shared conductance values
    exc_conductance = exc_factor * base_g_syn_max_exc_value * siemens
    inh_conductance = inh_factor * base_g_syn_max_inh_value * siemens

    synapse_configs = [
        (Pop_exc, Pop_exc, model_exc, onpre_exc, exc_conductance),
        (Pop_exc, Pop_inh, model_exc, onpre_exc, exc_conductance),
        (Pop_inh, Pop_exc, model_inh, onpre_inh, inh_conductance),
        (Pop_inh, Pop_inh, model_inh, onpre_inh, inh_conductance),
    ]

    all_synapses = []
    for source, target, model, onpre, conductance in synapse_configs:
        syn = Synapses(source, target, model=model, on_pre=onpre, dt=set_dt)
        syn.connect(p=connection_prob)
        syn.peak_conductance = conductance
        all_synapses.append(syn)

    return {
        'Pop_exc': Pop_exc,
        'Pop_inh': Pop_inh,
        'Syn_exc_to_exc': all_synapses[0],
        'Syn_exc_to_inh': all_synapses[1],
        'Syn_inh_to_exc': all_synapses[2],
        'Syn_inh_to_inh': all_synapses[3],
        'synapses': all_synapses
    }
