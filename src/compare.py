import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

SAVE_RESULTS = False

SAMPLING_TIME = 0.02
HORIZON = 20


N_SAMPLES = 390
LABEL = 'iz0.8'


data = np.load('/home/elitedog/Dynamic_bic_mpc/gpnew/set/DYN-NMPC-{}.npz'.format(LABEL))
time_dyn = data['time'][:N_SAMPLES+1]
states_dyn = data['states'][:, :N_SAMPLES+1]
inputs_dyn = data['inputs'][:, :N_SAMPLES]

data = np.load('/home/elitedog/Dynamic_bic_mpc/gpnew/set/DYN-GPNMPC-{}.npz'.format(LABEL))
time_gp = data['time'][:N_SAMPLES+1]
states_gp = data['states'][:, :N_SAMPLES+1]
inputs_gp = data['inputs'][:, :N_SAMPLES+1]


plt.figure()
plt.axis('equal')
plt.plot(states_gp[0], states_gp[1], 'r', lw=1, label='MPC (GP correction)')
plt.plot(states_dyn[0], states_dyn[1], '--g', lw=1, label='MPC (ground truth)')
plt.xlabel('$x$ [m]')
plt.ylabel('$y$ [m]')
plt.legend(loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.15), frameon=False)


plt.show()
