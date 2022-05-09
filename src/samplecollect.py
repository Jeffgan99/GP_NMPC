import time
import numpy as np
from utils import systemdy
from params_mpc import gt
from dynmodel import Dynamic
from ftocp import FTOCPNLP


## Parameters initialization
N = 25
n_dy = 6
d = 2
x0_dy = np.array([0, 0, 0, 3, 0, 0])
dt = 0.12
sys_dy = systemdy(x0_dy, dt)
maxTime = 25
xRef_dy = np.array([10, 10, np.pi/2, 3, 0, 0])

R = 1*np.eye(d)
Q_dy = 1*np.eye(n_dy)
Qf_dy = np.diag([11.8, 2.0, 50.0, 280.0, 100.0, 1000.0])

bx_dy = np.array([15, 15, 15, 15, 15, 15])
bu = np.array([10, 0.5])

LABEL = 'iz08'
SAVE_RESULTS = True

params = gt()
model = Dynamic(**params)

states = np.zeros([n_dy, maxTime + 1])
dstates = np.zeros([n_dy, maxTime + 1])
inputs = np.zeros([d, maxTime + 1])
timearr = np.linspace(0, maxTime+1, maxTime+1) * dt

Ffy = np.zeros([maxTime + 1])
Frx = np.zeros([maxTime + 1])
Fry = np.zeros([maxTime + 1])

x_init = x0_dy
dstates[0, 0] = x_init[3]
states[:, 0] = x_init
print('Starting at ({:.1f},{:.1f})'.format(x_init[0], x_init[1]))

nlp_dy = FTOCPNLP(N, Q_dy, R, Qf_dy, xRef_dy, dt, bx_dy, bu)
ut_dy = nlp_dy.solve(x0_dy)
inputs[:, 0] = ut_dy

for t in range(0, maxTime):

    x0 = states[:, t]
    xt_dy = sys_dy.x[-1]

    start = time.time()
    ut_dy = nlp_dy.solve(xt_dy)
    end = time.time()
    inputs[:, t + 1] = ut_dy
    print("Iteration: {}, time to solve: {:.2f}".format(t, end - start))

    x_next, dxdt_next = model.sim_continuous(states[:, t], inputs[:, t].reshape(-1, 1), [0, dt])
    states[:, t + 1] = x_next[:, -1]
    dstates[:, t + 1] = dxdt_next[:, -1]
    Ffy[t + 1], Frx[t + 1], Fry[t + 1] = model.calc_forces(states[:, t], inputs[:, t])
    sys_dy.applyInput(ut_dy)


if SAVE_RESULTS:
    np.savez(
        '/home/elitedog/Dynamic_bic_mpc/gpnew/set/DYN-simu-{}.npz'.format(LABEL),
        time=timearr,
        states=states,
        dstates=dstates,
        inputs=inputs,
    )

