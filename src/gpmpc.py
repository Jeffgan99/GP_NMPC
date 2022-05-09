import time
import numpy as np
import _pickle as pickle
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pickle
from utils import systemdy
from gpftocp import FTOCPNLP
from params_mpc import gt
from dynmodel import Dynamic
from gps import GPModel


SAVE_RESULTS = True
LABEL = 'iz0.8'

N = 25
n = 6
d = 2
x0_dy = np.array([0, 0, 0, 3, 0, 0])
dt = 0.12
sys_dy = systemdy(x0_dy, dt)
maxTime = 25
xRef_dy = np.array([10, 10, np.pi/2, 3, 0, 0])

R = 1*np.eye(d)
Q_dy = 1*np.eye(n)
Qf_dy = np.diag([11.8, 2.0, 50.0, 280.0, 100.0, 1000.0])

bx_dy = np.array([15, 15, 15, 15, 15, 15])
bu = np.array([10, 0.5])
dt = 0.12
HORIZON = 25
Q = np.diag([1, 1])
Qf = np.diag([0, 0])

with open('/home/elitedog/Dynamic_bic_mpc/gpnew/set/vxgpiz08.pickle', 'rb') as f:
    (vxmodel, vxxscaler, vxyscaler) = pickle.load(f)
vxgp = GPModel('vx', vxmodel, vxyscaler)
with open('/home/elitedog/Dynamic_bic_mpc/gpnew/set/vygpiz08.pickle', 'rb') as f:
    (vymodel, vyxscaler, vyyscaler) = pickle.load(f)
vygp = GPModel('vy', vymodel, vyyscaler)
with open('/home/elitedog/Dynamic_bic_mpc/gpnew/set/omegagpiz08.pickle', 'rb') as f:
    (omegamodel, omegaxscaler, omegayscaler) = pickle.load(f)
omegagp = GPModel('omega', omegamodel, omegayscaler)
gpmodels = {
    'vx': vxgp,
    'vy': vygp,
    'omega': omegagp,
    'xscaler': vxxscaler,
    'yscaler': vxyscaler,
}

params = gt()
model = Dynamic(**params)

states = np.zeros([n, maxTime + 1])
dstates = np.zeros([n, maxTime + 1])
inputs = np.zeros([d, maxTime + 1])
timearr = np.linspace(0, maxTime+1, maxTime + 1) * dt
Ffy = np.zeros([maxTime + 1])
Frx = np.zeros([maxTime + 1])
Fry = np.zeros([maxTime + 1])
hstates = np.zeros([n, N + 1])
hstates2 = np.zeros([n, N + 1])


nlp_dy = FTOCPNLP(N, Q_dy, R, Qf_dy, xRef_dy, dt, bx_dy, bu, gpmodels)
ut_dy = nlp_dy.solve(x0_dy)

x_init = x0_dy
dstates[0, 0] = x_init[3]
states[:, 0] = x_init
inputs[:, 0] = ut_dy
print('Starting at ({:.1f},{:.1f})'.format(x_init[0], x_init[1]))

xPredNLP_dy = []
uPredNLP_dy = []
CostSolved_dy = []

for t in range(0, maxTime):

    uprev = inputs[:, t - 1]
    x0 = states[:, t]
    xt_dy = sys_dy.x[-1]

    start = time.time()
    ut_dy = nlp_dy.solve(xt_dy)
    end = time.time()
    umpc = nlp_dy.uPred
    xmpc = nlp_dy.xPred
    inputs[:, t + 1] = umpc[0, :]
    print("Iteration: {}, time to solve: {:.2f}".format(t, end - start))

    xPredNLP_dy.append(nlp_dy.xPred)
    uPredNLP_dy.append(nlp_dy.uPred)
    CostSolved_dy.append(nlp_dy.qcost)

    x_next, dxdt_next = model.sim_continuous(states[:, t], inputs[:, t].reshape(-1, 1), [0, dt])
    states[:, t + 1] = x_next[:, -1]
    dstates[:, t + 1] = dxdt_next[:, -1]
    Ffy[t + 1], Frx[t + 1], Fry[t + 1] = model.calc_forces(states[:, t], inputs[:, t])
    sys_dy.applyInput(ut_dy)

    hstates[:, 0] = x0[:n]
    hstates2[:, 0] = x0[:n]
    for j in range(N):
        x_next, dxdt_next = model.sim_continuous(hstates[:n, j], umpc[j, :].reshape(-1, 1), [0, dt])
        hstates[:, j + 1] = x_next[:, -1]
        hstates2[:, j + 1] = xmpc[j + 1, :]

x_cl_nlp_dy = np.array(sys_dy.x)
u_cl_nlp_dy = np.array(sys_dy.u)

if SAVE_RESULTS:
    np.savez(
        '/home/elitedog/Dynamic_bic_mpc/gpnew/set/DYN-GPNMPC-{}.npz'.format(LABEL),
        time=timearr,
        states=states,
        dstates=dstates,
        inputs=inputs,
    )

arr = np.array(xPredNLP_dy)
arr_2 = arr.reshape(650, 6)
arr_1 = np.array(sys_dy.x)
arr_3 = np.zeros(26)
for i in range(26):
    arr_3[i] = arr_2[i, 3]
plt.figure()
time = np.linspace(0, 25, 26)
for t in range(0, maxTime):
    if t == 0:
        plt.plot(xPredNLP_dy[t][:, 0], '--.b', label='NLP-predicted x position')
    else:
        time_1 = np.linspace(t, 25, 26-t)
        time_1 = time_1.tolist()
        xPredNLP_dy[t][t, 0] = arr_1[t, 0]
        plt.plot(time_1, xPredNLP_dy[t][t:26, 0], '--.b')

plt.plot(time, arr_1[:, 0], '-*r', label="Close-loop simulated x position")
plt.xlabel('Time')
plt.ylabel('X-Position')
plt.legend()
plt.show()

for timeToPlot in [0, 10]:
    plt.figure()
    plt.plot(xPredNLP_dy[timeToPlot][:,0], xPredNLP_dy[timeToPlot][:,1], '--.b', label="Simulated trajectory using NLP-aided MPC at time $t = $"+str(timeToPlot))
    plt.plot(xPredNLP_dy[timeToPlot][0,0], xPredNLP_dy[timeToPlot][0,1], 'ok', label="$x_t$ at time $t = $"+str(timeToPlot))
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.xlim(-1,15)
    plt.ylim(-1,15)
    plt.legend()
    plt.show()

plt.figure()
for t in range(0, maxTime):
    if t == 0:
        plt.plot(xPredNLP_dy[t][:,0], xPredNLP_dy[t][:,1], '--.b', label='Simulated trajectory using NLP-aided MPC')
    else:
        plt.plot(xPredNLP_dy[t][:,0], xPredNLP_dy[t][:,1], '--.b')
plt.plot(x_cl_nlp_dy[:,0], x_cl_nlp_dy[:,1], '-*r', label="Closed-loop trajectory")
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.xlim(-1,15)
plt.ylim(-1,15)
plt.legend()
plt.show()

plt.figure()
plt.plot(xPredNLP_dy[0][:,0], xPredNLP_dy[0][:,1], '--.b', label='Simulated trajectory using NLP-aided MPC')
plt.plot(x_cl_nlp_dy[:,0], x_cl_nlp_dy[:,1], '-*r', label="Closed-loop trajectory")
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.xlim(-1,15)
plt.ylim(-1,15)
plt.legend()
plt.show()

plt.figure()
plt.plot(u_cl_nlp_dy[:,0], '-*r', label="Closed-loop input: Acceleration")
plt.plot(uPredNLP_dy[0][:,0], '-ob', label="Simulated input: Acceleration")
plt.xlabel('Time')
plt.ylabel('Acceleration')
plt.legend()
plt.show()

plt.figure()
plt.plot(u_cl_nlp_dy[:,1], '-*r', label="Closed-loop input: Steering")
plt.plot(uPredNLP_dy[0][:,1], '-ob', label="Simulated input: Steering")
plt.xlabel('Time')
plt.ylabel('Steering')
plt.legend()
plt.show()

plt.figure()
plt.plot(xPredNLP_dy[0][:,0], xPredNLP_dy[0][:,1], '-*r', label='Solution from the NLP')
plt.title('Simulated trajectory')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.xlim(-1,15)
plt.ylim(-1,15)
plt.legend()
plt.show()

plt.figure()
plt.plot(xPredNLP_dy[0][:,3], '-*r', label='NLP performance')
plt.plot(x_cl_nlp_dy[:,3], 'ok', label='Closed-loop performance')
plt.xlabel('Time')
plt.ylabel('Velocity of the x-axis')
plt.legend()
plt.show()


plt.figure()
plt.plot(CostSolved_dy, '-ob')
plt.xlabel('Time')
plt.ylabel('Iteration cost')
plt.legend()
plt.show()
