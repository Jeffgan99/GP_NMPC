import numpy as np
import casadi as cs
import matplotlib.pyplot as plt


def odeintRK6(fun, y0, t, args=()):
    gamma = np.asarray([16 / 135, 0, 6656 / 12825, 28561 / 56430, -9 / 50, 2 / 55])
    y_next = np.zeros([len(t) - 1, len(y0)])

    for i in range(len(t) - 1):
        h = t[i + 1] - t[i]
        k1 = h * fun(t[i], y0, *args)
        k2 = h * fun(t[i] + h / 4, y0 + k1 / 4, *args)
        k3 = h * fun(t[i] + 3 / 8 * h, y0 + 3 / 32 * k1 + 9 / 32 * k2, *args)
        k4 = h * fun(t[i] + 12 / 13 * h, y0 + 1932 / 2197 * k1 - 7200 / 2197 * k2 + 7296 / 2197 * k3, *args)
        k5 = h * fun(t[i] + h, y0 + 439 / 216 * k1 - 8 * k2 + 3680 / 513 * k3 - 845 / 4104 * k4, *args)
        k6 = h * fun(t[i] + h / 2, y0 - 8 / 27 * k1 + 2 * k2 - 3544 / 2565 * k3 + 1859 / 4104 * k4 - 11 / 40 * k5,
                     *args)
        K = np.asarray([k1, k2, k3, k4, k5, k6])
        y_next[i, :] = y0 + gamma @ K
        y0 = y0 + gamma @ K
    return y_next


class Model:
    def __init__(self):
        pass

    def _integrate(self, x_t, u_t, t_start, t_end):
        fun = self._diffequation
        odesol = odeintRK6(
            fun=fun,
            y0=x_t,
            t=[t_start, t_end],
            args=(u_t,))
        return odesol[-1, :]

    def plot_results(self, t, x, dxdt, u, friction_circle=False):

        plt.figure()
        plt.plot(x[0, :], x[1, :])
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plt.axis('equal')
        plt.grid(True)

        plt.figure()
        plt.plot(t, x[0, :], label='x')
        plt.plot(t, x[1, :], label='y')
        plt.xlabel('time [s]')
        plt.ylabel('position [m]')
        plt.grid(True)
        plt.legend()

        if dxdt is not None:
            plt.figure()
            plt.plot(t, dxdt[0, :], label='speed x')
            plt.plot(t, dxdt[1, :], label='speed y')
            if not friction_circle:
                plt.plot(t, dxdt[2, :], label='yaw rate')
            plt.plot(t, np.sqrt(dxdt[0, :] ** 2 + dxdt[1, :] ** 2), '--', label='speed abs')
            plt.xlabel('time [s]')
            plt.ylabel('velocity [m/s]')
            plt.grid(True)
            plt.legend()

            plt.figure()
            plt.plot(dxdt[0, :], dxdt[1, :])
            plt.xlabel('speed x [m/s]')
            plt.ylabel('speed y [m/s]')
            plt.axis('equal')
            plt.grid(True)

        plt.figure()
        if friction_circle:
            plt.plot(t, np.arctan2(dxdt[1, :], dxdt[0, :]))
        else:
            plt.plot(t, x[2, :])
        plt.ylabel('yaw (heading) [rad]')
        plt.xlabel('time [s]')
        plt.grid(True)

        # plot inputs
        plt.figure()
        if friction_circle:
            plt.plot(t[1:], u[0, :], label='force x')
            plt.plot(t[1:], u[1, :], label='force y')
        else:
            plt.plot(t[1:], u[0, :], label='acceleration')
            plt.plot(t[1:], u[1, :], label='steering')
        plt.ylabel('inputs')
        plt.grid(True)
        plt.legend()
        plt.show()


class Dynamic(Model):

    def __init__(self, lf, lr, mass, Iz, Cf, Cr, Bf, Br, Df, Dr, Cm1, Cm2, Cr0, Cr2, **kwargs):
        self.lf = lf
        self.lr = lr
        self.dr = lr / (lf + lr)
        self.mass = mass
        self.Iz = Iz

        self.Cf = Cf
        self.Cr = Cr

        self.Bf = Bf
        self.Br = Br
        self.Df = Df
        self.Dr = Dr

        self.Cm1 = Cm1
        self.Cm2 = Cm2
        self.Cr0 = Cr0
        self.Cr2 = Cr2

        self.n_states = 6
        self.n_inputs = 2
        Model.__init__(self)

    def sim_continuous(self, x0, u, t):
        n_steps = u.shape[1]
        x = np.zeros([6, n_steps + 1])
        dxdt = np.zeros([6, n_steps + 1])
        dxdt[:, 0] = self._diffequation(None, x0, [0, 0])
        x[:, 0] = x0
        for ids in range(1, n_steps + 1):
            x[:, ids] = self._integrate(x[:, ids - 1], u[:, ids - 1], t[ids - 1], t[ids])
            dxdt[:, ids] = self._diffequation(None, x[:, ids], u[:, ids - 1])

        return x, dxdt

    def _diffequation(self, t, x, u):

        steer = u[1]
        psi = x[2]
        vx = x[3]
        vy = x[4]
        omega = x[5]

        Ffy, Frx, Fry = self.calc_forces(x, u)

        dxdt = np.zeros(6)
        dxdt[0] = vx * np.cos(psi) - vy * np.sin(psi)
        dxdt[1] = vx * np.sin(psi) + vy * np.cos(psi)
        dxdt[2] = omega
        dxdt[3] = 1 / self.mass * (Frx - Ffy * np.sin(steer)) + vy * omega
        dxdt[4] = 1 / self.mass * (Fry + Ffy * np.cos(steer)) - vx * omega
        dxdt[5] = 1 / self.Iz * (Ffy * self.lf * np.cos(steer) - Fry * self.lr)

        return dxdt

    def calc_forces(self, x, u, five=False):
        steer = u[1]
        psi = x[2]
        vx = x[3]
        vy = x[4]
        omega = x[5]

        acc = u[0]
        Frx = (self.Cm1 - self.Cm2 * vx) * acc - self.Cr0 - self.Cr2 * (vx ** 2)

        alphaf = steer - np.arctan2((self.lf * omega + vy), abs(vx))
        alphar = np.arctan2((self.lr * omega - vy), abs(vx))
        Ffy = self.Df * np.sin(self.Cf * np.arctan(self.Bf * alphaf))
        Fry = self.Dr * np.sin(self.Cr * np.arctan(self.Br * alphar))

        if five:
            return Ffy, Frx, Fry, alphaf, alphar
        else:
            return Ffy, Frx, Fry

    def casadi(self, x, u, dxdt):
        acc = u[0]
        steer = u[1]
        psi = x[2]
        vx = x[3]
        vy = x[4]
        omega = x[5]

        vmin = 0.05
        vy = cs.if_else(vx < vmin, 0, vy)
        omega = cs.if_else(vx < vmin, 0, omega)
        steer = cs.if_else(vx < vmin, 0, steer)
        vx = cs.if_else(vx < vmin, vmin, vx)

        Frx = (self.Cm1 - self.Cm2 * vx) * acc - self.Cr0 - self.Cr2 * (vx ** 2)
        alphaf = steer - cs.atan2((self.lf * omega + vy), vx)
        alphar = cs.atan2((self.lr * omega - vy), vx)
        Ffy = self.Df * cs.sin(self.Cf * cs.arctan(self.Bf * alphaf))
        Fry = self.Dr * cs.sin(self.Cr * cs.arctan(self.Br * alphar))

        dxdt[0] = vx * cs.cos(psi) - vy * cs.sin(psi)
        dxdt[1] = vx * cs.sin(psi) + vy * cs.cos(psi)
        dxdt[2] = omega
        dxdt[3] = 1 / self.mass * (Frx - Ffy * cs.sin(steer)) + vy * omega
        dxdt[4] = 1 / self.mass * (Fry + Ffy * cs.cos(steer)) - vx * omega
        dxdt[5] = 1 / self.Iz * (Ffy * self.lf * cs.cos(steer) - Fry * self.lr)

        return dxdt

    def sim_discrete(self, x0, u, Ts):
        n_steps = u.shape[1]
        x = np.zeros([6, n_steps + 1])
        dxdt = np.zeros([6, n_steps + 1])
        dxdt[:, 0] = self._diffequation(None, x0, [0, 0])
        x[:, 0] = x0
        for ids in range(1, n_steps + 1):
            g = self._diffequation(None, x[:, ids - 1], u[:, ids - 1]).reshape(-1, )
            x[:, ids] = x[:, ids - 1] + g * Ts
            dxdt[:, ids] = self._diffequation(None, x[:, ids], u[:, ids - 1])

        return x, dxdt

    def linearize(self, x0, u0):
        acc = u0[1]
        psi = x0[2]
        vx = x0[3]
        vy = x0[4]
        omega = x0[5]

        vmin = 0.05
        if vx < vmin:
            vy = 0
            omega = 0
            steer = 0
            vx = vmin

        sindelta = np.sin(acc)
        cosdelta = np.cos(acc)
        sinpsi = np.sin(psi)
        cospsi = np.cos(psi)

        Ffy, Frx, Fry, alphaf, alphar = self.calc_forces(x0, u0, five=True)

        dFfy_dalphaf = self.Bf * self.Cf * self.Df * np.cos(self.Cf * np.arctan(self.Bf * alphaf))
        dFfy_dalphaf *= 1 / (1 + (self.Bf * alphaf) ** 2)

        dFry_dalphar = self.Br * self.Cr * self.Dr * np.cos(self.Cr * np.arctan(self.Br * alphar))
        dFry_dalphar *= 1 / (1 + (self.Br * alphar) ** 2)

        dFfy_dvx = dFfy_dalphaf * (self.lf * omega + vy) / ((self.lf * omega + vy) ** 2 + vx ** 2)
        dFfy_dvy = -dFfy_dalphaf * vx / ((self.lf * omega + vy) ** 2 + vx ** 2)
        dFfy_domega = -dFfy_dalphaf * self.lf * vx / ((self.lf * omega + vy) ** 2 + vx ** 2)

        acc = u0[0]
        dFrx_dvx = -self.Cm2 * acc - 2 * self.Cr2 * vx
        dFrx_dvu1 = self.Cm1 - self.Cm2 * vx

        dFry_dvx = -dFry_dalphar * (self.lr * omega - vy) / ((self.lr * omega - vy) ** 2 + vx ** 2)
        dFry_dvy = -dFry_dalphar * vx / ((self.lr * omega - vy) ** 2 + vx ** 2)
        dFry_domega = dFry_dalphar * self.lr * vx / ((self.lr * omega - vy) ** 2 + vx ** 2)

        dFfy_delta = dFfy_dalphaf

        f1_psi = -vx * sinpsi - vy * cospsi
        f1_vx = cospsi
        f1_vy = -sinpsi

        f2_psi = vx * cospsi - vy * sinpsi
        f2_vx = sinpsi
        f2_vy = cospsi

        f4_vx = 1 / self.mass * (dFrx_dvx - dFfy_dvx * sindelta)
        f4_vy = 1 / self.mass * (-dFfy_dvy * sindelta + self.mass * omega)
        f4_omega = 1 / self.mass * (-dFfy_domega * sindelta + self.mass * vy)

        f5_vx = 1 / self.mass * (dFry_dvx + dFfy_dvx * cosdelta - self.mass * omega)
        f5_vy = 1 / self.mass * (dFry_dvy + dFfy_dvy * cosdelta)
        f5_omega = 1 / self.mass * (dFry_domega + dFfy_domega * cosdelta - self.mass * vx)

        f6_vx = 1 / self.Iz * (dFfy_dvx * self.lf * cosdelta - dFry_dvx * self.lr)
        f6_vy = 1 / self.Iz * (dFfy_dvy * self.lf * cosdelta - dFry_dvy * self.lr)
        f6_omega = 1 / self.Iz * (dFfy_domega * self.lf * cosdelta - dFry_domega * self.lr)

        f4_u1 = dFrx_dvu1
        f4_delta = 1 / self.mass * (-dFfy_delta * sindelta - Ffy * cosdelta)
        f5_delta = 1 / self.mass * (dFfy_delta * cosdelta - Ffy * sindelta)
        f6_delta = 1 / self.Iz * (dFfy_delta * self.lf * cosdelta - Ffy * self.lf * sindelta)

        A = np.array([
            [0, 0, f1_psi, f1_vx, f1_vy, 0],
            [0, 0, f2_psi, f2_vx, f2_vy, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, f4_vx, f4_vy, f4_omega],
            [0, 0, 0, f5_vx, f5_vy, f5_omega],
            [0, 0, 0, f6_vx, f6_vy, f6_omega],
        ])
        B = np.array([
            [0, 0],
            [0, 0],
            [0, 0],
            [f4_u1, f4_delta],
            [0, f5_delta],
            [0, f6_delta],
        ])
        g = self._diffequation(None, x0, u0).reshape(-1, )

        return A, B, g
