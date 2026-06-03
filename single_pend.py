"""
Single-pendulum payload dynamics for a crawler-crane / suspended-load AMC.

Derives the equations of motion symbolically via Lagrangian mechanics (SymPy),
then lambdifies them to JAX (or NumPy) callables for fast evaluation inside
the JIT-compiled DWA planner.

Author: Zhuomin Zhou
"""

import sympy as sp
import numpy as np
from sympy.matrices import Matrix
from scipy.integrate import solve_ivp
from functools import partial


class SinglePendulum:
    def __init__(self, l, m) -> None:
        """
        Args:
            l: rope length [m]
            m: payload mass [kg]
        """
        self.l, self.m = l, m
        self.g = 9.8  # gravitational acceleration

        # Generalised coordinates (5 DOF: cart x,y,z + pendulum theta_y, theta_z)
        self.x, self.y, self.z = sp.symbols('x y z')
        self.x_dot, self.y_dot, self.z_dot = sp.symbols('x_dot y_dot z_dot')
        self.x_ddot, self.y_ddot, self.z_ddot = sp.symbols('x_ddot y_ddot z_ddot')

        self.theta_y, self.theta_z = sp.symbols('theta_y theta_z')
        self.thetadot_y, self.thetadot_z = sp.symbols('thetadot_y thetadot_z')
        self.thetaddot_y, self.thetaddot_z = sp.symbols('thetaddot_y thetaddot_z')

        self.q = sp.Matrix([self.x, self.y, self.z, self.theta_y, self.theta_z])
        self.q_dot = sp.Matrix([self.x_dot, self.y_dot, self.z_dot,
                                self.thetadot_y, self.thetadot_z])
        self.q_ddot = sp.Matrix([self.x_ddot, self.y_ddot, self.z_ddot,
                                 self.thetaddot_y, self.thetaddot_z])

        # Free (unconstrained) coordinates = pendulum angles
        self.q_free = sp.Matrix([self.theta_y, self.theta_z])
        self.qdot_free = sp.Matrix([self.thetadot_y, self.thetadot_z])
        self.qddot_free = sp.Matrix([self.thetaddot_y, self.thetaddot_z])

    def forward_kinemaics(self):
        """Position of the payload mass in the world frame."""
        x = self.x + self.l * sp.sin(self.theta_y)
        y = self.y + self.l * sp.cos(self.theta_y) * sp.sin(self.theta_z)
        z = self.z - self.l * sp.cos(self.theta_y) * sp.cos(self.theta_z)
        return sp.Matrix([x, y, z])

    def cal_kinemaic_energy(self):
        p_load = self.forward_kinemaics()
        v_p = p_load.jacobian(self.q) * self.q_dot
        T = self.m * v_p.T * v_p / 2
        return T

    def cal_potential_energy(self):
        p_load = self.forward_kinemaics()
        U = Matrix([self.m * self.g * p_load[2]])
        return U

    def cal_lagrangian(self):
        return self.cal_kinemaic_energy() - self.cal_potential_energy()

    def cal_eom(self, IF_JAX=False):
        """
        Derive the analytical equation of motion for theta_y_ddot and theta_z_ddot,
        then lambdify into callables.

        Args:
            IF_JAX: if True, return jax.numpy-compatible callables (for use inside JIT).

        Returns:
            (acc_beta_y, acc_beta_z): callables taking
                (x_ddot, y_ddot, z_ddot, theta_y, thetadot_y, theta_z, thetadot_z)
                and returning the second derivatives of the pendulum angles.
        """
        L = self.cal_lagrangian()
        L_q = L.jacobian(self.q_free).T
        L_qdot = L.jacobian(self.qdot_free).T
        # d/dt of dL/dqdot via chain rule
        L_qdotT = (L_qdot.jacobian(self.q) * self.q_dot
                   + L_qdot.jacobian(self.q_dot) * self.q_ddot)
        Lqs = L_qdotT - L_q

        qddot_free = sp.solve(Lqs, self.qddot_free, simplify=False)
        print('The equation of motion is solved')

        theta_y_ddot_eom = qddot_free[self.thetaddot_y]
        theta_z_ddot_eom = qddot_free[self.thetaddot_z]
        syms = [self.x_ddot, self.y_ddot, self.z_ddot,
                self.theta_y, self.thetadot_y,
                self.theta_z, self.thetadot_z]
        modules = 'jax' if IF_JAX else 'numpy'
        acc_beta_y = sp.lambdify(syms, theta_y_ddot_eom, modules=modules)
        acc_beta_z = sp.lambdify(syms, theta_z_ddot_eom, modules=modules)
        return (acc_beta_y, acc_beta_z)


def solve_anatical_eq1(s0, t, controll_coordinates, q_freelamda):
    """
    Reference NumPy ODE solver using scipy.integrate.solve_ivp.
    Kept for offline verification; the planner itself uses a JAX RK4 integrator.

    Args:
        s0: initial state [theta_y, theta_z, thetadot_y, thetadot_z]
        t:  time vector
        controll_coordinates: callable returning (x_ddot, y_ddot, z_ddot) at time t
        q_freelamda: (acc_beta_x, acc_beta_z) from SinglePendulum.cal_eom()

    Returns:
        Trajectory of the pendulum state, shape (len(t), 4).
    """
    acc_beta_x, acc_beta_z = q_freelamda
    dsdt_fun_p = partial(
        dsdt_fun,
        controll_coordinates=controll_coordinates,
        acc_beta_x=acc_beta_x,
        acc_beta_z=acc_beta_z,
    )
    sol = solve_ivp(dsdt_fun_p, t_span=(t[0], t[-1]), y0=s0, t_eval=t)
    return np.nan_to_num(sol.y.T)


def dsdt_fun(_t, s, controll_coordinates, acc_beta_x, acc_beta_z):
    theta_y, theta_z, thetadot_y, thetadot_z = s
    x_ddot, y_ddot, z_ddot = controll_coordinates(_t)

    thetaddot_y = acc_beta_x(x_ddot, y_ddot, z_ddot,
                             theta_y, thetadot_y,
                             theta_z, thetadot_z)
    thetaddot_z = acc_beta_z(x_ddot, y_ddot, z_ddot,
                             theta_y, thetadot_y,
                             theta_z, thetadot_z)
    dydt = np.zeros(4)
    dydt[0] = thetadot_y
    dydt[1] = thetadot_z
    dydt[2] = thetaddot_y
    dydt[3] = thetaddot_z
    return dydt
