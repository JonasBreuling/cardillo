import numpy as np 
from cardillo.math.algebra import cross3, ax2skew, quat2mat, quat2mat_p, norm4, quat2rot, quat2rot_p

class Spherical_joint():
    def __init__(self, r_OB1, A_IB1, q0=None, u0=None):
        self.__nq = 4
        self.__nu = 3
        self.q0 = np.array([1, 0, 0, 0]) if q0 is None else q0
        self.u0 = np.zeros(self.__nu) if u0 is None else u0
        self.r_OB1 = r_OB1
        self.A_IB1 = A_IB1
       
    def get_nq(self):
        return self.__nq

    def get_nu(self):
        return self.__nu

    def q_dot(self, t, q, u):
        return self.B(t, q) @ u

    def q_ddot(self, t, q, u, u_dot):
        q2 = q @ q
        B_q = quat2mat_p(q) / (2 * q2) \
            - np.einsum('ij,k->ijk', quat2mat(q), q / (q2**2))
        return self.B(t, q) @ u_dot + np.einsum('ijk,k,j->i', B_q[:, 1:], self.q_dot(t, q, u), u)

    def B(self, t, q):
        Q = quat2mat(q) / (2 * q @ q)
        return Q[:, 1:]

    def q_dot_q(self, t, q, u):
        q2 = q @ q
        B_q = quat2mat_p(q) / (2 * q2) \
            - np.einsum('ij,k->ijk', quat2mat(q), q / (q2**2))
        return np.einsum('ijk,j->ik', B_q[:, 1:], u)

    def A_B1B2(self, t, q):
        return quat2rot(q)

    def A_B1B2_q(self, t, q):
        return quat2rot_p(q)

    def B1_r_B1B2(self, t, q):
        return np.zeros(3)

    def B1_r_B1B2_q(self, t, q):
        return np.zeros((3, self.__nq))

    def B1_v_B1B2(self, t, q, u):
        return np.zeros(3)

    def B1_J_B1B2(self, t, q):
        return np.zeros((3, self.__nu))

    def B1_J_B1B2_q(self, t, q):
        return np.zeros((3, self.__nu, self.__nq))

    def B1_a_B1B2(self, t, q, u, u_dot):
        return np.zeros(3)

    def B1_kappa_B1B2(self, t, q, u):
        return np.zeros(3)

    def B1_kappa_B1B2_q(self, t, q, u):
        return np.zeros((3, self.__nq))

    def B1_Omega_B1B2(self, t, q, u):
        return u

    def B1_Omega_B1B2_q(self, t, q, u):
        return np.zeros((3, self.__nq))

    def B1_J_R_B1B2(self, t, q):
        return np.eye(3)

    def B1_J_R_B1B2_q(self, t, q):
        return np.zeros((3, self.__nu, self.__nq))

    def B1_Psi_B1B2(self, t, q, u, u_dot):
        return u_dot

    def B1_kappa_R_B1B2(self, t, q, u):
        return np.zeros(3)

    def B1_kappa_R_B1B2_q(self, t, q, u):
        return np.zeros((3, self.__nq))

    def B1_kappa_R_B1B2_u(self, t, q, u):
        return np.zeros((3, self.__nu))