import numpy as np
from cardillo.math import norm


class TwoPointInteraction:
    def __init__(
        self,
        subsystem1,
        subsystem2,
        frame_ID1=np.zeros(3, dtype=float),
        frame_ID2=np.zeros(3, dtype=float),
        K_r_SP1=np.zeros(3, dtype=float),
        K_r_SP2=np.zeros(3, dtype=float),
    ):
        self.subsystem1 = subsystem1
        self.frame_ID1 = frame_ID1
        self.K_r_SP1 = K_r_SP1

        self.subsystem2 = subsystem2
        self.frame_ID2 = frame_ID2
        self.K_r_SP2 = K_r_SP2

    def assembler_callback(self):
        qDOF1 = self.subsystem1.qDOF
        qDOF2 = self.subsystem2.qDOF
        local_qDOF1 = self.subsystem1.local_qDOF_P(self.frame_ID1)
        local_qDOF2 = self.subsystem2.local_qDOF_P(self.frame_ID2)
        self.qDOF = np.concatenate((qDOF1[local_qDOF1], qDOF2[local_qDOF2]))
        self._nq1 = len(local_qDOF1)
        self._nq2 = len(local_qDOF2)
        self._nq = self._nq1 + self._nq2
        q01 = self.subsystem1.q0
        q02 = self.subsystem2.q0
        self.q0 = np.concatenate((q01[local_qDOF1], q02[local_qDOF2]))

        uDOF1 = self.subsystem1.uDOF
        uDOF2 = self.subsystem2.uDOF
        local_uDOF1 = self.subsystem1.local_uDOF_P(self.frame_ID1)
        local_uDOF2 = self.subsystem2.local_uDOF_P(self.frame_ID2)
        self.uDOF = np.concatenate((uDOF1[local_uDOF1], uDOF2[local_uDOF2]))
        self._nu1 = len(local_uDOF1)
        self._nu2 = len(local_uDOF2)
        self._nu = self._nu1 + self._nu2
        u01 = self.subsystem1.u0
        u02 = self.subsystem2.u0
        self.u0 = np.concatenate((u01[local_uDOF1], u02[local_uDOF2]))

        self.r_OP1 = lambda t, q: self.subsystem1.r_OP(
            t, q[: self._nq1], self.frame_ID1, self.K_r_SP1
        )
        self.r_OP1_q = lambda t, q: self.subsystem1.r_OP_q(
            t, q[: self._nq1], self.frame_ID1, self.K_r_SP1
        )
        self.J_P1 = lambda t, q: self.subsystem1.J_P(
            t, q[: self._nq1], self.frame_ID1, self.K_r_SP1
        )
        self.J_P1_q = lambda t, q: self.subsystem1.J_P_q(
            t, q[: self._nq1], self.frame_ID1, self.K_r_SP1
        )
        self.v_P1 = lambda t, q, u: self.subsystem1.v_P(
            t, q[: self._nq1], u[: self._nu1], self.frame_ID1, self.K_r_SP1
        )
        self.v_P1_q = lambda t, q, u: self.subsystem1.v_P_q(
            t, q[: self._nq1], u[: self._nu1], self.frame_ID1, self.K_r_SP1
        )

        self.r_OP2 = lambda t, q: self.subsystem2.r_OP(
            t, q[self._nq1 :], self.frame_ID2, self.K_r_SP2
        )
        self.r_OP2_q = lambda t, q: self.subsystem2.r_OP_q(
            t, q[self._nq1 :], self.frame_ID2, self.K_r_SP2
        )
        self.J_P2 = lambda t, q: self.subsystem2.J_P(
            t, q[self._nq1 :], self.frame_ID2, self.K_r_SP2
        )
        self.J_P2_q = lambda t, q: self.subsystem2.J_P_q(
            t, q[self._nq1 :], self.frame_ID2, self.K_r_SP2
        )
        self.v_P2 = lambda t, q, u: self.subsystem2.v_P(
            t, q[self._nq1 :], u[self._nu1 :], self.frame_ID2, self.K_r_SP2
        )
        self.v_P2_q = lambda t, q, u: self.subsystem2.v_P_q(
            t, q[self._nq1 :], u[self._nu1 :], self.frame_ID2, self.K_r_SP2
        )

    # auxiliary functions
    def l(self, t, q):
        return norm(self.r_OP2(t, q) - self.r_OP1(t, q))

    def l_q(self, t, q):
        r_OP1_q = self.r_OP1_q(t, q)
        r_OP2_q = self.r_OP2_q(t, q)

        n = self._n(t, q)
        return np.hstack((-n @ r_OP1_q, n @ r_OP2_q))

    def l_dot(self, t, q, u):
        return self._n(t, q) @ (self.v_P2(t, q, u) - self.v_P1(t, q, u))

    def l_dot_q(self, t, q, u):
        n_q1, n_q2 = self._n_q(t, q)
        n = self._n(t, q)
        v_P1 = self.v_P1(t, q, u)
        v_P2 = self.v_P2(t, q, u)
        v_P1P2 = v_P2 - v_P1

        nq1 = self._nq1
        gamma_q = np.zeros(self._nq)
        gamma_q[:nq1] = -n @ self.v_P1_q(t, q, u) + v_P1P2 @ n_q1
        gamma_q[nq1:] = n @ self.v_P2_q(t, q, u) + v_P1P2 @ n_q2
        return gamma_q

    def l_dot_u(self, t, q, u):
        n = self._n(t, q)

        nu1 = self._nu1
        l_dot_u = np.zeros(self._nu)
        l_dot_u[:nu1] = -n @ self.J_P1(t, q)
        l_dot_u[nu1:] = n @ self.J_P2(t, q)
        return l_dot_u

    def _n(self, t, q):
        r_OP1 = self.r_OP1(t, q)
        r_OP2 = self.r_OP2(t, q)
        return (r_OP2 - r_OP1) / norm(r_OP2 - r_OP1)

    def _n_q(self, t, q):
        r_OP1_q = self.r_OP1_q(t, q)
        r_OP2_q = self.r_OP2_q(t, q)

        r_P1P2 = self.r_OP2(t, q) - self.r_OP1(t, q)
        g = norm(r_P1P2)
        n = r_P1P2 / g
        P = (np.eye(3) - np.outer(n, n)) / g
        n_q1 = -P @ r_OP1_q
        n_q2 = P @ r_OP2_q

        return n_q1, n_q2

    def W_l(self, t, q):
        n = self._n(t, q)
        J_P1 = self.J_P1(t, q)
        J_P2 = self.J_P2(t, q)
        return np.concatenate([-J_P1.T @ n, J_P2.T @ n])

    def W_l_q(self, t, q):
        nq1 = self._nq1
        nu1 = self._nu1
        n = self._n(t, q)
        n_q1, n_q2 = self._n_q(t, q)
        J_P1 = self.J_P1(t, q)
        J_P2 = self.J_P2(t, q)
        J_P1_q = self.J_P1_q(t, q)
        J_P2_q = self.J_P2_q(t, q)

        # dense blocks
        W_q = np.zeros((self._nu, self._nq))
        W_q[:nu1, :nq1] = -J_P1.T @ n_q1 + np.einsum("i,ijk->jk", -n, J_P1_q)
        W_q[:nu1, nq1:] = -J_P1.T @ n_q2
        W_q[nu1:, :nq1] = J_P2.T @ n_q1
        W_q[nu1:, nq1:] = J_P2.T @ n_q2 + np.einsum("i,ijk->jk", n, J_P2_q)

        return W_q

    def export(self, sol_i, **kwargs):
        points = [
            self.r_OP1(sol_i.t, sol_i.q[self.qDOF]),
            self.r_OP2(sol_i.t, sol_i.q[self.qDOF]),
        ]
        cells = [("line", [[0, 1]])]
        h = self._h(sol_i.t, sol_i.q[self.qDOF], sol_i.u[self.uDOF])
        la = self.W_l(sol_i.t, sol_i.q[self.qDOF]).T @ h
        n = self._n(sol_i.t, sol_i.q[self.qDOF])
        point_data = dict(la=[la, la], n=[n, -n])
        # cell_data = dict(h=[h])
        cell_data = dict(
            n=[[n]],
            g=[[self.l(sol_i.t, sol_i.q[self.qDOF])]],
            g_dot=[[self.l_dot(sol_i.t, sol_i.q[self.qDOF], sol_i.u[self.uDOF])]],
        )
        if hasattr(self, "E_pot"):
            E_pot = [self.E_pot(sol_i.t, sol_i.q[self.qDOF])]
            cell_data["E_pot"] = [E_pot]

        return points, cells, point_data, cell_data