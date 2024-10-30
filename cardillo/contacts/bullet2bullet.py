import numpy as np
import pybullet as p
from cardillo.constraints._base import concatenate_qDOF, concatenate_uDOF
from cardillo.math import Log_SO3_quat, approx_fprime
from cardillo.math.algebra import cross3, ax2skew


# TODO:
# - add pivoting friction
# - remove variation of functions with respect to closest point somehow
class Bullet2Bullet:
    def __init__(
        self,
        subsystem1Collision,
        subsystem1MultibodyBullet,
        subsystem1MultibodyCardillo,
        subsystem2Collision,
        subsystem2MultibodyBullet,
        subsystem2MultibodyCardillo,
        nla_N,
        mu,  # TODO: get this from bullet/ urdf
        e_N=None,  # TODO: get this from bullet/ urdf
        e_F=None,  # TODO: get this from bullet/ urdf
        frame_ID1=np.zeros(3),  # TODO: Are they required?
        frame_ID2=np.zeros(3),  # TODO: Are they required?
    ):
        self.subsystem1Collision = subsystem1Collision
        self.subsystem1MultibodyBullet = subsystem1MultibodyBullet
        self.subsystem1 = subsystem1MultibodyCardillo
        self.subsystem2Collision = subsystem2Collision
        self.subsystem2MultibodyBullet = subsystem2MultibodyBullet
        self.subsystem2 = subsystem2MultibodyCardillo
        self.frame_ID1 = frame_ID1
        self.frame_ID2 = frame_ID2
        self.nla_N = nla_N
        self.mu = mu * np.ones([self.nla_N])

        if np.allclose(self.mu, np.zeros_like(self.mu)):
            self.nla_F = 0
            # self.NF_connectivity = [[]]
            self.NF_connectivity = [[] for i in range(self.nla_N)]
        else:
            self.nla_F = 2 * self.nla_N
            # self.NF_connectivity = [[0, 1]]
            self.NF_connectivity = [[2 * i, 2 * i + 1] for i in range(self.nla_F)]
            self.gamma_F = self.__gamma_F

        self.e_N = np.zeros(self.nla_N) if e_N is None else e_N * np.ones(self.nla_N)
        self.e_F = np.zeros(self.nla_F) if e_F is None else e_F * np.ones(self.nla_F)

        # TODO: These have to be set before every evaluation!
        self._g_N = np.ones(self.nla_N)
        t1, t2, n = np.split(np.eye(3), 3)
        self._n = np.array([n for _ in range(self.nla_N)])
        self._t1 = np.array([t1 for _ in range(self.nla_N)])
        self._t2 = np.array([t2 for _ in range(self.nla_N)])
        # self._n = np.zeros((self.nla_N, 3))
        # self._t1 = np.zeros((self.nla_N, 3))
        # self._t2 = np.zeros((self.nla_N, 3))
        # self._n, self.t1, self.t2 = np.split(np.eye(3), 3)
        # self._K1_r_S1P1 = np.zeros(3)
        # self._K2_r_S2P2 = np.zeros(3)
        self._K1_r_S1P1 = np.zeros((self.nla_N, 3))
        self._K2_r_S2P2 = np.zeros((self.nla_N, 3))

    def assembler_callback(self):
        local_qDOF1, local_qDOF2 = concatenate_qDOF(self)
        concatenate_uDOF(self)

        nq1 = self._nq1
        nu1 = self._nu1

        # auxiliary functions for subsystem 1
        self.r_OP1 = lambda t, q: self.subsystem1.r_OP(
            t, q[:nq1], self.frame_ID1, self._K1_r_S1P1
        )
        self.r_OP1_q1 = lambda t, q: self.subsystem1.r_OP_q(
            t, q[:nq1], self.frame_ID1, self._K1_r_S1P1
        )
        self.v_P1 = lambda t, q, u: self.subsystem1.v_P(
            t, q[:nq1], u[:nu1], self.frame_ID1, self._K1_r_S1P1
        )
        self.v_P1_q1 = lambda t, q, u: self.subsystem1.v_P_q(
            t, q[:nq1], u[:nu1], self.frame_ID1, self._K1_r_S1P1
        )
        self.a_P1 = lambda t, q, u, u_dot: self.subsystem1.a_P(
            t, q[:nq1], u[:nu1], u_dot[:nu1], self.frame_ID1, self._K1_r_S1P1
        )
        self.a_P1_q1 = lambda t, q, u, u_dot: self.subsystem1.a_P_q(
            t, q[:nq1], u[:nu1], u_dot[:nu1], self.frame_ID1, self._K1_r_S1P1
        )
        self.a_P1_u1 = lambda t, q, u, u_dot: self.subsystem1.a_P_u(
            t, q[:nq1], u[:nu1], u_dot[:nu1], self.frame_ID1, self._K1_r_S1P1
        )
        self.J_P1 = lambda t, q: self.subsystem1.J_P(
            t, q[:nq1], self.frame_ID1, self._K1_r_S1P1
        )
        self.J_P1_q1 = lambda t, q: self.subsystem1.J_P_q(
            t, q[:nq1], self.frame_ID1, self._K1_r_S1P1
        )
        self.A_IK1 = lambda t, q: self.subsystem1.A_IK(t, q[:nq1], self.frame_ID1)
        self.A_IK1_q1 = lambda t, q: self.subsystem1.A_IK_q(t, q[:nq1], self.frame_ID1)
        self.Omega1 = lambda t, q, u: self.subsystem1.A_IK(
            t, q[:nq1], self.frame_ID1
        ) @ self.subsystem1.K_Omega(t, q[:nq1], u[:nu1], self.frame_ID1)
        self.Omega1_q1 = lambda t, q, u: np.einsum(
            "ijk,j->ik",
            self.subsystem1.A_IK_q(t, q[:nq1], frame_ID=self.frame_ID1),
            self.subsystem1.K_Omega(t, q[:nq1], u[:nu1], self.frame_ID1),
        ) + self.subsystem1.A_IK(
            t, q[:nq1], frame_ID=self.frame_ID1
        ) @ self.subsystem1.K_Omega_q(
            t, q[:nq1], u[:nu1], self.frame_ID1
        )

        self.Psi1 = lambda t, q, u, u_dot: self.subsystem1.A_IK(
            t, q[:nq1], self.frame_ID1
        ) @ self.subsystem1.K_Psi(t, q[:nq1], u[:nu1], u_dot[:nu1], self.frame_ID1)
        self.Psi1_q1 = lambda t, q, u, u_dot: np.einsum(
            "ijk,j->ik",
            self.subsystem1.A_IK_q(t, q[:nq1], frame_ID=self.frame_ID1),
            self.subsystem1.K_Psi(t, q[:nq1], u[:nu1], u_dot[:nu1], self.frame_ID1),
        ) + self.subsystem1.A_IK(
            t, q[:nq1], frame_ID=self.frame_ID1
        ) @ self.subsystem1.K_Psi_q(
            t, q[:nq1], u[:nu1], u_dot[:nu1], self.frame_ID1
        )
        self.Psi1_u1 = lambda t, q, u, u_dot: self.subsystem1.A_IK(
            t, q[:nq1], frame_ID=self.frame_ID1
        ) @ self.subsystem1.K_Psi_u(t, q[:nq1], u[:nu1], u_dot[:nu1], self.frame_ID1)

        self.J_R1 = lambda t, q: self.subsystem1.A_IK(
            t, q[:nq1], self.frame_ID1
        ) @ self.subsystem1.K_J_R(t, q[:nq1], self.frame_ID1)
        self.J_R1_q1 = lambda t, q: np.einsum(
            "ijk,jl->ilk",
            self.subsystem1.A_IK_q(t, q[:nq1], self.frame_ID1),
            self.subsystem1.K_J_R(t, q[:nq1], self.frame_ID1),
        ) + np.einsum(
            "ij,jkl->ikl",
            self.subsystem1.A_IK(t, q[:nq1], self.frame_ID1),
            self.subsystem1.K_J_R_q(t, q[:nq1], self.frame_ID1),
        )

        # auxiliary functions for subsystem 2
        self.r_OP2 = lambda t, q: self.subsystem2.r_OP(
            t, q[nq1:], self.frame_ID2, self._K2_r_S2P2
        )
        self.r_OP2_q2 = lambda t, q: self.subsystem2.r_OP_q(
            t, q[nq1:], self.frame_ID2, self._K2_r_S2P2
        )
        self.v_P2 = lambda t, q, u: self.subsystem2.v_P(
            t, q[nq1:], u[nu1:], self.frame_ID2, self._K2_r_S2P2
        )
        self.v_P2_q2 = lambda t, q, u: self.subsystem2.v_P_q(
            t, q[nq1:], u[nu1:], self.frame_ID2, self._K2_r_S2P2
        )
        self.a_P2 = lambda t, q, u, u_dot: self.subsystem2.a_P(
            t, q[nq1:], u[nu1:], u_dot[nu1:], self.frame_ID2, self._K2_r_S2P2
        )
        self.a_P2_q2 = lambda t, q, u, u_dot: self.subsystem2.a_P_q(
            t, q[nq1:], u[nu1:], u_dot[nu1:], self.frame_ID2, self._K2_r_S2P2
        )
        self.a_P2_u2 = lambda t, q, u, u_dot: self.subsystem2.a_P_u(
            t, q[nq1:], u[nu1:], u_dot[nu1:], self.frame_ID2, self._K2_r_S2P2
        )
        self.J_P2 = lambda t, q: self.subsystem2.J_P(
            t, q[nq1:], self.frame_ID2, self._K2_r_S2P2
        )
        self.J_P2_q2 = lambda t, q: self.subsystem2.J_P_q(
            t, q[nq1:], self.frame_ID2, self._K2_r_S2P2
        )
        self.A_IK2 = lambda t, q: self.subsystem2.A_IK(t, q[nq1:], self.frame_ID2)
        self.A_IK2_q2 = lambda t, q: self.subsystem2.A_IK_q(t, q[nq1:], self.frame_ID2)
        self.Omega2 = lambda t, q, u: self.subsystem2.A_IK(
            t, q[nq1:], self.frame_ID2
        ) @ self.subsystem2.K_Omega(t, q[nq1:], u[nu1:], self.frame_ID2)
        self.Omega2_q2 = lambda t, q, u: np.einsum(
            "ijk,j->ik",
            self.subsystem2.A_IK_q(t, q[nq1:], frame_ID=self.frame_ID2),
            self.subsystem2.K_Omega(t, q[nq1:], u[nu1:], self.frame_ID2),
        ) + self.subsystem2.A_IK(
            t, q[nq1:], frame_ID=self.frame_ID2
        ) @ self.subsystem2.K_Omega_q(
            t, q[nq1:], u[nu1:], self.frame_ID2
        )

        self.Psi2 = lambda t, q, u, u_dot: self.subsystem2.A_IK(
            t, q[nq1:], self.frame_ID2
        ) @ self.subsystem2.K_Psi(t, q[nq1:], u[nu1:], u_dot[nu1:], self.frame_ID2)
        self.Psi2_q2 = lambda t, q, u, u_dot: np.einsum(
            "ijk,j->ik",
            self.subsystem2.A_IK_q(t, q[nq1:], frame_ID=self.frame_ID2),
            self.subsystem2.K_Psi(t, q[nq1:], u[nu1:], u_dot[nu1:], self.frame_ID2),
        ) + self.subsystem2.A_IK(
            t, q[nq1:], frame_ID=self.frame_ID2
        ) @ self.subsystem2.K_Psi_q(
            t, q[nq1:], u[nu1:], u_dot[nu1:], self.frame_ID2
        )
        self.Psi2_u2 = lambda t, q, u, u_dot: self.subsystem2.A_IK(
            t, q[nq1:], frame_ID=self.frame_ID2
        ) @ self.subsystem2.K_Psi_u(t, q[nq1:], u[nu1:], u_dot[nu1:], self.frame_ID2)

        self.J_R2 = lambda t, q: self.subsystem2.A_IK(
            t, q[nq1:], self.frame_ID2
        ) @ self.subsystem2.K_J_R(t, q[nq1:], self.frame_ID2)
        self.J_R2_q2 = lambda t, q: np.einsum(
            "ijk,jl->ilk",
            self.subsystem2.A_IK_q(t, q[nq1:], self.frame_ID2),
            self.subsystem2.K_J_R(t, q[nq1:], self.frame_ID2),
        ) + np.einsum(
            "ij,jkl->ikl",
            self.subsystem2.A_IK(t, q[nq1:], self.frame_ID2),
            self.subsystem2.K_J_R_q(t, q[nq1:], self.frame_ID2),
        )

    #################
    # normal contacts
    #################
    def pre_iteration_update(self, t, q, u):
        # print(f"pre_iteration_update called")
        self.__perform_collision_detection(t, q)

    # TODO:
    # - ensure that this is only called once for a constant (t, q)
    def __perform_collision_detection(self, t, q, debug=False):
        ##################################
        # update state of involved objects
        ##################################
        r_OS1 = self.subsystem1.r_OP(t, q[: self._nq1])
        A_IK1 = self.subsystem1.A_IK(t, q[: self._nq1])
        # TODO: Since we restrict ourselves on quaternion parametrization this
        #       Exp(Log(*)) is useless and prone to round-off errors.
        p.resetBasePositionAndOrientation(
            self.subsystem1MultibodyBullet, r_OS1, np.roll(Log_SO3_quat(A_IK1), -1)
        )

        r_OS2 = self.subsystem2.r_OP(t, q[self._nq1 :])
        A_IK2 = self.subsystem2.A_IK(t, q[self._nq1 :])
        # TODO: Since we restrict ourselves on quaternion parametrization this
        #       Exp(Log(*)) is useless and prone to round-off errors.
        p.resetBasePositionAndOrientation(
            self.subsystem2MultibodyBullet, r_OS2, np.roll(Log_SO3_quat(A_IK2), -1)
        )

        ################################################################
        # perform collision detection, since "getClosestPoints" does not
        # compute the friction directions
        ################################################################
        p.performCollisionDetection()
        pts = p.getContactPoints(
            bodyA=self.subsystem1MultibodyBullet, bodyB=self.subsystem2MultibodyBullet
        )

        # pts = p.getClosestPoints(
        #     bodyA=self.subsystem1MultibodyBullet, bodyB=self.subsystem2MultibodyBullet, distance=1000,
        # )
        # if len(pts) > 1:
        #     print(f"")

        for i, point in enumerate(pts):
            print(f"len(pts): {len(pts)}")
            if i > self.nla_N - 1:
                break

            (
                contactFlag,
                bodyUniqueIdA,
                bodyUniqueIdB,
                linkIndexA,
                linkIndexB,
                positionOnA,
                positionOnB,
                contactNormalOnB,
                contactDistance,
                normalForce,
                lateralFriction1,
                lateralFrictionDir1,
                lateralFriction2,
                lateralFrictionDir2,
            ) = point

            # lateralFrictionDir1 = np.array([1, 0, 0])
            # lateralFrictionDir2 = np.array([0, 1, 0])
            # contactNormalOnB = np.array([0, 0, 1])

            if debug:
                print(f"cotact point {i}:")
                print(f" - bodyUniqueIdA: {bodyUniqueIdA}")
                print(f" - bodyUniqueIdB: {bodyUniqueIdB}")
                print(f" - contactDistance: {contactDistance}")
                print(f" - positionOnA: {positionOnA}")
                print(f" - positionOnB: {positionOnB}")
                print(f" - contactNormalOnB: {contactNormalOnB}")
                print(f" - lateralFrictionDir1: {lateralFrictionDir1}")
                print(f" - lateralFrictionDir2: {lateralFrictionDir2}")
                # A = np.vstack(
                #     (lateralFrictionDir1, lateralFrictionDir2, contactNormalOnB)
                # )
                # print(f" - A:\n{A}")
                # print(f" - det(A): {np.linalg.det(A)}")

            self._g_N[i] = contactDistance
            self._n[i] = np.array(contactNormalOnB)
            # TODO: Why is this ordering strange?
            # self._t1[i] = np.array(lateralFrictionDir1)
            # self._t2[i] = np.array(lateralFrictionDir2)
            self._t1[i] = np.array(lateralFrictionDir2)
            self._t2[i] = np.array(lateralFrictionDir1)
            self._K1_r_S1P1[i] = A_IK1.T @ (positionOnA - r_OS1)
            self._K2_r_S2P2[i] = A_IK2.T @ (positionOnB - r_OS2)

    def g_N(self, t, q):
        g_N = np.zeros(self.nla_N, dtype=q.dtype)
        for i in range(self.nla_N):
            r_OP1 = self.subsystem1.r_OP(t, q[: self._nq1], K_r_SP=self._K1_r_S1P1[i])
            r_OP2 = self.subsystem2.r_OP(t, q[self._nq1 :], K_r_SP=self._K2_r_S2P2[i])
            g_N[i] = self._n[i] @ (r_OP1 - r_OP2)
            # g_N[i] = self._n[i] @ (r_OP2 - r_OP1)
        return g_N

    def g_N_q(self, t, q):
        # return approx_fprime(q, lambda q: self.g_N(t, q)).reshape(self.nla_N, self._nq)
        g_N_q = np.zeros((self.nla_N, self._nq))
        for i in range(self.nla_N):
            r_OP1_q1 = self.subsystem1.r_OP_q(
                t, q[: self._nq1], K_r_SP=self._K1_r_S1P1[i]
            )
            r_OP2_q2 = self.subsystem2.r_OP_q(
                t, q[self._nq1 :], K_r_SP=self._K2_r_S2P2[i]
            )
            g_N_q[i, : self._nq1] = self._n[i] @ r_OP1_q1
            g_N_q[i, self._nq1 :] = -self._n[i] @ r_OP2_q2
            # g_N_q[i, : self._nq1] = -self._n[i] @ r_OP1_q1
            # g_N_q[i, self._nq1 :] = self._n[i] @ r_OP2_q2
        return g_N_q

    def g_N_dot(self, t, q, u):
        g_N_dot = np.zeros(self.nla_N, dtype=np.common_type(q, u))
        for i in range(self.nla_N):
            v_P1 = self.subsystem1.v_P(
                t, q[: self._nq1], u[: self._nu1], K_r_SP=self._K1_r_S1P1[i]
            )
            v_P2 = self.subsystem2.v_P(
                t, q[self._nq1 :], u[self._nu1 :], K_r_SP=self._K2_r_S2P2[i]
            )
            g_N_dot[i] = self._n[i] @ (v_P1 - v_P2)
            # g_N_dot[i] = self._n[i] @ (v_P2 - v_P1)
        return g_N_dot

    # TODO: n_q!
    def g_N_dot_q(self, t, q, u):
        raise NotImplementedError
        g_N_dot_q = np.zeros((self.nla_N, self.nq))
        g_N_dot_q[0, : self._nq1] = self._n @ self.v_P1_q1(t, q)
        g_N_dot_q[0, self._nq1 :] = -self._n @ self.v_P2_q2(t, q)
        return g_N_dot_q

    def g_N_dot_u(self, t, q):
        g_N_dot_u = np.zeros((self.nla_N, self._nu))
        for i in range(self.nla_N):
            J_P1 = self.subsystem1.J_P(t, q[: self._nq1], K_r_SP=self._K1_r_S1P1[i])
            J_P2 = self.subsystem2.J_P(t, q[self._nq1 :], K_r_SP=self._K2_r_S2P2[i])
            g_N_dot_u[i, : self._nu1] = self._n[i] @ J_P1
            g_N_dot_u[i, self._nu1 :] = -self._n[i] @ J_P2
            # g_N_dot_u[i, : self._nu1] = -self._n[i] @ J_P1
            # g_N_dot_u[i, self._nu1 :] = self._n[i] @ J_P2
        return g_N_dot_u

    def W_N(self, t, q):
        return self.g_N_dot_u(t, q).T

    def g_N_ddot(self, t, q, u, u_dot):
        g_N_ddot = np.zeros(self.nla_N, dtype=np.common_type(q, u, u_dot))
        for i in range(self.nla_N):
            a_P1 = self.subsystem1.a_P(
                t,
                q[: self._nq1],
                u[: self._nu1],
                u_dot[: self._nu1],
                K_r_SP=self._K1_r_S1P1[i],
            )
            a_P2 = self.subsystem2.a_P(
                t,
                q[self._nq1 :],
                u[self._nu1 :],
                u_dot[self._nu1 :],
                K_r_SP=self._K2_r_S2P2[i],
            )
            g_N_ddot[i] = self._n[i] @ (a_P1 - a_P2)
            # g_N_ddot[i] = self._n[i] @ (a_P2 - a_P1)
        return g_N_ddot

    def g_N_ddot_q(self, t, q, u, u_dot):
        raise NotImplementedError
        return np.array(
            [self._n @ self.a_P_q(t, q, u, u_dot)], dtype=np.common_type(q, u, u_dot)
        )

    def g_N_ddot_u(self, t, q, u, u_dot):
        raise NotImplementedError
        return np.array(
            [self._n @ self.a_P_u(t, q, u, u_dot)], dtype=np.common_type(q, u, u_dot)
        )

    def Wla_N_q(self, t, q, la_N):
        return approx_fprime(q, lambda q: self.W_N(t, q) @ la_N)
        # raise NotImplementedError
        # return la_N[0] * np.einsum("i,ijk->jk", self._n, self.J_P_q(t, q))

    def __gamma_F(self, t, q, u):
        gamma_F = np.zeros(self.nla_F, dtype=np.common_type(q, u))
        for i in range(self.nla_N):
            v_P1 = self.subsystem1.v_P(
                t, q[: self._nq1], u[: self._nu1], K_r_SP=self._K1_r_S1P1[i]
            )
            v_P2 = self.subsystem2.v_P(
                t, q[self._nq1 :], u[self._nu1 :], K_r_SP=self._K2_r_S2P2[i]
            )
            gamma_F[2 * i] = self._t1[i] @ (v_P1 - v_P2)
            gamma_F[2 * i + 1] = self._t2[i] @ (v_P1 - v_P2)
        return gamma_F

    def gamma_F_dot(self, t, q, u, u_dot):
        gamma_F = np.zeros(self.nla_F, dtype=np.common_type(q, u, u_dot))
        for i in range(self.nla_N):
            a_P1 = self.subsystem1.a_P(
                t,
                q[: self._nq1],
                u[: self._nu1],
                u_dot[: self._nu1],
                K_r_SP=self._K1_r_S1P1[i],
            )
            a_P2 = self.subsystem2.a_P(
                t,
                q[self._nq1 :],
                u[self._nu1 :],
                u_dot[self._nu1 :],
                K_r_SP=self._K2_r_S2P2[i],
            )
            gamma_F[2 * i] = self._t1[i] @ (a_P1 - a_P2)
            gamma_F[2 * i + 1] = self._t2[i] @ (a_P1 - a_P2)
        return gamma_F

    def gamma_F_q(self, t, q, u):
        # return approx_fprime(q, lambda q: self.gamma_F(t, q, u))
        gamma_F_q = np.zeros((self.nla_F, self._nq))
        for i in range(self.nla_N):
            v_P1_q1 = self.subsystem1.v_P_q(
                t, q[: self._nq1], u[: self._nu1], K_r_SP=self._K1_r_S1P1[i]
            )
            v_P2_q2 = self.subsystem2.v_P_q(
                t, q[self._nq1 :], u[self._nu1 :], K_r_SP=self._K2_r_S2P2[i]
            )
            gamma_F_q[2 * i, : self._nq1] = self._t1[i] @ v_P1_q1
            gamma_F_q[2 * i, self._nq1 :] = -self._t1[i] @ v_P2_q2
            gamma_F_q[2 * i + 1, : self._nq1] = self._t2[i] @ v_P1_q1
            gamma_F_q[2 * i + 1, self._nq1 :] = -self._t2[i] @ v_P2_q2
        return gamma_F_q

    def gamma_F_u(self, t, q):
        gamma_F_u = np.zeros((self.nla_F, self._nu))
        for i in range(self.nla_N):
            J_P1 = self.subsystem1.J_P(t, q[: self._nq1], K_r_SP=self._K1_r_S1P1[i])
            J_P2 = self.subsystem2.J_P(t, q[self._nq1 :], K_r_SP=self._K2_r_S2P2[i])
            gamma_F_u[2 * i, : self._nu1] = self._t1[i] @ J_P1
            gamma_F_u[2 * i, self._nu1 :] = -self._t1[i] @ J_P2
            gamma_F_u[2 * i + 1, : self._nu1] = self._t2[i] @ J_P1
            gamma_F_u[2 * i + 1, self._nu1 :] = -self._t2[i] @ J_P2
        return gamma_F_u

    # def gamma_F_dot_q(self, t, q, u, u_dot):
    #     # return approx_fprime(q, lambda q: self.gamma_F_dot(t, q, u, u_dot))
    #     r_PC = -self.r * self.n
    #     a_C_q = self.a_P_q(t, q, u, u_dot) - ax2skew(r_PC) @ self.Psi_q(t, q, u, u_dot)
    #     return self.t1t2 @ a_C_q

    # def gamma_F_dot_u(self, t, q, u, u_dot):
    #     # return approx_fprime(u, lambda u: self.gamma_F_dot(t, q, u, u_dot))
    #     r_PC = -self.r * self.n
    #     a_C_u = self.a_P_u(t, q, u, u_dot) - ax2skew(r_PC) @ self.Psi_u(t, q, u, u_dot)
    #     return self.t1t2 @ a_C_u

    def W_F(self, t, q):
        return self.gamma_F_u(t, q).T

    def Wla_F_q(self, t, q, la_F):
        return approx_fprime(q, lambda q: self.gamma_F_u(t, q).T @ la_F)

    #     # J_C_q = self.J_P_q(t, q) + self.r * np.einsum(
    #     #     "ij,jkl->ikl", ax2skew(self.n), self.J_R_q(t, q)
    #     # )
    #     # dense = np.einsum("i,ij,jkl->kl", la_F, self.t1t2, J_C_q)
