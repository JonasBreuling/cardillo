import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pytest

from cardillo import System
from cardillo.solver import ScipyIVP, EulerBackward, MoreauClassical
from cardillo.constraints import Revolute
from cardillo.discrete import RigidBodyAxisAngle, RigidBodyQuaternion
from cardillo.forces import (
    LinearSpring,
    LinearDamper,
    PDRotational,
)
from cardillo.math import Exp_SO3, axis_angle2quat, norm

# solver parameters
t_span = (0.0, 2.0)
t0, t1 = t_span
dt = 1.0e-3

# axis angle rotation axis
psi = np.random.rand(3)
A_IK0 = Exp_SO3(psi)
print(f"A_IK0:\n{A_IK0}")

# initial rotational velocity e_z^K axis
alpha_dot0 = 2
rotation_axis = 0

# parameters
k = 1e1
d = 0.05
g_ref = 2 * np.pi
l = 0.1
m = 1
r = 0.2
A = 1 / 4 * m * r**2 + 1 / 12 * m * l**2
C = 1 / 2 * m * r**2
K_theta_S = np.diag(np.array([A, A, C]))

show = False


def run(
    RigidBody,
    Solver,
    **solver_kwargs,
):
    ############################################################################
    #                   system setup
    ############################################################################

    ####################
    # initial conditions
    ####################
    r_OP0 = np.zeros(3)
    v_P0 = np.zeros(3)
    K_Omega0 = alpha_dot0 * np.eye(3)[rotation_axis]
    u0 = np.hstack((v_P0, K_Omega0))

    ######################
    # define contributions
    ######################
    system = System()

    if RigidBody == RigidBodyAxisAngle:
        q0 = np.hstack((r_OP0, psi))
    elif RigidBody == RigidBodyQuaternion:
        n_psi = norm(psi)
        p = axis_angle2quat(psi / n_psi, n_psi)
        q0 = np.hstack((r_OP0, p))
    else:
        raise (TypeError)
    rigid_body = RigidBody(m, K_theta_S, q0, u0)

    joint = PDRotational(Revolute, Spring=LinearSpring, Damper=LinearDamper)(
        subsystem1=system.origin,
        subsystem2=rigid_body,
        axis=rotation_axis,
        r_OB0=np.zeros(3),
        A_IB0=A_IK0,
        k=k,
        d=d,
        g_ref=g_ref,
    )

    system.add(rigid_body, joint)
    system.assemble()

    ############################################################################
    #                   DAE solution
    ############################################################################
    sol = Solver(system, t1, dt, **solver_kwargs).solve()
    t, q = sol.t, sol.q

    ############################################################################
    #                   plot
    ############################################################################
    if show:
        joint.reset()
        alpha_cmp = [joint.angle(ti, qi[joint.qDOF]) for ti, qi in zip(t, q)]

        Theta = K_theta_S[rotation_axis, rotation_axis]

        def eqm(t, x):
            dx = np.zeros(2)
            dx[0] = x[1]
            dx[1] = -1 / Theta * (d * x[1] + k * (x[0] - g_ref))
            return dx

        x0 = np.array((0, alpha_dot0))
        ref = solve_ivp(eqm, [0, t1], x0, method="RK45", rtol=1e-8, atol=1e-12)
        x = ref.y
        t_ref = ref.t
        alpha_ref = x[0]

        fig, ax = plt.subplots(1, 1)

        ax.plot(t, alpha_cmp, "-k", label="alpha")
        ax.plot(t_ref, alpha_ref, "-.r", label="alpha_ref")
        ax.legend()

        plt.show()


################################################################################
# test setup
################################################################################

test_parameters = [
    (ScipyIVP, {}),
    (MoreauClassical, {}),
    (EulerBackward, {"method": "index 1"}),
    (EulerBackward, {"method": "index 2"}),
    (EulerBackward, {"method": "index 3"}),
    (EulerBackward, {"method": "index 2 GGL"}),
]


@pytest.mark.parametrize("Solver, kwargs", test_parameters)
def test_torsional_oscillator(Solver, kwargs):
    run(RigidBodyQuaternion, Solver, **kwargs)


if __name__ == "__main__":
    show = True

    # run(RigidBodyQuaternion, ScipyIVP)

    # run(RigidBodyQuaternion, MoreauClassical)

    # run(RigidBodyQuaternion, EulerBackward, method="index 1")
    # run(RigidBodyQuaternion, EulerBackward, method="index 2")
    # run(RigidBodyQuaternion, EulerBackward, method="index 3")
    # run(RigidBodyQuaternion, EulerBackward, method="index 2 GGL")