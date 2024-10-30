# file taken from https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/examples/getClosestPoints.py
# most import information is found in https://github.com/bulletphysics/bullet3/blob/master/docs/pybullet_quickstartguide.pdf
import pybullet as p
import time
import pybullet_data
import numpy as np

from cardillo.math import Exp_SO3, Log_SO3_quat
from cardillo.discrete import Frame, RigidBodyQuaternion
from cardillo import System
from cardillo.forces import Force
from cardillo.contacts import Bullet2Bullet
from cardillo.solver import Moreau

# case = "sphere"
case = "cuboid"
# case = "cylinder"
# case = "stl_car"

# TODO:
# - implement a generic contact between two cardillo-wrapped bullet objects
# - implement a slightly faster contact between cardillo-wrapped bullet object and plane?
# - check quaternion ordering: I think bullet uses the last entray for the scalar part


def collision_shape_sphere(radius):
    return p.createCollisionShape(p.GEOM_SPHERE, radius=radius)


def collision_shape_cuboid(width, height, length):
    return p.createCollisionShape(
        p.GEOM_BOX, halfExtents=[0.5 * width, 0.5 * height, 0.5 * length]
    )


def collision_shape_cylinder(radius, height):
    return p.createCollisionShape(
        p.GEOM_CYLINDER,
        radius=radius,
        height=height,
    )


def from_stl(
    fileName,
    mass,
    scale=1,
    r_OS0=np.zeros(3),
    A_IK0=np.eye(3),
    v_S0=np.zeros(3),
    K_omega_IK0=np.zeros(3),
):
    col_shape = p.createCollisionShape(
        shapeType=p.GEOM_MESH,
        fileName=fileName,
        meshScale=np.ones(3) * scale,
        flags=p.URDF_INITIALIZE_SAT_FEATURES,  # |p.GEOM_FORCE_CONCAVE_TRIMESH should only be used with fixed (mass=0) objects!
        # flags=p.URDF_INITIALIZE_SAT_FEATURES | p.GEOM_FORCE_CONCAVE_TRIMESH,
    )

    viz_shape = p.createVisualShape(
        shapeType=p.GEOM_MESH,
        fileName=fileName,
        meshScale=np.ones(3) * scale,
    )

    P0 = np.roll(Log_SO3_quat(A_IK0), -1)
    multibodyBullet = p.createMultiBody(
        baseMass=mass,
        baseCollisionShapeIndex=col_shape,
        baseVisualShapeIndex=viz_shape,
        basePosition=r_OS0,
        baseOrientation=P0,
    )

    mass, K_Theta_S, M = inertia_from_bullet(multibodyBullet)

    q0 = np.concatenate((r_OS0, P0))
    u0 = np.concatenate((v_S0, K_omega_IK0))
    multibodyCardillo = RigidBodyQuaternion(mass, K_Theta_S, q0=q0, u0=u0)

    return col_shape, viz_shape, multibodyBullet, multibodyCardillo


def inertia_from_bullet(object_ID):
    # inertia matrix from bullet (no link position)
    M = np.array(p.calculateMassMatrix(object_ID, []))

    # total mass
    mass = M[-1, -1]
    assert np.allclose(mass * np.ones(3), np.diag(M[3:, 3:]))

    # inertia
    K_Theta_S = M[:3, :3]

    return mass, K_Theta_S, M


def create_multibody(
    mass,
    collision_shape,
    r_OS0=np.zeros(3),
    A_IK0=np.eye(3),
    v_S0=np.zeros(3),
    K_omega_IK0=np.zeros(3),
    debug=True,
):
    # create rigid body
    # TODO: Why this brakes physics???
    P0 = np.roll(Log_SO3_quat(A_IK0), -1)
    # P0 = Log_SO3_quat(A_IK0)
    multibodyBullet = p.createMultiBody(
        baseMass=mass,
        baseCollisionShapeIndex=collision_shape,
        basePosition=r_OS0,
        baseOrientation=P0,
    )

    mass, K_Theta_S, M = inertia_from_bullet(multibodyBullet)
    if debug:
        print(f"body: ???")
        print(f" - mass: {mass}")
        print(f" - K_Theta_S:\n{K_Theta_S}")
        print(f" - bullet M:\n{M}")

        dynamics_info = p.getDynamicsInfo(multibodyBullet, -1)
        (
            mass,
            lateral_friction,
            local_inertia_diagonal,
            local_inertial_pos,
            local_inertial_orn,
            restitution,
            rolling_friction,
            spinning_friction,
            contact_damping,
            contact_stiffness,
            body_type,
            collision_margin,
        ) = dynamics_info
        print(f" - mass: {mass}")
        print(f" - lateral_friction: {lateral_friction}")
        print(f" - local_inertia_diagonal: {local_inertia_diagonal}")
        print(f" - local_inertial_pos: {local_inertial_pos}")
        print(f" - local_inertial_orn: {local_inertial_orn}")
        print(f" - restitution: {restitution}")
        print(f" - rolling_friction: {rolling_friction}")
        print(f" - spinning_friction: {spinning_friction}")
        print(f" - contact_damping: {contact_damping}")
        print(f" - contact_stiffness: {contact_stiffness}")
        print(f" - body_type: {body_type}")
        print(f" - collision_margin: {collision_margin}")

    # cardillo rigid body
    q0 = np.concatenate((r_OS0, P0))
    u0 = np.concatenate((v_S0, K_omega_IK0))
    multibodyCardillo = RigidBodyQuaternion(mass, K_Theta_S, q0=q0, u0=u0)

    return collision_shape, multibodyBullet, multibodyCardillo


def create_plane(
    r_OS0=np.zeros(3),
    A_IK0=np.eye(3),
):
    # create rigid body
    P0 = np.roll(Log_SO3_quat(A_IK0), -1)
    planeBullet = p.loadURDF(
        "plane.urdf",
        basePosition=r_OS0,
        baseOrientation=P0,
        useFixedBase=True,
    )

    # cardillo frame
    planeCardillo = Frame(r_OP=r_OS0, A_IK=A_IK0)
    # planeCardillo = Frame()

    return planeBullet, planeCardillo


p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# make visualization clean
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

print(f"-" * 80)
print(f"simulation part")
print(f"-" * 80)

# create plane
planeBullet, planeCardillo = create_plane()

# system with plane
system = System()
system.add(planeCardillo)

if case == "sphere":
    # create sphere
    sphere_mass = 0.8
    sphere_radius = 0.1
    sphere_r_OP0 = np.array([0, 0, 1.0])
    v_S0 = np.array([0, 3, 0])
    K_omega_IK0 = np.array([200, 10, 0])
    sphereCollision, sphereMultibodyBullet, sphereMultibodyCardillo = create_multibody(
        sphere_mass,
        collision_shape_sphere(sphere_radius),
        sphere_r_OP0,
        v_S0=v_S0,
        K_omega_IK0=K_omega_IK0,
    )

    contact = Bullet2Bullet(
        sphereCollision,
        sphereMultibodyBullet,
        sphereMultibodyCardillo,
        planeBullet,
        planeBullet,
        planeCardillo,
        nla_N=1,
        mu=0.3,
        e_N=0.5,
    )

    system.add(
        sphereMultibodyCardillo, Force(np.array([0, 0, -10]), sphereMultibodyCardillo)
    )

    bodies = [(sphereMultibodyBullet, sphereMultibodyCardillo)]

elif case == "cuboid":
    # create cuboid
    cuboid_mass = 1.2
    # cuboid_mass = 0.1
    cuboid_width = 0.1
    cuboid_height = 0.2
    cuboid_length = 0.3
    cuboid_r_OP0 = np.array([0.25, 0.4, 1.0])
    cuboid_A_IK0 = Exp_SO3(np.random.rand(3))
    # cuboid_A_IK0 = np.eye(3)
    K_omega_IK0 = np.random.rand(3)
    cuboidCollision, cuboidMultibodyBullet, cuboidMultibodyCardillo = create_multibody(
        cuboid_mass,
        collision_shape_cuboid(cuboid_width, cuboid_height, cuboid_length),
        cuboid_r_OP0,
        cuboid_A_IK0,
        K_omega_IK0=K_omega_IK0,
    )

    contact = Bullet2Bullet(
        cuboidCollision,
        cuboidMultibodyBullet,
        cuboidMultibodyCardillo,
        planeBullet,
        planeBullet,
        planeCardillo,
        nla_N=5,
        # mu=0.3,
        mu=0.0,
        # e_N=0.5,
        # e_N=0.001,
        e_N=0.0,
    )

    system.add(
        cuboidMultibodyCardillo, Force(np.array([0, 0, -10]), cuboidMultibodyCardillo)
    )

    bodies = [(cuboidMultibodyBullet, cuboidMultibodyCardillo)]

elif case == "cylinder":
    mass = 1.2
    radius = 0.1
    height = 0.4
    r_OP0 = np.array([0.25, 0.4, 1.0])
    # A_IK0 = Exp_SO3(np.random.rand(3))
    A_IK0 = np.eye(3)
    collision, multibodyBullet, multibodyCardillo = create_multibody(
        mass,
        collision_shape_cylinder(radius, height),
        r_OP0,
        A_IK0,
    )

    contact = Bullet2Bullet(
        collision,
        multibodyBullet,
        multibodyCardillo,
        planeBullet,
        planeBullet,
        planeCardillo,
        nla_N=5,
        mu=0.3,
        e_N=0.0,
    )

    system.add(multibodyCardillo, Force(np.array([0, 0, -10]), multibodyCardillo))

    bodies = [(multibodyBullet, multibodyCardillo)]

elif case == "stl_car":
    mass = 1.2
    r_OS0 = np.array([0, 0, 5.0])
    # A_IK0 = np.eye(3)
    A_IK0 = Exp_SO3(np.random.rand(3))
    collision, visualization, multibodyBullet, multibodyCardillo = from_stl(
        "examples/PyBullet/MB_S_class_W_140_Pullman.STL",
        mass,
        scale=1e-3,
        r_OS0=r_OS0,
        A_IK0=A_IK0,
    )

    contact = Bullet2Bullet(
        collision,
        multibodyBullet,
        multibodyCardillo,
        planeBullet,
        planeBullet,
        planeCardillo,
        nla_N=10,
        mu=0.3,
        e_N=0.0,
    )

    system.add(multibodyCardillo, Force(np.array([0, 0, -10]), multibodyCardillo))

    bodies = [(multibodyBullet, multibodyCardillo)]

else:
    raise NotImplementedError

system.add(contact)
system.assemble()

t1 = 2
dt = 1e-2
# dt = 5e-3
# dt = 1e-3
# sol = MoreauClassical(system, t1=t1, dt=dt).solve()
sol = NonsmoothSimplecticEulerProjected(system, t1=t1, h=dt, atol=1e-10).solve()
# sol = NPIRK(system, t1=t1, h=dt, butcher_tableau=RadauIIATableau()).solve()
t, q = sol.t, sol.q

# TODO: loop over solution here
# p.setGravity(0, 0, -10)
print(f"-" * 80)
print(f"animation part")
print(f"-" * 80)
while p.isConnected():
    # p.stepSimulation()
    # time.sleep(1 / 240)

    for i in range(len(t)):
        for bodyBullet, bodyCardillo in bodies:
            r_OP = bodyCardillo.r_OP(t[i], q[i][bodyCardillo.qDOF])
            A_IK = bodyCardillo.A_IK(t[i], q[i][bodyCardillo.qDOF])
            p.resetBasePositionAndOrientation(
                bodyBullet, r_OP, np.roll(Log_SO3_quat(A_IK), -1)
            )

        time.sleep(dt)
