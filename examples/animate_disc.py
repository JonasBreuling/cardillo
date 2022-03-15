import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from cardillo.math import A_IK_basic, pi

if __name__ == "__main__":
    # auxiliary simulation data and auxiliary rigid body functons
    t0 = 0
    t1 = 1
    num = 100
    t = np.linspace(t0, t1, num)
    q = np.random.rand(num, 6)

    class Disc:
        def __init__(self):
            self.radius = 1
            self.r_OP0 = np.zeros(3)
            self.r_OP1 = np.random.rand(3)

        def r_OP(self, t, q):
            return self.r_OP0 + t * (self.r_OP1 - self.r_OP0)

        def A_IK(self, t, q):
            phi = 2 * pi * t
            basic = A_IK_basic(phi)
            return basic.x() @ basic.y() @ basic.z()

        def boundary(self, t, q, num=100):
            phi = np.linspace(0, 2 * np.pi, num, endpoint=True)
            K_r_SP = self.radius * np.vstack([np.sin(phi), np.zeros(num), np.cos(phi)])
            return (
                np.repeat(self.r_OP(t, q), num).reshape(3, num)
                + self.A_IK(t, q) @ K_r_SP
            )

    disc = Disc()

    scale = 1

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_xlim3d(left=-scale, right=scale)
    ax.set_ylim3d(bottom=-scale, top=scale)
    ax.set_zlim3d(bottom=0, top=2 * scale)

    slowmotion = 1
    fps = 200
    animation_time = slowmotion * t1
    target_frames = int(fps * animation_time)
    frac = max(1, int(len(t) / target_frames))
    if frac == 1:
        target_frames = len(t)
    interval = 1000 / fps

    frames = target_frames
    t = t[::frac]
    q = q[::frac]

    def create(t, q):
        x_S, y_S, z_S = disc.r_OP(t, q)

        A_IK = disc.A_IK(t, q)
        d1 = A_IK[:, 0] * disc.radius
        d2 = A_IK[:, 1] * disc.radius
        d3 = A_IK[:, 2] * disc.radius

        (COM,) = ax.plot([x_S], [y_S], [z_S], "ok")
        (bdry,) = ax.plot([], [], [], "-k")
        (trace,) = ax.plot([], [], [], "--k")
        (d1_,) = ax.plot(
            [x_S, x_S + d1[0]], [y_S, y_S + d1[1]], [z_S, z_S + d1[2]], "-r"
        )
        (d2_,) = ax.plot(
            [x_S, x_S + d2[0]], [y_S, y_S + d2[1]], [z_S, z_S + d2[2]], "-g"
        )
        (d3_,) = ax.plot(
            [x_S, x_S + d3[0]], [y_S, y_S + d3[1]], [z_S, z_S + d3[2]], "-b"
        )

        return COM, bdry, trace, d1_, d2_, d3_

    COM, bdry, trace, d1_, d2_, d3_ = create(0, q[0])

    def update(t, q, COM, bdry, trace, d1_, d2_, d3_):
        x_S, y_S, z_S = disc.r_OP(t, q)

        x_bdry, y_bdry, z_bdry = disc.boundary(t, q)

        A_IK = disc.A_IK(t, q)
        d1 = A_IK[:, 0] * disc.radius
        d2 = A_IK[:, 1] * disc.radius
        d3 = A_IK[:, 2] * disc.radius

        COM.set_data(np.array([x_S]), np.array([y_S]))
        COM.set_3d_properties(np.array([z_S]))

        bdry.set_data(np.array(x_bdry), np.array(y_bdry))
        bdry.set_3d_properties(np.array(z_bdry))

        d1_.set_data(np.array([x_S, x_S + d1[0]]), np.array([y_S, y_S + d1[1]]))
        d1_.set_3d_properties(np.array([z_S, z_S + d1[2]]))

        d2_.set_data(np.array([x_S, x_S + d2[0]]), np.array([y_S, y_S + d2[1]]))
        d2_.set_3d_properties(np.array([z_S, z_S + d2[2]]))

        d3_.set_data(np.array([x_S, x_S + d3[0]]), np.array([y_S, y_S + d3[1]]))
        d3_.set_3d_properties(np.array([z_S, z_S + d3[2]]))

        return COM, bdry, trace, d1_, d2_, d3_

    def animate(i):
        update(t[i], q[i], COM, bdry, trace, d1_, d2_, d3_)

    anim = animation.FuncAnimation(
        fig, animate, frames=frames, interval=interval, blit=False
    )

    plt.show()