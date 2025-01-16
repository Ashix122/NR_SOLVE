import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# Parameters
nx, ny, nz = 32, 32, 32  # Grid points in x, y, z
Lx, Ly, Lz = 1.0, 1.0, 1.0  # Domain size
nu = 0.2                # Viscosity
T = 2.0                   # Total simulation time
dt = 0.002                # Time step
nt = int(T / dt)          # Number of time steps
dx, dy, dz = Lx / nx, Ly / ny, Lz / nz  # Grid spacing

# Spatial grid
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
z = np.linspace(0, Lz, nz)
X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

# Initialize velocity and pressure fields
u = np.zeros((nx, ny, nz, nt))  # Velocity in x-direction
v = np.zeros((nx, ny, nz, nt))  # Velocity in y-direction
w = np.zeros((nx, ny, nz, nt))  # Velocity in z-direction
p = np.zeros((nx, ny, nz))      # Pressure

# Initial velocity field: small sinusoidal disturbance
u[:, :, :, 0] = np.sin(2 * np.pi * Y) * np.sin(2 * np.pi * Z)
v[:, :, :, 0] = np.sin(2 * np.pi * X) * np.sin(2 * np.pi * Z)
w[:, :, :, 0] = np.sin(2 * np.pi * X) * np.sin(2 * np.pi * Y)

# Compute the Laplacian (with periodic BCs)
def laplacian(field):
    return (
        (np.roll(field, 1, axis=0) - 2 * field + np.roll(field, -1, axis=0)) / dx**2 +
        (np.roll(field, 1, axis=1) - 2 * field + np.roll(field, -1, axis=1)) / dy**2 +
        (np.roll(field, 1, axis=2) - 2 * field + np.roll(field, -1, axis=2)) / dz**2
    )

# Compute the divergence (with periodic BCs)
def divergence(u, v, w):
    return (
        (np.roll(u, -1, axis=0) - u) / dx +
        (np.roll(v, -1, axis=1) - v) / dy +
        (np.roll(w, -1, axis=2) - w) / dz
    )

# Pressure correction using Jacobi iteration
def pressure_correction(p, div_u, iterations=50):
    for _ in range(iterations):
        p = (
            (np.roll(p, 1, axis=0) + np.roll(p, -1, axis=0)) / dx**2 +
            (np.roll(p, 1, axis=1) + np.roll(p, -1, axis=1)) / dy**2 +
            (np.roll(p, 1, axis=2) + np.roll(p, -1, axis=2)) / dz**2 -
            div_u
        ) / (2 / dx**2 + 2 / dy**2 + 2 / dz**2)
    return p

# Time-stepping using Euler's method
for n in range(nt - 1):
    # Compute nonlinear term (advection)
    u_adv = u[:, :, :, n] * np.roll(u[:, :, :, n], -1, axis=0) + \
            v[:, :, :, n] * np.roll(u[:, :, :, n], -1, axis=1) + \
            w[:, :, :, n] * np.roll(u[:, :, :, n], -1, axis=2)
    v_adv = u[:, :, :, n] * np.roll(v[:, :, :, n], -1, axis=0) + \
            v[:, :, :, n] * np.roll(v[:, :, :, n], -1, axis=1) + \
            w[:, :, :, n] * np.roll(v[:, :, :, n], -1, axis=2)
    w_adv = u[:, :, :, n] * np.roll(w[:, :, :, n], -1, axis=0) + \
            v[:, :, :, n] * np.roll(w[:, :, :, n], -1, axis=1) + \
            w[:, :, :, n] * np.roll(w[:, :, :, n], -1, axis=2)

    # Compute viscous term
    u_diff = nu * laplacian(u[:, :, :, n])
    v_diff = nu * laplacian(v[:, :, :, n])
    w_diff = nu * laplacian(w[:, :, :, n])

    # Update velocities with advection and diffusion
    u_new = u[:, :, :, n] + dt * (-u_adv + u_diff)
    v_new = v[:, :, :, n] + dt * (-v_adv + v_diff)
    w_new = w[:, :, :, n] + dt * (-w_adv + w_diff)

    # Compute divergence
    div_u = divergence(u_new, v_new, w_new)

    # Pressure correction
    p = pressure_correction(p, div_u)

    # Correct velocities
    u[:, :, :, n + 1] = u_new - dt * (np.roll(p, -1, axis=0) - p) / dx
    v[:, :, :, n + 1] = v_new - dt * (np.roll(p, -1, axis=1) - p) / dy
    w[:, :, :, n + 1] = w_new - dt * (np.roll(p, -1, axis=2) - p) / dz

# Animation: Velocity magnitude
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection="3d")
velocity_magnitude = np.sqrt(u**2 + v**2 + w**2)

# Initial plot
surf = ax.plot_surface(
    X[:, :, nz // 2], Y[:, :, nz // 2], velocity_magnitude[:, :, nz // 2, 0],
    cmap="viridis", edgecolor="none"
)
ax.set_xlim(0, Lx)
ax.set_ylim(0, Ly)
ax.set_zlim(0, 1)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("Velocity Magnitude")
ax.set_title("3D Navier-Stokes Velocity Magnitude")

# Animation update function
def update(frame):
    ax.clear()
    surf = ax.plot_surface(
        X[:, :, nz // 2], Y[:, :, nz // 2], velocity_magnitude[:, :, nz // 2, frame],
        cmap="viridis", edgecolor="none"
    )
    ax.set_xlim(0, Lx)
    ax.set_ylim(0, Ly)
    ax.set_zlim(0, 1)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("Velocity Magnitude")
    ax.set_title(f"3D Navier-Stokes Velocity Magnitude at t={frame * dt:.2f}")
    return surf,

# Create animation
ani = FuncAnimation(fig, update, frames=nt, interval=1)
plt.show()
