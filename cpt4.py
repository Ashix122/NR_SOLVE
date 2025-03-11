import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# Parameters
nx, ny = 20 ,20 # Number of spatial points in x and y
nt = 1000       # Number of time steps
T = 10     # Total simulation time
Lx, Ly = 1.0, 1.0  # Dimensions of the domain
dx, dy = Lx / (nx - 1), Ly / (ny - 1)  # Spatial steps
dt = T / nt                        # Time step

# Check CFL condition for stability
cfl_x = dt / dx
cfl_y = dt / dy
if max(cfl_x, cfl_y) > 1:
    raise ValueError(f"CFL condition violated: max(cfl_x, cfl_y) = {max(cfl_x, cfl_y):.2f} > 1")

# Spatial grid
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
X, Y = np.meshgrid(x, y)

# Initialize fields
phi = np.zeros((nx, ny, nt), dtype=complex)
pi = np.zeros((nx, ny, nt), dtype=complex)

# Initial conditions: Gaussian bump
phi0 = np.exp(-100 * ((X - 0.5) ** 2 + (Y - 0.5) ** 2))  # Initial displacement
pi0 = np.zeros_like(phi0)                                # Initial velocity
phi[:, :, 0], pi[:, :, 0] = phi0, pi0

# Laplacian with periodic boundary conditions
def laplacian(phi, dx, dy):
    d2phi_dx2 = (np.roll(phi, 1, axis=0) - 2 * phi + np.roll(phi, -1, axis=0)) / dx**2
    d2phi_dy2 = (np.roll(phi, 1, axis=1) - 2 * phi + np.roll(phi, -1, axis=1)) / dy**2
    return d2phi_dx2 + d2phi_dy2

# RK4 time-stepping
def rk4_step(phi, pi, dt, dx, dy):
    k1_phi = pi
    k1_pi = laplacian(phi, dx, dy)

    k2_phi = pi + 0.5 * dt * k1_pi
    k2_pi = laplacian(phi + 0.5 * dt * k1_phi, dx, dy)

    k3_phi = pi + 0.5 * dt * k2_pi
    k3_pi = laplacian(phi + 0.5 * dt * k2_phi, dx, dy)

    k4_phi = pi + dt * k3_pi
    k4_pi = laplacian(phi + dt * k3_phi, dx, dy)

    phi_new = phi + (dt / 6) * (k1_phi + 2 * k2_phi + 2 * k3_phi + k4_phi)
    pi_new = pi + (dt / 6) * (k1_pi + 2 * k2_pi + 2 * k3_pi + k4_pi)

    return phi_new, pi_new

# Solve the wave equation using RK4
for n in range(nt - 1):
    phi[:, :, n + 1], pi[:, :, n + 1] = rk4_step(phi[:, :, n], pi[:, :, n], dt, dx, dy)

# Animate the 3D solution
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection="3d")

# Initial plot
surf = ax.plot_surface(X, Y, np.real(phi[:, :, 0]), cmap="viridis", edgecolor="none")
ax.set_xlim(0, Lx)
ax.set_ylim(0, Ly)
ax.set_zlim(-0.5, 1.0)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("phi")
ax.set_title("2D Wave Equation Solution")

# Animation function
def update(frame):
    ax.clear()
    surf = ax.plot_surface(X, Y, np.real(phi[:, :, frame]), cmap="viridis", edgecolor="none")
    ax.set_xlim(0, Lx)
    ax.set_ylim(0, Ly)
    ax.set_zlim(-0.5, 1.0)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("phi")
    ax.set_title(f"2D Wave Equation Solution at t={frame * dt:.2f}")
    return surf,

# Animate
ani = FuncAnimation(fig, update, frames=nt, interval=50)
plt.show()
