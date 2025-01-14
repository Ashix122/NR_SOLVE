import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameters
T = 1
nx = 100  # Number of spatial points
nt = 100  # Number of time steps
dx = 1.0 / nx
dt = T / nt

# Spatial and temporal grids
x = np.linspace(0, 1, nx, endpoint=False)  # Spatial grid
t = np.linspace(0, T, nt)  # Temporal grid
phi = np.zeros((nx, nt), dtype=complex)
pi = np.zeros((nx, nt), dtype=complex)

# Initial conditions
def initial(x):
    phi0 = np.exp(1j * 2 * np.pi * x)  # Initial condition for phi
    pi0 = np.exp(1j * 2 * np.pi * x)  # Initial condition for pi
    return phi0, pi0

phi[:, 0], pi[:, 0] = initial(x)

# Laplacian with periodic boundary conditions
def laplacian(phi, dx, index):
    left = (index - 1) % nx
    right = (index + 1) % nx
    return (phi[right] - 2 * phi[index] + phi[left]) / dx**2

# RK4 solver
def solverk4(phi, pi, dt):
    for n in range(nt - 1):
        for i in range(nx):
            # Compute intermediate steps for phi and pi
            k1_phi = -dt * pi[i, n]
            k1_pi = dt * laplacian(phi[:, n], dx, i)
            
            k2_phi = -dt * (pi[i, n] + 0.5 * k1_pi)
            k2_pi = dt * laplacian(phi[:, n] + 0.5 * k1_phi, dx, i)
            
            k3_phi = -dt * (pi[i, n] + 0.5 * k2_pi)
            k3_pi = dt * laplacian(phi[:, n] + 0.5 * k2_phi, dx, i)
            
            k4_phi = -dt * (pi[i, n] + k3_pi)
            k4_pi = dt * laplacian(phi[:, n] + k3_phi, dx, i)
            
            # Update phi and pi
            phi[i, n + 1] = phi[i, n] + (k1_phi + 2 * k2_phi + 2 * k3_phi + k4_phi) / 6
            pi[i, n + 1] = pi[i, n] + (k1_pi + 2 * k2_pi + 2 * k3_pi + k4_pi) / 6
    return phi, pi

phi, pi = solverk4(phi, pi, dt)

# Visualization
fig, ax = plt.subplots()
line_real, = ax.plot(x, np.real(phi[:, 0]), color="blue", label="Re(phi)")
line_imag, = ax.plot(x, np.imag(phi[:, 0]), color="red", label="Im(phi)")
ax.set_xlim(0, 1)
ax.set_ylim(-1.5, 1.5)
ax.set_xlabel("x")
ax.set_ylabel("phi")
ax.set_title("Wave Equation Solution")
ax.legend()

# Animation function
def update(frame):
    line_real.set_ydata(np.real(phi[:, frame]))
    line_imag.set_ydata(np.imag(phi[:, frame]))
    return line_real, line_imag

ani = FuncAnimation(fig, update, frames=nt, interval=50)
plt.show()
