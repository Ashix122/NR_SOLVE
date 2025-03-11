import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameters
nt = 1000
nx = 100
T = 1
dx = 1 / (nx + 1)
dt = T / (nt + 1)

x = np.linspace(0, 1, nx)  # Spatial grid
t = np.linspace(0, T, nt)  # Time grid

# Initialize phi and pi
phi = np.zeros((nx, nt), dtype=complex)
pi = np.zeros((nx, nt), dtype=complex)

phi0 = np.sin(2 * np.pi * x)
pi0 = np.sin(2 * np.pi * x)
phi[:, 0] = phi0
pi[:, 0] = pi0

# Laplacian with periodic boundary conditions
def laplacian(phi, dx):
    phidash = np.zeros_like(phi)
    for i in range(len(phi)):
        left = (i - 1) % len(phi)  # Wrap around left index
        right = (i + 1) % len(phi)  # Wrap around right index
        phidash[i] = (phi[right] - 2 * phi[i] + phi[left]) / dx**2
    return phidash

# Time evolution using Runge-Kutta 4
for i in range(nt - 1):
    k1pi = -laplacian(phi[:, i], dx)
    k1phi = -pi[:, i]

    k2pi = -laplacian(phi[:, i] + 0.5 * dt * k1phi, dx)
    k2phi = -(pi[:, i] + 0.5 * dt * k1pi)

    k3pi = -laplacian(phi[:, i] + 0.5 * dt * k2phi, dx)
    k3phi = -(pi[:, i] + 0.5 * dt * k2pi)

    k4pi = -laplacian(phi[:, i] + dt * k3phi, dx)
    k4phi = -(pi[:, i] + dt * k3pi)

    pi[:, i + 1] = pi[:, i] + (dt / 6) * (k1pi + 2 * k2pi + 2 * k3pi + k4pi)
    phi[:, i + 1] = phi[:, i] + (dt / 6) * (k1phi + 2 * k2phi + 2 * k3phi + k4phi)

# Visualization
fig, ax = plt.subplots()
line_real, = ax.plot(x, np.real(phi[:, 0]), color="blue", label="Re(phi)")
line_imag, = ax.plot(x, np.imag(phi[:, 0]), color="red", label="Im(phi)")
ax.set_xlim(0, 1)
ax.set_ylim(-3, 3)
ax.set_xlabel("x")
ax.set_ylabel("phi")
ax.set_title("Wave Equation Solution")
ax.legend()

# Animation function
def update(frame):
    line_real.set_ydata(np.real(phi[:, frame]))
    line_imag.set_ydata(np.imag(phi[:, frame]))
    ax.set_title(f"Wave Equation Solution at t={t[frame]:.2f}")
    return line_real, line_imag

ani = FuncAnimation(fig, update,  interval=1)
plt.show()
