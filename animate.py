import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameters
T = 1
dx = 1 / 100
dt = T / 100
x = np.linspace(0, 1, 100)  # Spatial grid
t = np.linspace(0, T, 100)  # Time grid

# Initialize arrays
phi = np.zeros((len(x), len(t)), dtype=complex)
pi = np.zeros((len(x), len(t)), dtype=complex)

# Periodic boundary conditions
def initial(xin):
    phi0 = np.exp-( 2 * np.pi * xin)  # Initial condition
    pi0 = np.sin( 2 * np.pi * xin)  # Momentum
    return phi0, pi0

# Apply initial conditions
for i in range(len(x)):
    phi[i, 0], pi[i, 0] = initial(x[i])

# Laplacian with periodic BCs
def laplacian(phi, dx, index):
    N = len(phi)
    return (-phi[(index - 1) % N] + 2 * phi[index] - phi[(index + 1) % N]) / dx**2

# RK4 solver
def solverk4(phi, pi, timesteps, indexi, dt):
    for indexa, t in enumerate(timesteps[:-1]):  # Loop over time
        k1 = -dt * pi[indexi, indexa]
        k2 = -dt * (pi[indexi, indexa] + 0.5 * dt * laplacian(phi[:, indexa], dx, indexi))
        k3 = -dt * (pi[indexi, indexa] + 0.5 * dt * laplacian(phi[:, indexa], dx, indexi))
        k4 = -dt * (pi[indexi, indexa] + dt * laplacian(phi[:, indexa], dx, indexi))
        
        phi[indexi, indexa + 1] = phi[indexi, indexa] + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return phi

# Solve for all spatial points
for i in range(len(x)):
    phi = solverk4(phi, pi, t, i, dt)

# Animation setup
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

# Animate
ani = FuncAnimation(fig, update, frames=len(t), interval=1000)
plt.show()
