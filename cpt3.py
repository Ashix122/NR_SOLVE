import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameters
nt = 1000 # Number of time steps
nx = 40     # Number of spatial points
T = 1       # Total simulation time
dx = 1 / (nx + 1)  # Spatial step
dt = T / (nt + 1)  # Time step

# Check CFL condition for stability
cfl = dt / dx
if cfl > 1:
    raise ValueError(f"CFL condition violated: dt/dx = {cfl:.2f} > 1")

# Spatial and temporal grids
x = np.linspace(0, 1, nx)
t = np.linspace(0, T, nt)

# Initialize fields
phi = np.zeros((nx, nt), dtype=complex)
pi = np.zeros((nx, nt), dtype=complex)

# Initial conditions
phi0 = np.sin(2 * np.pi * x)  # Initial displacement
pi0 = np.cos(2 * np.pi * x)  # Initial velocity
phi[:, 0], pi[:, 0] = phi0, pi0

# Laplacian with periodic boundary conditions
def laplacian(phi, dx, index):
    left = (index - 1) % nx
    right = (index + 1) % nx
    return (phi[right] - 2 * phi[index] + phi[left]) / dx**2

# Time-stepping function
def solve_wave_equation(phi, pi, dt, dx, nt, nx):
    for n in range(nt - 1):
        # Update pi using the spatial derivative of phi
        for i in range(nx):
            pi[i, n + 1] = pi[i, n] + dt * laplacian(phi[:, n], dx, i)
        
        # Update phi using pi
        for i in range(nx):
            phi[i, n + 1] = phi[i, n] + dt * pi[i, n + 1]
        
        # Enforce periodic boundary conditions
        phi[0, n + 1] = phi[-1, n + 1]=0
        pi[0, n + 1] = pi[-1, n + 1]=0

    return phi, pi

# Solve the wave equation
phi, pi = solve_wave_equation(phi, pi, dt, dx, nt, nx)

# Plotting setup
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
    ax.set_title(f"Wave Equation Solution at t={t[frame]:.3f}")
    return line_real, line_imag

# Animate
ani = FuncAnimation(fig, update, frames=nt, interval=1)
plt.show()
