import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def wave_solver(nx, nt, T=10):
    dx = 1.0 / (nx + 1)
    dt = T / (nt + 1)
    x = np.linspace(0, 1, nx)
    t = np.linspace(0, T, nt)
    
    phi = np.zeros((nx, nt), dtype=complex)
    pi = np.zeros((nx, nt), dtype=complex)
    
    phi0 = np.sin(2 * np.pi * x)
    pi0 = np.zeros_like(phi0)
    phi[:, 0], pi[:, 0] = phi0, pi0
    
    def laplacian(phi):
        phidash=np.zeros_like(x,dtype=complex)     
        for index in range(nx):
            left = (index - 1)
            if (left==-1):
                left=(len(x)-2)
                right = (index + 1)
            if right==(len(x)):
                right=1
        phidash[index]=(phi[right] - 2 * phi[index] + phi[left])/ dx**2
        return phidash
        

    for i in range(nt - 1):
        k1pi = -laplacian(phi[:, i])
        k1phi = -pi[:, i]
        k2pi = -laplacian(phi[:, i] + 0.5 * k1phi * dt)
        k2phi = k1phi - 0.5 * dt * k1pi
        k3pi = -laplacian(phi[:, i] + 0.5 * k2phi * dt)
        k3phi = k1phi - 0.5 * dt * k2pi
        k4pi = -laplacian(phi[:, i] + k3phi * dt)
        k4phi = k1phi - dt * k3pi
        
        pi[:, i+1] = pi[:, i] + (dt / 6) * (k1pi + 2 * k2pi + 2 * k3pi + k4pi)
        phi[:, i+1] = phi[:, i] + (dt / 6) * (k1phi + 2 * k2phi + 2 * k3phi + k4phi)
    
    return x, t, phi

def exact_solution(x, t):
    return np.sin(2 * np.pi * (x - t))

def compute_error(nx, nt):
    x, t, phi_num = wave_solver(nx, nt)
    dx = 1.0 / (nx + 1)
    dt = 10 / (nt + 1)
    error = np.zeros(nt)
    
    for i in range(nt):
        phi_exact = exact_solution(x, t[i])
        error[i] = np.linalg.norm(np.absolute(np.real(phi_num[:, i]) - phi_exact)*dx , 2)
    
    return dx, np.mean(error)

resolutions = [10, 20, 40, 80, 160]
errors = []
dxs = []
for nx in resolutions:
    dx, error = compute_error(nx, 5000)
    dxs.append(dx)
    errors.append(error)

plt.loglog(dxs, errors, marker='o', linestyle='-', label='Error')
plt.xlabel('dx')
plt.ylabel('Error')
plt.title('Convergence Study')
plt.legend()
plt.grid(True)
plt.show()


