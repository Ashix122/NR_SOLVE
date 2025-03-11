import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def wave_solver(nx, nt, T=10):
    dx = 1.0 / (nx + 1)
    dt = T / (nt + 1)
    x = np.linspace(0, 1, nx)
    t = np.linspace(0, T, nt)
    if (dx<dt):
        raise ValueError("CFL condition violated: Decrease dt or increase dx.")
    phi = np.zeros((nx, nt), dtype=complex)
    pi = np.zeros((nx, nt), dtype=complex)
    
    phi0 = np.sin(2 * np.pi * x)
    #phi0=np.exp(-0.5*((x-0.5)/0.09)**2)/(np.sqrt(2*np.pi)*0.09)
    pi0 = 2*np.pi*np.cos(2 * np.pi * x)
    phi[:, 0], pi[:, 0] = phi0, pi0
    
    def laplacian(phi):
        phidash=np.zeros_like(x,dtype=complex)
        for index in range(nx):
            left = (index - 1)
            right = (index + 1)
            if (left==-1):
                left=(len(x)-2)     
            if right==(len(x)):
                right=1
            phidash[index]=(phi[right] - 2 * phi[index] + phi[left])/ dx**2
            phidash[0]=phidash[len(x)-1]
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


def compute_pointwise(nx, nt, nxfine, ntfine,param,nxvfine,ntvfine):
    x, t, phi_num = wave_solver(nx, nt)
    x_fine, t_fine, phi_num_fine = wave_solver(nxfine, ntfine)
    x_vfine,t_vfine,phi_num_vfine=wave_solver(nxvfine,ntvfine)
    x_vvfine,t_vvfine,phi_num_vvfine=wave_solver(2*nxvfine,2*ntvfine)
    print(x_vfine)
    print(x_vvfine)
    #store=phi_num
    # Match indices directly since x_fine is 2x the resolution of x
    phi_num_matched = phi_num_fine[::2, ::2]
    phi_num_matched_vfine=phi_num_vfine[::4,::4]
    phi_num_matched_vvfine=phi_num_vvfine[::8,::8]
    global tcalc
    tcalc=t
    # Compute the difference
    store = ((phi_num_matched - phi_num))
    store2=2*((phi_num_matched_vfine-phi_num_matched))
    store3=4*((phi_num_matched_vvfine-phi_num_matched_vfine))
    point_plot(store,store2,store3,x)
    
def point_plot(store,store2,store3,x):
    fig, ax = plt.subplots()
    line_real, = ax.plot(x, np.real(store[:, 0]), color="blue", label="Medium-low")
    line_imag, = ax.plot(x, np.imag(store2[:, 0]), color="red", label="2*(high-medium)")
    line3,=ax.plot(x, np.imag(store3[:, 0]), color="red", label="4*(higher-high)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0,np.max([store,store2,store3]))
    ax.set_xlabel("x")
    ax.set_ylabel("phi")
    ax.set_title("Wave Equation Solution")
    ax.legend()

    # Animation function
    def update(frame):
        line_real.set_ydata((store[:, frame]))
        line_imag.set_ydata((store2[:, frame]))
        line3.set_ydata((store3[:, frame]))

        ax.set_title(f"Wave Equation Solution at t={tcalc[frame]:.2f}")
        return line_real, line_imag

    ani = FuncAnimation(fig, update, interval=10, frames=len(tcalc), repeat_delay=10000)
    ani.save("conv.mp4")

compute_pointwise(100,2000,200,4000,2,400,8000)