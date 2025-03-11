import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def wave_solver(dx, dt, T=10):
    x = np.arange(0, 1 + dx, dx)
    t = np.arange(0, T + dt, dt)
    nx, nt = len(x), len(t)
    
    if dx < dt:
        raise ValueError("CFL condition violated: Decrease dt or increase dx.")
    
    phi = np.zeros((nx, nt), dtype=complex)
    pi = np.zeros((nx, nt), dtype=complex)
    
    phi[:, 0] = np.sin(2 * np.pi * x)
    pi[:, 0] = 2 * np.pi * np.cos(2 * np.pi * x)
    
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
            #phidash[0]=phidash[len(x)-1]
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
        
        pi[:, i + 1] = pi[:, i] + (dt / 6) * (k1pi + 2 * k2pi + 2 * k3pi + k4pi)
        phi[:, i + 1] = phi[:, i] + (dt / 6) * (k1phi + 2 * k2phi + 2 * k3phi + k4phi)
    
    return x, t, phi

def compute_pointwise_self(dx, dt):
    x, t, phi_num = wave_solver(dx, dt)
    x_fine, t_fine, phi_num_fine = wave_solver(dx/2, dt/2)
    x_vfine, t_vfine, phi_num_vfine = wave_solver(dx/4, dt/4)
    x_vvfine, t_vvfine, phi_num_vvfine = wave_solver(dx/8, dt/8)
    
    phi_num_matched = phi_num_fine[::2, ::2]
    phi_num_matched_vfine = phi_num_vfine[::4, ::4]
    phi_num_matched_vvfine = phi_num_vvfine[::8, ::8]
    
    store = phi_num_matched - phi_num
    store2 = 4 * (phi_num_matched_vfine - phi_num_matched)
    store3 = 16 * (phi_num_matched_vvfine - phi_num_matched_vfine)
    
    point_plot(store, store2, store3, x, t,"self")

def point_plot(store, store2, store3, x,t,typer):
    fig, ax = plt.subplots()
    if typer == "self":
        a="Medium-low"
        b="4*(high-medium)"
        c="16*(higher-high)"
    elif typer == "exact":
        a="low-exact"
        b="Medium-exact"
        c="High-exact"
    line_real, = ax.plot(x, np.real(store[:, 0]), color="blue", label=a)
    line_imag, = ax.plot(x, np.imag(store2[:, 0]), color="red", label=b)
    line3, = ax.plot(x, np.imag(store3[:, 0]), color="green", label=c)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, np.max([store, store2, store3]))
    ax.set_xlabel("x")
    ax.set_ylabel("phi")
    ax.set_title("Wave Equation Solution")
    ax.legend()
    
    def update(frame):
        line_real.set_ydata(store[:, frame])
        line_imag.set_ydata(store2[:, frame])
        line3.set_ydata(store3[:, frame])
        ax.set_title(f"Wave Equation Solution at t={t[frame]:.2f}")
        return line_real, line_imag, line3
    
    ani = FuncAnimation(fig, update, interval=10, frames=len(t), repeat_delay=10000)
    plt.show()

def exact_solution(dx, dt,T=10):
    x = np.arange(0, 1 + dx, dx)
    t = np.arange(0, T + dt, dt)
    nx, nt = len(x), len(t)
    phi = np.zeros((nx, nt), dtype=complex)

    for i in range(nx):
        for j in range(nt):
            phi[i,j]=np.sin(2 * np.pi * (x[i] - t[j]))

    return phi

def compute_norm_exact(dx, dt):
    
    x, t, phi_num = wave_solver(dx, dt)
    x_fine, t_fine, phi_num_fine = wave_solver(dx/2, dt/2)
    x_vfine, t_vfine, phi_num_vfine = wave_solver(dx/4, dt/4)
    x_vvfine, t_vvfine, phi_num_vvfine = wave_solver(dx/8, dt/8)
    store=phi_num-exact_solution(dx,dt)
    store2=(phi_num_fine-exact_solution(dx/2,dt/2))[:,::2]
    store3=(phi_num_vfine-exact_solution(dx/4,dt/4))[:,::4]
    store4=(phi_num_vvfine-exact_solution(dx/8,dx/8))[:,::8]
    error1=error2=error3=error4=np.zeros_like(t)
    for i in range(len(t)):
        error1[i]=np.linalg.norm(np.abs(store[:,i])*dx,2)
        error2[i]=np.linalg.norm(np.abs(store2[:,i])*dx/2,2)
        error3[i]=np.linalg.norm(np.abs(store3[:,i])*dx/4,2)
        error4[i]=np.linalg.norm(np.abs(store4[:,i])*dx/8,2)
    error1=error2/error1
    error2=error3/error2
    error3=error4/error3
    plt.plot(t,error1,color="red",label="(Medium-Exact)/(Low-Exact)")
    plt.plot(t,error2,color="blue",label="(High-Exact)/(Medium-Exact)")
    plt.plot(t,error3,color="green",label="(Higher-Exact)/(High-Exact)")
    plt.xlabel("Time")
    plt.ylabel("Norm Convergence Factor")
    plt.title("Norm Exact Convergence")
    plt.legend()
    plt.show()
    plt.show()

def compute_norm_self(dx, dt):
    
    x, t, phi_num = wave_solver(dx, dt)
    x_fine, t_fine, phi_num_fine = wave_solver(dx/2, dt/2)
    x_vfine, t_vfine, phi_num_vfine = wave_solver(dx/4, dt/4)
    x_vvfine, t_vvfine, phi_num_vvfine = wave_solver(dx/8, dt/8)
    
    phi_num_matched = phi_num_fine[::2, ::2]
    phi_num_matched_vfine = phi_num_vfine[::4, ::4]
    phi_num_matched_vvfine = phi_num_vvfine[::8, ::8]
    
    store = phi_num_matched - phi_num
    store2 = 4 * (phi_num_matched_vfine - phi_num_matched)
    store3 = 16 * (phi_num_matched_vvfine - phi_num_matched_vfine)
    error1=error2=error3=error4=np.zeros_like(t)

    for i in range(len(t)):
        error1[i]=np.linalg.norm(np.abs(store[:,i])*dx,2)
        error2[i]=np.linalg.norm(np.abs(store2[:,i])*dx/2,2)
        error3[i]=np.linalg.norm(np.abs(store3[:,i])*dx/4,2)
        
    error1=error2/error1
    error2=error3/error2
    
    plt.plot(t,error1,color="red",label="High-Medium/Medium-Low")
    plt.plot(t,error2,color="Blue",label="Higher-High/High-Medium")
    plt.xlabel("Time")
    plt.ylabel("Norm Convergence Factor")
    plt.title("Norm Self Convergence")
    plt.legend()
    plt.show()
def compute_pointwise_exact(dx,dt):   

    x, t, phi_num = wave_solver(dx, dt)
    x_fine, t_fine, phi_num_fine = wave_solver(dx/2, dt/2)
    x_vfine, t_vfine, phi_num_vfine = wave_solver(dx/4, dt/4)
    x_vvfine, t_vvfine, phi_num_vvfine = wave_solver(dx/8, dt/8)

    
    
    store = phi_num - exact_solution(dx,dt)
    store2 = 4 * (phi_num_fine - exact_solution(dx/2,dt/2))
    store3 = 16 * (phi_num_vfine - exact_solution(dx/4,dt/4))
    store = store
    store2 = store2[::2, ::2]
    store3 = store3[::4, ::4]
    
    point_plot(store, store2, store3, x, t,"exact")


#compute_pointwise_self(dx=0.01, dt=0.01)   
#compute_norm_self(0.01,0.01)
#compute_norm_exact(0.01,0.01)
compute_pointwise_exact(0.01,0.01)


