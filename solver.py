import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
nt=500
nx=100

T=2
dx=(1-0)/(nx+1)
dt=(T-0)/(nt+1)
x=np.linspace(0,1,nx) #Discretisation of x(Spatial Grid)
t=np.linspace(0,T,nt)
timesteps=t
phi = np.zeros((len(x), len(t)), dtype=complex)
pi = np.zeros((len(x), len(t)), dtype=complex)

#phi0=np.exp(-0.5*((x-0.5)/0.05)**2)/(np.sqrt(2*np.pi)*0.05) #Periodic boundary condition
#pi0=np.exp(-0.5*((x-0.5)/0.05)**2)/(np.sqrt(2*np.pi)*0.05)

phi0=np.sin(2*np.pi*x)
#pi0=np.sin(x*2*np.pi)
#pi0=np.exp(1j*2*np.pi*x)
pi0=np.zeros_like(phi0)
#pi0=np.sin(2*np.pi*x)
phi[:,0],pi[:,0]=phi0,pi0
phidash=np.zeros_like(x,dtype=complex)
def laplacian(phi, t,dx):
    for index in range(nx-1):
        left = (index - 1)
        if (left==-1):
            left=(len(x)-1)
        right = (index + 1)
        if right==(len(x)):
            right=0
        phidash[index]=(phi[right] - 2 * phi[index] + phi[left])/ dx**2
    return phidash
    

for i in range(nt-1):
    k1pi=-1*laplacian(phi[:,i],i,dx)
    k1phi=-1*pi[:,i]
    k2pi=-1*laplacian(phi[:,i]+0.5*k1phi*dt,i,dx)
    k2phi=(k1phi-dt*0.5*k1pi)
    k3pi=-1*laplacian(phi[:,i]+0.5*k2phi*dt,i,dx)
    k3phi=k2phi-dt*0.5*k1pi 
    k4pi=k3pi=-1*laplacian(phi[:,i]+k3phi*dt,i,dx)
    k4phi=k1phi-dt*k1pi
    pi[:,i+1]=pi[:,i]+(1/6)*(k1pi+2*k2pi+2*k3pi+k4pi)*dt
    phi[:,i+1]=phi[:,i]+(1/6)*(k1phi+2*k2phi+2*k3phi+k4phi)*dt
    phiavg=(phi[0,i+1]+phi[-1,i+1]+phi[1,i+1]+phi[-2,i+1])/4
    piavg=(pi[0,i+1]+pi[-1,i+1]+pi[1,i+1]+pi[-2,i+1])/4

    phi[-1, i + 1] = phi[1, i + 1]=phi[0,i+1]=phi[-2,i+1]=phiavg

    pi[0, i + 1] = pi[-2, i + 1]=pi[-1, i + 1] = pi[1, i + 1]=piavg
    

    

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
