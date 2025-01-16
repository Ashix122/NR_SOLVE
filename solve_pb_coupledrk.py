import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
nt=10000
nx=50
T=0.5
dx=(1-0)/(nx+1)
dt=(T-0)/(nt+1)
x=np.linspace(0,1,nx) #Discretisation of x(Spatial Grid)
t=np.linspace(0,T,nt)
timesteps=t
phi = np.zeros((len(x), len(t)), dtype=complex)
pi = np.zeros((len(x), len(t)), dtype=complex)

#phi0=np.exp(-0.5*((x-0.5)/0.1)**2)/(np.sqrt(2*np.pi)*0.1) #Periodic boundary condition
#pi0=np.exp(-0.5*((x-0.5)/0.1)**2)/(np.sqrt(2*np.pi)*0.1)

phi0=1*np.sin(x*2*np.pi)
pi0=1*np.sin(x*2*np.pi)

phi[:,0],pi[:,0]=phi0,pi0

def laplacian(phi, dx,index,indext):
    """Compute the second spatial derivative with periodic BCs."""
    left = (index - 1) % nx
    right = (index + 1) % nx
    phi[-1:indext]=phi[0,indext]
    return (phi[right,indext] - 2 * phi[index,indext] + phi[left,indext])/ dx**2


    
def solverk4(phi,pi,timesteps,indexi,dt):
    
    if indexi==0:
        pi[-1,:]=pi[indexi,:]
    indexa=0
    for t in timesteps:
        if(t==T):
            break
        k1phi=pi[indexi,indexa]
        k1pi=laplacian(phi,dx,indexi,indexa)
        k2phi=(pi[indexi,indexa]+dt*0.5*k1pi)
        k2pi=laplacian(phi,dx,indexi,indexa)+0.5*dt*k1phi
        k3phi=k2phi
        k3pi=k2pi
        k4phi=k3phi+dt*k3pi
        k4pi=k3pi+dt*k3phi+((dt**2)*(1/6)*k1pi)
        phi[indexi,indexa+1]=phi[indexi,indexa]-(k1phi+2*k2phi+2*k3phi+k4phi)*dt/6
        pi[indexi,indexa+1]=pi[indexi,indexa]-(k1pi+2*k2pi+2*k3pi+k4pi)*dt/6
        indexa=indexa+1 
    return phi,pi
   
indexi=0

for i in x[0:-1]:
    phi,pi=solverk4(phi,pi,timesteps,indexi,dt)
    if(indexi==1):
        phi[-1,:]=phi[0,:]
        pi[-1,:]=pi[0,:]
    indexi=indexi+1


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
    ax.set_title("Wave Equation Solution at t="+str(t[frame]))
    return line_real, line_imag

# Animate
ani = FuncAnimation(fig, update, frames=len(t), interval=1)
ani.save("solverk4.mp4")
