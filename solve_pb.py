import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
nt=500
nx=100
T=1
dx=(1-0)/(nx+1)
dt=(T-0)/(nt+1)
x=np.linspace(0,1,nx) #Discretisation of x(Spatial Grid)
t=np.linspace(0,T,nt)
timesteps=t
phi = np.zeros((len(x), len(t)), dtype=complex)
pi = np.zeros((len(x), len(t)), dtype=complex)

phi0=np.exp(-((x-0.5)/0.3)**2)/(np.sqrt(2*np.pi)*0.3) #Periodic boundary condition
pi0=np.exp(-((x-0.5)/0.3)**2)/(np.sqrt(2*np.pi)*0.3)



phi[:,0],pi[:,0]=phi0,pi0

def laplacian(phi, dx,index):
    """Compute the second spatial derivative with periodic BCs."""
    left = (index - 1) % nx
    right = (index + 1) % nx
    return (phi[right] - 2 * phi[index] + phi[left]) / dx**2



def calculatek(dt,t,phi,index):
    return (t+dt)*laplacian(phi,dx,index)
    
def solverk4(phi,pi,timesteps,indexi,dt):
    
    pi[indexi,:]=timesteps*laplacian(phi,dx,indexi)
    indexa=0
    for t in timesteps:
        if(t==T):
            break;
        k1=-dt*pi[indexi,indexa]
        k2=-dt*calculatek(dt/2,t,phi[:,indexa],indexi)
        k3=-dt*calculatek(dt/2,t,phi[:,indexa],indexi)
        k4=-dt*calculatek(dt,t,phi[:,indexa],indexi)
        phi[indexi,indexa+1]=phi[indexi,indexa]+(k1+2*k2+2*k3+k4)/6
        indexa=indexa+1 
    return phi,pi

for i in x:
    indexi=0
    for a in x:
        
        phi,pi=solverk4(phi,pi,timesteps,indexi,dt)
        if(a==1):
            break
        indexi=indexi+1

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
    ax.set_title("Wave Equation Solution at t="+str(t[frame]))
    return line_real, line_imag

# Animate
ani = FuncAnimation(fig, update, frames=len(t), interval=1000)
plt.show()
