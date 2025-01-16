import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
nt=1000 
nx=40
T=1
dx=(1-0)/nx
dt=(T-0)/nt
x=np.linspace(0,1,nx) #Discretisation of x(Spatial Grid)
t=np.linspace(0,T,nt)
timesteps=t
phi = np.zeros((len(x), len(t)), dtype=complex)
pi = np.zeros((len(x), len(t)), dtype=complex)
def initial(xin):
    phi0=np.exp(1j*2*np.pi*xin) #Periodic boundary condition
    pi0=np.exp(1j*2*np.pi*xin)
    return phi0,pi0

for i in range(len(x)):
    phi[i,0],pi[i,0]=initial(x[i])

def laplacian(phi, dx,index):
    """Compute the second spatial derivative with periodic BCs."""
    if index==0:
        return (2*phi[index]-5*phi[index+1]+4*phi[index+2]-phi[index+3])*-1/dx**3
    elif index==(len(x)-1):
        return (2*phi[-1]-5*phi[-2]+4*phi[-3]-phi[-4])*-1/dx**3
    else:
        return (-phi[index-1]+2 * phi[index] - phi[index+1]) / dx**2

dpdt0=(2*phi[0,0]-5*phi[0+1,0]+4*phi[0+2,0]-phi[0+3,0])*-1/dx**3
dpdt1=(2*phi[-1,0]-5*phi[-2,0]+4*phi[-3,0]-phi[-4,0])*-1/dx**3
pi[0,:]=dpdt0*timesteps
pi[-1,:]=dpdt1*timesteps

def calculatek(dt,t,phi,index):
    return (t+dt)*laplacian(phi,dx,index)
    
def solverk4(phi,pi,timesteps,indexi,dt):
    if((indexi!=0) and (indexi!=(len(x)))):
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
        if(indexi==1):
            phi[-1,:]=phi[0,:]
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
ani = FuncAnimation(fig, update, frames=len(t), interval=1)
plt.show()

