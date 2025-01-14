import numpy as np
T=1
dx=(1-0)/101
x=np.linspace(0,1,100) #Discretisation of x(Spatial Grid)
t=np.linspace(0,T,100)
timesteps=t
phi=np.zeros(shape=(len(x),len(t)))
pi=np.zeros(shape=(len(x),len(t)))
def initial(xin):
    phi0=np.exp(1j*2*np.pi*xin) #Periodic boundary condition
    pi0=np.exp(1j*2*np.pi*xin)
    return phi0,pi0

for i in len(x):
    phi[i,0],pi[i,0]=initial(x[i])


def laplacian(phi, dx,index):
    """Compute the second spatial derivative with periodic BCs."""
    return (np.roll(phi[index-1], -1) - 2 * phi[index] + np.roll(phi[index+1], 1)) / dx**2

def solverk4(phi,pi,timesteps,indexi,dt):
    dpdt0=(2*phi[indexi,0]-5*phi[indexi+1]+4*phi[indexi+2]-phi[indexi+3])/dx**2
    


       






for i in x:
    indexi=0

    phi[indexi]=solverk4(,timesteps,indexi)













