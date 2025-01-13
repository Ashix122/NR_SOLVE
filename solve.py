import numpy as np
T=1
dx=(1-0)/101
x=np.linspace(0,1,100) #Discretisation of x(Spatial Grid)
t=np.linspace(0,T,100)
timesteps=t
phi=np.zeros(shape=(len(x),len(t)))
def initial(xin):
    phi0=np.exp(1j*2*np.pi*xin) #Periodic boundary condition
    pi0=np.zeros_like(xin)
    return [phi0,pi0]

def laplacian(phi, dx,index):
    """Compute the second spatial derivative with periodic BCs."""
    return (np.roll(phi[index-1], -1) - 2 * phi[index] + np.roll(phi[index+1], 1)) / dx**2

def solverk4(phi,pi,timesteps,indexi,dt):
    phi[indexi,0],pi[indexi,0]=initial(x[indexi])
    kh1=-pi[indexi,0]*dt
    kp1=-laplacian(phi[:,t],dx,indexi)*dt
    kp2=-pi()
    for i in range(1,len(phi[indexi,:]):
        t=timesteps[i]
        kh1=-pi[index1,]
        kp1=



       






for i in x:
    indexi=0
    phi[indexi]=solverk4(,timesteps,indexi)













