import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
nt=5000
nx=100
T=10
dx=(1-0)/(nx+1)
dt=(T-0)/(nt+1)

x=np.linspace(0,1,nx) #Discretisation of x(Spatial Grid)
t=np.linspace(0,T,nt)
print(len(x))

timesteps=t
phi = np.zeros((len(x), len(t)), dtype=complex)
pi = np.zeros((len(x), len(t)), dtype=complex)

# #Periodic boundary condition
#phi0=np.sin(2*np.pi*x)
phi0=np.exp(-0.5*((x-0.5)/0.1)**2)/(np.sqrt(2*np.pi)*0.09)
#phi0=np.exp(1j*2*np.pi*x)
#phi0=1*np.sin(2*np.pi*x)
#pi0=2*np.pi*np.cos(2*np.pi*x)
#pi0=2*np.pi*1j*np.exp(1j*2*np.pi*x)
#pi0=np.sin(2*np.pi*x)
pi0=np.zeros_like(phi0)
phi[:,0],pi[:,0]=phi0,pi0
phidash=np.zeros_like(x,dtype=complex)
def laplacian(phi,dx=dx):
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

def rhs(vars):
    dpidt=-1*laplacian(vars[1])
    dphidt=-vars[0]
    return np.array([dpidt,dphidt])


for i in range(nt-1):
    k1=rhs([pi[:,i],phi[:,i]])
    k2=rhs([pi[:,i]+0.5*k1[0]*dt,phi[:,i]+0.5*k1[1]*dt])
    k3=rhs([pi[:,i]+0.5*k2[0]*dt,phi[:,i]+0.5*k2[1]*dt])
    k4=rhs([pi[:,i]+k3[0]*dt,phi[:,i]+k3[1]*dt])
    
    pi[:,i+1]=pi[:,i]+(1/6)*(k1+2*k2+2*k3+k4)[0]*dt
    
    phi[:,i+1]=phi[:,i]+(1/6)*(k1+2*k2+2*k3+k4)[1]*dt
    #phi[-1,i+1]=phi[0,i+1]

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
frame=range(nt)
ani = FuncAnimation(fig, update,  interval=1,frames=len(t),repeat_delay=10000)
plt.show()
