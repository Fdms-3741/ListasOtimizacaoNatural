# 20200713 gabriel@pads.ufrj.br

import numpy as np
import matplotlib.pyplot as plt

# Given Cost Function

def J(X):
    L=0.5
    Xorigin=np.zeros([2,1])    
    JA=np.mean(np.sum(np.power(X-np.tile(Xorigin,(1,P)),2),axis=0))
    JB=0
    for i in range(0,P):
        for j in range(i+1,P):
            JB+=1/np.sum(np.power(X[:,i]-X[:,j],2))
    JB=JB/(P*(P-1)/2)
    return JA+L*JB

# Definitions

N=int(1e5);  K=7; T0=5e-1; epsilon=1e-1 # (1.0211) (*)
# N=int(1e5);  K=8; T0=5;    epsilon=2e-1 # (1.1311)
# N=int(1e4);  K=8; T0=5e-1; epsilon=1e-1 # (1.0438)
# N=int(1e4);  K=8; T0=5e-1; epsilon=2e-1 # (1.0352)
# N=int(1e4);  K=8; T0=5e-1; epsilon=5e-1 # (1.0776)
# N=int(1e4);  K=8; T0=1;    epsilon=2e-1 # (1.0476)
# N=int(1e4); K=10; T0=5e-1; epsilon=2e-1 # (1.0352)
# N=int(1e4);  K=8; T0=2;    epsilon=2e-1 # (1.0751)
# N=int(1e4);  K=8; T0=0.2;  epsilon=2e-1 # (1.0256)
# N=int(1e5); K=16; T0=5e-1; epsilon=1e-1 # (1.0161)

# Initialize Main Loop

np.random.seed(0); P=5; X=np.random.normal(0,1,[2,P])
fim=0; n=0; k=0; Jmin=J(X); Xmin=X; T=T0;
history_J=np.zeros([int(N*K),1]); history_T=np.zeros([int(N*K),1])

# Main Loop

while not(fim):
    T=T0/np.log2(2+k)
    Jcur = J(X)
    for n in range(0,N):
        Xhat=X+epsilon*np.random.normal(0,1,np.shape(X)) 
        Jhat = J(Xhat)
        if np.random.uniform()<np.exp((J(X)-J(Xhat))/T):
            X=Xhat
            if J(X)<Jmin:
                Jmin=J(X); Xmin=X;
        history_J[k*N+n]=J(X)
        history_T[k*N+n]=T;
    print([k, Jmin])
    k+=1
    if k==K: fim=1

print(Jmin)

plt.rc('font',size=16,weight='bold')

plt.figure
plt.subplot(311)
plt.plot(history_J)
plt.grid()
plt.subplot(312)
plt.plot(history_T)
plt.grid()
plt.subplot(313)
plt.plot(Xmin[0,:],Xmin[1,:],'k.',markersize=30)
plt.axis('equal')
plt.grid()

h,e=np.histogram(history_J[-N/5:],20);
c=(e[0:-1]+e[1:])/2;

plt.figure()
plt.plot(c,h/(N/5),'b.-',markersize=30)
plt.plot(c,np.exp(-c/T)/np.sum(np.exp(-c/T)),'r.-',markersize=30)
plt.grid()
plt.show()
