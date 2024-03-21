import numpy as np

np.random.seed(0)
N=int(1e6)
x=np.random.uniform(0,1,(2,N))*2-1
l=np.sum(np.power(x,2),axis=0)
resultado_uniforme=np.sum(l*np.exp(-l))/N*4

epsilon=5e-1; X=np.zeros([2,1]); historicoX=np.zeros([2,N])
for n in range(0,N):
    Xhat=X+epsilon*np.random.normal(0,1,np.shape(X))
    if (np.abs(Xhat[0])>1)or(np.abs(Xhat[1])>1):
        if Xhat[0]>1: Xhat[0]=Xhat[0]-2
        if Xhat[0]<-1: Xhat[0]=Xhat[0]+2
        if Xhat[1]>1: Xhat[1]=Xhat[1]-2
        if Xhat[1]<-1: Xhat[1]=Xhat[1]+2
    if np.random.uniform()<np.exp(np.sum(np.power(X,2))-np.sum(np.power(Xhat,2))):
        X=Xhat;
    historicoX[0,n]=X[0]
    historicoX[1,n]=X[1]
    if np.remainder(n+1,np.round(N/10))==0:
        print(round((n+1)/N*100))
        
D=historicoX[:,int(0.8*N):]
resultado_metropolis=np.mean(np.sum(np.power(D,2),axis=0))

x=np.random.uniform(0,1,(2,N))*2-1
l=np.sum(np.power(x,2),axis=0)
fator_Z=np.sum(np.exp(-l))/N*4

print([resultado_metropolis,fator_Z,resultado_metropolis*fator_Z,resultado_uniforme])
# [0.5075226283761246, 2.230594404146958, 1.1320771348337397, 1.1324600788802643]
