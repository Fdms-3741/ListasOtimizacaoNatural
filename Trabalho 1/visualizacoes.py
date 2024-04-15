import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt


def CriarFig(fig,ax):
    if fig is None:
        fig,ax = plt.subplots()
    elif ax is None:
        raise Exception("Tem que passar fig e ax juntos")
    return fig,ax

def PlotPontosTSP(posicoes,fig=None,ax=None):
    
    fig,ax = CriarFig(fig,ax)

    ax.scatter(posicoes[0,:],posicoes[1,:])

    return fig, ax

def PlotResultadoSA(x,posicoes,fig=None,ax=None):
    fig,ax = CriarFig(fig,ax)
    
    indices = np.concatenate([x,[x[0]]])
    dados = posicoes[:,indices]
    ax.plot(dados[0,:],dados[1,:], '.r-',markersize=20) 
    
    return fig,ax


