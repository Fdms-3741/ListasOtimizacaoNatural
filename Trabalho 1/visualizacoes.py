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

if __name__ == "__main__":
    import pandas as pd 
    import sys

    resultados = pd.read_pickle(sys.argv[1])
    
    for key, item in resultados.T.items():
        posicoes = item['Posições']
        epsilon = item['$\epsilon$']
        params = f" (K={item['K']}; N={item['N']}; $T_0$={item['$T_0$']};" + " $\epsilon$=" + f'{epsilon})'
        PlotPontosTSP(posicoes)
        plt.title("Posições das cidades")
        plt.savefig("Posições.png")
        PlotResultadoSA(item['$X_0$'],posicoes)
        plt.title(f"Resultado inicial (J={item['$J_0$']})"+params)
        plt.savefig("Resultado inicial.png")
        PlotResultadoSA(item['X'],posicoes)
        plt.title(f"Resultado final (J={item['J']})"+params)
        plt.savefig("Resultado final.png")
        plt.close() 
        # Plot da evolução do algoritm
#        for elements in 

        # Plot da evolução ao longo do algoritmo
        for nome in ['Evolução J','Evolução T','Evolução $J_{min}$']:
            fig,ax = plt.subplots()
            print(item[nome].shape)
            ax.plot(item[nome][10000:])
            ax.set_title(nome + params)
            fig.savefig(nome+".png")
            plt.close() 

        transicoes = pd.DataFrame(item['Transições'])
        for key, itemzin in transicoes.T.items():
            PlotResultadoSA(itemzin['x'],posicoes)
            plt.title(f"Estado da solução em $T={item['$T_0$']/np.log(2+itemzin['k']):.2f}$ ($J={itemzin['jx']:.3f}$)")
            plt.savefig(f"resultado-{key}.png")
            plt.close() 


        print(f"Tempo de execução: {item['Total']}")
