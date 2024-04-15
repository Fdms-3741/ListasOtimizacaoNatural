from pathlib import Path
import time as tm 
import json 

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 

from visualizacoes import PlotPontosTSP, PlotResultadoSA
from generate_tsp import GerarProblemaRadialTSP, GerarProblemaRetangularTSP
from sa_tsp import SimulatedAnnealingTSP, Custo

def ExperimentoTSP(posicoes,K,N,T0,epsilon,valorOtimo=None):

    # Calcula solução e custo da solução
    x0 = np.arange(posicoes.shape[1])

    np.random.shuffle(x0)
    
    xmin,jmin,jhist,thist,tempoTotal, histTransicao, parada = SimulatedAnnealingTSP(x0,posicoes,K,N,T0,epsilon,valorOtimo)
    
    resultados = {}
    resultados['Parâmetros'] = {}
    resultados['Parâmetros']['K'] = K
    resultados['Parâmetros']['N'] = N 
    resultados['Parâmetros']['T0'] = T0 
    resultados['Parâmetros']['epsilon'] = epsilon
    resultados['Parâmetros']['Posições'] = posicoes

    resultados['Resultados'] = {}
    resultados['Resultados']['X'] = xmin
    resultados['Resultados']['J'] = jmin
    resultados['Resultados']['Condição de parada'] = parada
    resultados['Resultados']['Tempos'] = {}
    resultados['Resultados']['Tempos']['Total'] = tempoTotal 

    resultados['Histórico'] = {}
    resultados['Histórico']['X inicial'] = x0
    resultados['Histórico']['J'] = jhist
    resultados['Histórico']['T'] = thist
    resultados['Histórico']['Transições'] = histTransicao
    
    return resultados

def ExperimentoRadialTSP(tamanho,K,N,T0,epsilon):
    posicoes = GerarProblemaRadialTSP(tamanho)
    valorOtimo = Custo(np.arange(posicoes.shape[1]),posicoes)
    resultado = ExperimentoTSP(posicoes, K, N, T0, epsilon, valorOtimo)
    resultado['Histórico']['Valor ótimo'] = valorOtimo
    return resultado 

def ExperimentoRetangularTSP(ladoCentros, cidades, K,N,T0,epsilon):
    posicoes = GerarProblemaRetangularTSP(ladoCentros, cidades)
    resultado =  ExperimentoTSP(posicoes, K, N, T0, epsilon)
    resultado['Histórico']['Valor ótimo'] = None
    return resultado

if __name__ == "__main__":
    
    ladoCentros = 4
    cidadesCentros = 3
    tamanho = (ladoCentros**2)*cidadesCentros
    K = 40
    N = 10**5
    T0 = 30
    epsilon = 1

    #resultados = ExperimentoRetangularTSP(ladoCentros,cidadesCentros, K, N, T0, epsilon)
    resultados = ExperimentoRadialTSP(tamanho, K, N, T0, epsilon)
    
    # Salva resultados 
    #arquivoResultado = Path('resultado.json')
    #with arquivoResultado.open('w') as fil:
    #    json.dump(resultados,fil)

    # Mostrar resultados 
    posicoes =  np.array(resultados['Parâmetros']['Posições'])
    PlotPontosTSP(posicoes)
    plt.show()
    PlotResultadoSA(np.array(resultados['Histórico']['X inicial']),posicoes)
    plt.show()
    PlotResultadoSA(np.array(resultados['Resultados']['X']),posicoes)
    plt.show()
    fig,ax = plt.subplots(1,2)
    ax[0].plot(resultados['Histórico']['J'][:np.argwhere(np.abs(resultados['Histórico']['J'])<1e-10)[0][0]])
    ax[1].plot(resultados['Histórico']['T'][:np.argwhere(np.abs(resultados['Histórico']['J'])<1e-10)[0][0]])
    plt.show()
    print('Custo mínimo:', resultados['Histórico']['Valor ótimo']) 
    print('Custo inicial:', Custo(resultados['Histórico']['X inicial'],posicoes)) 
    print('Custo calculado:', Custo(resultados['Resultados']['X'],posicoes)) 
    print('Custo obtido:', resultados['Resultados']['J']) 
    exit()

if __name__ == "__main__":
    
    with open('experimentos.json') as fil:
        experiments = json.load(fil)
    
    inicio = np.ceil(tm.time())
    idx = -1

    for tamanho,N,T,K,epsilon in experiments:
        idx += 1
        arquivoResultado = Path(f'resultado_{idx}.json') 
        if arquivoResultado.exists():
            continue
        print(tamanho,N,T,K,epsilon)

        resultados = ExperimentoTSP(tamanho,K,N,T,epsilon)
        resultados['Descrição'] = {}
        resultados['Descrição']['Data/Hora'] = inicio
        resultados['Descrição']['indice'] = idx 
        
        with arquivoResultado.open('w') as fil:
            json.dump(resultados,fil)

