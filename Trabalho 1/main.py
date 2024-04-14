from pathlib import Path
import time as tm 
import json 

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 

from visualizacoes import PlotPontosTSP, PlotResultadoSA
from generate_tsp import GerarProblemaTSP
from sa_tsp import SimulatedAnnealingTSP, Custo

def ExperimentoTSP(tamanho,K,N,T0,epsilon):

    # Gera as posições do problema
    posicoes = GerarProblemaTSP(tamanho,3)

    # Calcula solução e custo da solução
    x0 = np.arange(tamanho)
    valorOtimo = Custo(x0,posicoes)

    np.random.shuffle(x0)
    
    xmin,jmin,jhist,xhist,tempoTotal, mediaCusto, mediaPertubacao, mediaPasso, mediaTemperatura, histTransicao, parada = SimulatedAnnealingTSP(x0,posicoes,K,N,T0,epsilon,valorOtimo)
    
    resultados = {}
    resultados['Parâmetros'] = {}
    resultados['Parâmetros']['Número de cidades'] = tamanho
    resultados['Parâmetros']['K'] = K
    resultados['Parâmetros']['N'] = N 
    resultados['Parâmetros']['T0'] = T0 
    resultados['Parâmetros']['epsilon'] = epsilon

    resultados['Resultados'] = {}
    resultados['Resultados']['X'] = xmin.tolist()
    resultados['Resultados']['J'] = jmin
    resultados['Resultados']['Condição de parada'] = parada
    resultados['Resultados']['Tempos'] = {}
    resultados['Resultados']['Tempos']['Total'] = tempoTotal 
    resultados['Resultados']['Tempos']['Custo'] = mediaCusto
    resultados['Resultados']['Tempos']['Pertubação'] = mediaPertubacao
    resultados['Resultados']['Tempos']['Passo'] = mediaPasso
    resultados['Resultados']['Tempos']['Temperatura'] = mediaTemperatura

    
    resultados['Histórico'] = {}
    resultados['Histórico']['X inicial'] = x0.tolist()
    resultados['Histórico']['J'] = jhist.tolist()
    resultados['Histórico']['X'] = xhist.tolist()
    resultados['Histórico']['Transições'] = histTransicao
    
    return resultados

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

