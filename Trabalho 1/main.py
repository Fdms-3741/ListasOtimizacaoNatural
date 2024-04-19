import json 
import time as tm 
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 

from sa_tsp import SimulatedAnnealingTSP, Custo, PertubacaoSwitch, PertubacaoLin
from visualizacoes import PlotPontosTSP, PlotResultadoSA
from generate_tsp import GerarProblemaRadialTSP, GerarProblemaRetangularTSP
from find_temperature import EncontrarTemperatura

def IterarExperimentos(posicoes,K,N,T0,epsilon,valorOtimo=[None],Pertubacoes=[PertubacaoLin]):
    resultados = {}
    posicoes = [posicoes] if type(posicoes)!=list else posicoes
    # Conversão pra listas
    K = [K] if type(K)!=list else K
    N = [N] if type(N)!=list else N
    T0 = [T0] if type(T0)!=list else T0
    epsilon = [epsilon] if type(epsilon)!=list else epsilon
    valorOtimo = [valorOtimo] if type(valorOtimo)!=list else valorOtimo
    Pertubacoes = [Pertubacoes] if type(Pertubacoes)!=list else Pertubacoes
    # Iteração entre diferentes experimentos
    for posicao, K, N, T0, epsilon, valorOtimo, Pertubacao in product(posicoes,K,N,T0,epsilon,valorOtimo,Pertubacoes):
        print(f"Expr: K={K}; N={N}; T0={T0}; epsilon={epsilon}, posicao={posicao.shape}, valorOtimo={valorOtimo}, Pertubacao={Pertubacao}")
        # Execução do experimento
        resultado = ExperimentoTSP(posicao,K,N,T0,epsilon,valorOtimo,Pertubacao=Pertubacao)
        # Concatenação do resultado
        for key in resultado.keys():
            if key in resultados.keys():
                resultados[key].append(resultado[key])
            else:
                resultados[key] = [resultado[key]]
    
    return resultados

def ExperimentoTSP(posicoes,K,N,T0,epsilon,valorOtimo=None,Pertubacao=PertubacaoLin):
    # Calcula solução e custo da solução
    x0 = np.arange(posicoes.shape[1])
    j0 = Custo(x0,posicoes)

    np.random.shuffle(x0)
    
    xmin,jmin,jhist,thist,jminhist, exphist, tempoTotal, histTransicao, parada = SimulatedAnnealingTSP(x0,posicoes,K,N,T0,epsilon,valorOtimo,Pertubacao=Pertubacao)
    
    resultados = {}
    resultados['K'] = K
    resultados['N'] = N 
    resultados['$T_0$'] = T0 
    resultados['$\epsilon$'] = epsilon
    resultados['Posições'] = posicoes
    resultados['Valor ótimo'] = valorOtimo
    resultados['X'] = xmin
    resultados['J'] = jmin
    resultados['Condição de parada'] = parada
    resultados['Total'] = tempoTotal 

    resultados['$X_0$'] = x0
    resultados['$J_0$'] = j0
    resultados['Evolução J'] = jhist
    resultados['Evolução T'] = thist
    resultados['Evolução $J_{min}$'] = jminhist
    resultados['Evolução aceitação'] = exphist
    resultados['Transições'] = histTransicao
    
    return resultados

def ExperimentoRadialTSP(tamanho,K,N,T0,epsilon,Pertubacao=PertubacaoLin):
    posicoes = GerarProblemaRadialTSP(tamanho)
    valorOtimo = Custo(np.arange(posicoes.shape[1]),posicoes)
    resultados = IterarExperimentos([posicoes],K,N,T0,epsilon,valorOtimo=valorOtimo,Pertubacoes=Pertubacao)
    return resultados

def ExperimentoRetangularTSP(ladoCentros, cidades, K,N,T0,epsilon):
    posicoes = GerarProblemaRetangularTSP(ladoCentros, cidades)
    valorOtimo = None 
    resultados = IterarExperimentos([posicoes],K,N,T0,epsilon,valorOtimo=valorOtimo)

    return resultados


if __name__ == "__main__":
    
    ladoCentros = 4
    cidadesCentros = 3
    tamanho = (ladoCentros**2)*cidadesCentros
    K = 40
    N = 10**4
    T0 = 1
    epsilon = 1

    resultados = ExperimentoRetangularTSP(ladoCentros,cidadesCentros, K, N, T0, epsilon)
    #resultados = ExperimentoRadialTSP(tamanho, K, N, T0, epsilon,PertubacaoLin)
    resultados = pd.DataFrame(resultados)
    
    # Salva os resultados 
    resultados.to_pickle(f'resultados.pickle')
