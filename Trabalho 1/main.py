import json 
import time as tm 
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 

# Funções para o Simulated Annealing
from sa_tsp import SimulatedAnnealingTSP, Custo, PertubacaoSwitch, PertubacaoLin
# Geradores de problemas com soluções conhecidas
from generate_tsp import GerarProblemaRadialTSP, GerarProblemaRetangularTSP
# (Falho) Tentativa de encontrar temperatura
from find_temperature import EncontrarTemperatura
# Gerador de problemas com base na lista do TSPLIB
from tsplib_problems import GerarProblemaTSPLIB


def IterarExperimentos(posicoes,Ks,Ns,T0s,epsilons,valoresOtimos=[None],gaps=[0.99999999],Pertubacoes=[PertubacaoLin]):
    """
    Itera entre inúmeros parâmetros para tentar solucionar o problema
    """
    resultados = {}
    posicoes = [posicoes] if type(posicoes)!=list else posicoes
    # Conversão pra listas
    Ks = [Ks] if type(Ks)!=list else Ks
    Ns = [Ns] if type(Ns)!=list else Ns
    gaps = [gaps] if type(gaps)!=list else gaps
    T0s = [T0s] if type(T0s)!=list else T0s
    epsilons = [epsilons] if type(epsilons)!=list else epsilons
    valoresOtimos = [valoresOtimos] if type(valoresOtimos)!=list else valoresOtimos
    Pertubacoes = [Pertubacoes] if type(Pertubacoes)!=list else Pertubacoes
    # Iteração entre diferentes experimentos
    for posicao, K, N, T0, epsilon, valorOtimo, gap, Pertubacao in product(posicoes,Ks,Ns,T0s,epsilons,valoresOtimos,gaps,Pertubacoes):
        print(f"Expr: K={K}; N={N}; T0={T0}; epsilon={epsilon}, posicao={posicao.shape}, valorOtimo={valorOtimo},gap={gap} Pertubacao={Pertubacao}")
        # Execução do experimento
        resultado = ExperimentoTSP(posicao,K,N,T0,epsilon,valorOtimo,gap=gap,Pertubacao=Pertubacao)
        # Concatenação do resultado
        for key in resultado.keys():
            if key in resultados.keys():
                resultados[key].append(resultado[key])
            else:
                resultados[key] = [resultado[key]]
    
    return resultados

def ExperimentoTSP(posicoes,K,N,T0,epsilon,valorOtimo=None,gap=0.999999,Pertubacao=PertubacaoLin):
    # Calcula solução e custo da solução
    x0 = np.arange(posicoes.shape[1])
    j0 = Custo(x0,posicoes)
    
    # Escolhe vetor aleatório
    np.random.shuffle(x0)
    
    # Salva resultados
    xmin, jmin, jhist, thist, jminhist, exphist, tempoTotal, histTransicao, parada, tempos = SimulatedAnnealingTSP(x0,posicoes,K,N,T0,epsilon,valorOtimo,gap=gap,Pertubacao=Pertubacao)
    
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
    resultados['Tempo total'] = tempoTotal 

    resultados['$X_0$'] = x0
    resultados['$J_0$'] = j0
    resultados['Evolução J'] = jhist
    resultados['Evolução T'] = thist
    resultados['Evolução $J_{min}$'] = jminhist
    resultados['Evolução aceitação'] = exphist
    resultados['Transições'] = histTransicao
    
    resultados['Gap'] = jmin/valorOtimo if valorOtimo else None

    return resultados

def ExperimentoRadialTSP(tamanho,K,N,T0,epsilon,gap=0.99999,Pertubacao=PertubacaoLin):
    posicoes = GerarProblemaRadialTSP(tamanho)
    valorOtimo = Custo(np.arange(posicoes.shape[1]),posicoes)
    resultados = IterarExperimentos([posicoes],K,N,T0,epsilon,valoresOtimos=valorOtimo,gaps=gap,Pertubacoes=Pertubacao)
    return resultados

def ExperimentoRetangularTSP(ladoCentros, cidades, K,N,T0,epsilon,gap=0.99999):
    posicoes = GerarProblemaRetangularTSP(ladoCentros, cidades)
    valorOtimo = None 
    resultados = IterarExperimentos([posicoes],K,N,T0,epsilon,valoresOtimos=valorOtimo,gaps=gap)
    return resultados


if __name__ == "__main__":
    
    ladoCentros = 4
    cidadesCentros = 3
    tamanho = (ladoCentros**2)*cidadesCentros
    K = 40
    N = 10**3
    T0 = 1
    epsilon = 1

    resultados = ExperimentoRetangularTSP(ladoCentros,cidadesCentros, K, N, T0, epsilon)
    #resultados = ExperimentoRadialTSP(tamanho, K, N, T0, epsilon,PertubacaoLin)
    resultados = pd.DataFrame(resultados)
    
    # Salva os resultados 
    resultados.to_pickle(f'resultados.pickle')
