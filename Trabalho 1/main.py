import time as tm 
import json 

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 

from visualizacoes import PlotPontosTSP, PlotResultadoSA
from generate_tsp import GerarProblemaTSP


def Pertubacao(x,epsilon=1):
    xnew = x.copy()
    for i in range(epsilon):
        origem = np.random.randint(0,x.shape[0])
        destino = origem
        while destino == origem:
            destino = np.random.randint(0,x.shape[0])
        xnew[origem],xnew[destino] = (xnew[destino],xnew[origem])
    return xnew

def Custo(x,posicoes):
    origem  = posicoes[:,x][:,:-1]
    destino = posicoes[:,x][:,1:]
    return np.sum(np.linalg.norm(destino-origem,axis=0))


def SimulatedAnnealingTSP(x,posicoes,Custo,Pertubacao,K=6,N=100000,T0=10,epsilon=1,passoGravacao=100):

    jx = Custo(x,posicoes)
    xmin = x
    jmin = jx

    jhist = np.zeros(K*N//passoGravacao)
    xhist = np.zeros((K*N//passoGravacao,*x.shape))
    # Tempos em diferentes partes do código
    mediaCusto = np.zeros(K*N//passoGravacao)
    mediaPertubacao = np.zeros(K*N//passoGravacao)
    mediaPasso = np.zeros(K*N//passoGravacao)
    mediaTemperatura = np.zeros(K)
    somaPertubacao = 0
    somaCusto = 0
    
    histTransicao = {}
    histTransicao['k'] = []
    histTransicao['jmin'] = []
    histTransicao['jx'] = []
    histTransicao['xmin'] = []
    histTransicao['x'] = []


    start_time = tm.time()
    for k in range(K):
        T = T0/np.log2(2+k)
        temperatura_inicio = tm.time()
        passo_inicio = tm.time()
        for n in range(N):
            # Pertuba e soma o tempo
            pertubacao_inicio = tm.time()
            xhat = Pertubacao(x,epsilon)
            somaPertubacao += tm.time() - pertubacao_inicio
            
            # Calcula o custo e soma o tempo
            custo_inicio = tm.time()
            jhat = Custo(x,posicoes)
            somaCusto += tm.time() - custo_inicio

            # Decisão de mudança de estado
            if np.random.uniform(0,1) < np.exp((jx-jhat)/T):
                x = xhat
                jx = jhat
                if jx < jmin:
                    xmin = x
                    jmin = jx

            # Histórico dos passos
            if (n%passoGravacao)==0:
                # Média do passo
                mediaPasso[(k*N+n)//passoGravacao] = (tm.time() - passo_inicio)/passoGravacao
                passo_inicio = tm.time()
                # Salva espaço e custo atual
                jhist[(k*N+n)//passoGravacao] = jx 
                xhist[(k*N+n)//passoGravacao] = x
                # Tempos médios nas funções custosas
                mediaCusto[(k*N+n)//passoGravacao] = somaCusto/passoGravacao
                mediaPertubacao[(k*N+n)//passoGravacao] = somaPertubacao/passoGravacao
        mediaTemperatura[k] = tm.time() - temperatura_inicio
        print(k,xhat,jhat)
        histTransicao['k'].append(k)
        histTransicao['jmin'].append(jmin)
        histTransicao['jx'].append(jx)
        histTransicao['xmin'].append(xmin.tolist())
        histTransicao['x'].append(x.tolist())

    end_time = tm.time()
            

    return xmin, jmin, jhist, xhist, end_time - start_time, np.mean(mediaCusto), np.mean(mediaPertubacao), np.mean(mediaPasso), np.mean(mediaTemperatura),histTransicao

def ExperimentoTSP(tamanho,K,N,T0,epsilon):
    posicoes = GerarProblemaTSP(tamanho,3)

    
    x0 = np.arange(tamanho)
    np.random.shuffle(x0)
    
    xmin,jmin,jhist,xhist,tempoTotal, mediaCusto, mediaPertubacao, mediaPasso, mediaTemperatura, histTransicao = SimulatedAnnealingTSP(x0,posicoes,Custo,Pertubacao,K,N,T0,epsilon)
    
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
    resultados = ExperimentoTSP(5,6,100,10,1)
    with open('resultado.json','w') as fil:
        json.dump(resultados,fil)
