# sa_tsp.py 
# Aluno: Fernando Dias 
#
# Esse código tem a implementação do simulated annealing para resolver o problema do caixeiro viajante
# 
# 
import time as tm 

import numpy as np

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


def SimulatedAnnealingTSP(x,posicoes,K=6,N=100000,T0=10,epsilon=1,resultado=None):
    
    passoGravacao = N //1000
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

    parada = False
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
                    if resultado and np.abs(jmin-resultado) < 1e-10:
                        parada = True
                        break

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
        print(k,xmin,jmin)
        histTransicao['k'].append(k)
        histTransicao['jmin'].append(jmin)
        histTransicao['jx'].append(jx)
        histTransicao['xmin'].append(xmin.tolist())
        histTransicao['x'].append(x.tolist())
        if parada:
            break
    end_time = tm.time()
            
    parada = 'Solução encontrada' if parada else "Fim da execução"
    return xmin, jmin, jhist, xhist, end_time - start_time, np.mean(mediaCusto), np.mean(mediaPertubacao), np.mean(mediaPasso), np.mean(mediaTemperatura),histTransicao, parada


