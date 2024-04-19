# sa_tsp.py 
# Aluno: Fernando Dias 
#
# Esse código tem a implementação do simulated annealing para resolver o problema do caixeiro viajante
# 
# 
import time as tm 
import numpy as np


def PertubacaoSwitch(x,epsilon=1):
    xnew = x.copy()
    numPosicoes = np.ceil(np.abs(np.random.normal(0,epsilon))).astype(np.int64)
    #numPosicoes = 1
    for i in range(numPosicoes):
        origem = np.random.randint(0,x.shape[0])
        destino = origem
        while destino == origem:
            destino = np.random.randint(0,x.shape[0])
        xnew[origem],xnew[destino] = (xnew[destino],xnew[origem])
    assert xnew.shape == x.shape, f"Falta de elementos. {xnew.shape} e {x.shape}" 
    assert np.unique(xnew).shape[0] == x.shape[0], f"Elementos repetidos: {xnew}"
    return xnew

def PertubacaoFlip(x,epsilon=1):
    tamanho = np.min([x.shape[0],1+np.ceil(np.abs(np.random.normal(0,epsilon))).astype(np.int64)])
    inicio = np.random.randint(x.shape[0]-tamanho) if tamanho < x.shape[0] else 0
    fim = inicio+tamanho
    xnew = x.copy()
    xnew[inicio:fim] = np.flip(xnew[inicio:fim])
    assert xnew.shape == x.shape, f"Falta de elementos. {xnew.shape} e {x.shape}" 
    assert np.unique(xnew).shape[0] == x.shape[0], f"Elementos repetidos: {xnew}"
    return xnew

def PertubacaoCut(x,epsilon=1):
    xnew = x.copy()
    # Região de corte 
    localCorte = np.random.randint(x.shape[0]-1)
    tamanho = np.min([x.shape[0]-localCorte,np.ceil(np.abs(np.random.normal(0,epsilon))).astype(np.int64)])
    xcut = xnew[localCorte:localCorte+tamanho]
    # Região de inserção
    xnew = np.concatenate([xnew[:localCorte],xnew[localCorte+tamanho:]])
    novaPos = np.random.randint(xnew.shape[0]-1)
    xnew = np.concatenate([xnew[:novaPos],xcut,xnew[novaPos:]])
    assert xnew.shape == x.shape, f"Falta de elementos. {xnew.shape} e {x.shape}" 
    assert np.unique(xnew).shape[0] == x.shape[0], f"Elementos repetidos: {xnew}"
    return xnew 

def PertubacaoSwitchNeighbor(x,epsilon=1):
    xnew = x.copy()
    pos = np.random.randint(x.shape[0]-1)
    xnew[pos],xnew[pos+1] = xnew[pos+1],xnew[pos]
    assert xnew.shape == x.shape, f"Falta de elementos. {xnew.shape} e {x.shape}" 
    assert np.unique(xnew).shape[0] == x.shape[0], f"Elementos repetidos: {xnew}"
    return xnew

def PertubacaoLin(x,epsilon=1):
    xnew = x.copy()
    if np.random.randint(0,2) == 0:
        xnew = PertubacaoFlip(xnew)
    else: 
        xnew = PertubacaoCut(xnew)
    assert xnew.shape == x.shape, f"Falta de elementos. {xnew.shape} e {x.shape}" 
    assert np.unique(xnew).shape[0] == x.shape[0], f"Elementos repetidos: {xnew}"
    return xnew 

Pertubacao = PertubacaoLin

def Custo(x,posicoes):
    origem  = posicoes[:,x][:,:-1]
    destino = posicoes[:,x][:,1:]
    return np.sum(np.linalg.norm(destino-origem,axis=0))


def SimulatedAnnealingTSP(x,posicoes,K=6,N=100000,T0=10,epsilon=1,resultado=None,Custo=Custo,Pertubacao=Pertubacao):
    
    passoGravacao = N //1000
    jx = Custo(x,posicoes)
    xmin = x.copy()
    jmin = jx.copy()

    jhist = np.zeros(K*N)
    jminhist = np.zeros(K*N)
    thist = np.zeros(K*N)
    exphist = np.zeros(K*N)

    histTransicao = {}
    histTransicao['k'] = []
    histTransicao['jmin'] = []
    histTransicao['jx'] = []
    histTransicao['xmin'] = []
    histTransicao['x'] = []
    
    parada = False
    semMudanca = 0 
    start_time = tm.time()
    # 
    for k in range(K):
        T = T0/np.log2(2+k)
        #T = (0.99**k)*T0    
        for n in range(N):
            # Pertuba e soma o tempo
            xhat = Pertubacao(x,epsilon)
            # Calcula o custo e soma o tempo
            jhat = Custo(xhat,posicoes)
            aceitacao = np.min([1.00001, np.exp((jx-jhat)/T)])
            # Decisão de mudança de estado
            if np.random.uniform(0,1) < aceitacao:
                semMudanca = 0
                x = xhat
                jx = jhat
                if jx < jmin:
                    xmin = x
                    jmin = jx
                    if resultado and np.abs(jmin-resultado) < 1e-10:
                        parada = "Ótimo encontrado"
                        break
            else:
                semMudanca += 1
                #if semMudanca == 100000: # Sistema congelado
                #    parada = True
                #    break

            jhist[(k*N+n)] = jx
            jminhist[(k*N+n)] = jmin
            thist[(k*N+n)] = T
            exphist[(k*N+n)] = aceitacao

        print(k,T,xmin,jmin)
        histTransicao['k'].append(k)
        histTransicao['jmin'].append(jmin)
        histTransicao['jx'].append(jx)
        histTransicao['xmin'].append(xmin.tolist())
        histTransicao['x'].append(x.tolist())
        
        if parada:
            break
    end_time = tm.time()
            
    if not parada:
        parada = "Fim da execução"

    return xmin, jmin, jhist, thist, jminhist, exphist, end_time - start_time, histTransicao, parada
