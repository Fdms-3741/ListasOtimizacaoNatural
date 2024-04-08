import numpy as np

def DistribuicaoBoltzmann(FuncaoCusto,T):
    return lambda x : np.exp(-FuncaoCusto(x)/T)

def AlgoritmoMetropolisHastings(x0,FuncaoProbabilidade,N,epsilon,numerosSaida=1,passoSalvarProgressao=None,R=np.random.normal,Rargs=(0,1),FuncaoRejeicao=None):
    """
    Algoritmo de Metropolis modificado para ser um RNG de qualquer distribuição.
    Essa função tem as funcionalidades:
        * Escolha da quantidade de números de saída
        * Escolha do salvamento do progresso de custo e estados ao longo das iterações
        * Modificação do 
        * Definição da função de passo e seus argumentos
    Parâmetros:
        * x0: Estado inicial. Deve ser um array com as dimensões dos números de estados
        * FuncaoDensidadeProbabilidade: Função cuja densidade de probabilidade deve ser seguida.
            * Para seguir o algoritmo de metropolis original, recebe aqui a distribuição Boltzmann-Gibbs
        * N: Número de iterações **antes de começar a salvar os números**.
        * epsilon: Magnitude do passo aleatório dado
        * numerosSaida: Quantidade de números a serem gerados
        * passoSalvarProgressao: Quantidade de valores a serem salvos durante o processo de iteração
            * Amostra em x pontos distribuídos de forma igualmente espaçada entre 0 e N
        * R: Função de passo. Recebe um RNG que segue a distribuição para o passo da caminhada aleatória.
        * Rargs: Argumentos da função R
        * FuncaoRejeicao: 
    Retorna:
        * xOut: os <numerosSaida> valores de x
        * jOut: a probabilidade desses números aparecerem
        * xMin: Melhor resultado encontrado ao longo de todo o processo
        * jMin: Custo do melhor resultado
        * xProg: Os estados ao longo das iterações 
        * jProg: Os custos respectivos
    """
    # Definições de variáveis inicia
    x = x0.copy()
    xMin = x.copy()
    if not type(x) is np.ndarray:
        raise Exception("x0 deve ser do tipo np.array")
    
    j = FuncaoProbabilidade(x0)
    jMin = j
    if not np.isscalar(j):
        raise Exception(f"A PDF deve retornar um valor escalar (python ou numpy (retornou {type(j)})")
    
    # Inicialização de variáveis de saída
    xOut = np.zeros((numerosSaida,*x.shape))
    jOut = np.zeros(numerosSaida)
    
    indiceSalvamento = N - numerosSaida

    # Inicialização de variáveis de progresso
    xProg,jProg = (None, None)
    if passoSalvarProgressao:
        valorEfetivo = int(N/np.floor(N/passoSalvarProgressao))
        xProg = np.zeros((valorEfetivo,*x.shape))
        jProg = np.zeros((valorEfetivo))
        indiceSalvar = 0

    # -----------------------------------------------------
    #
    #       LOOP PRINCIPAL DO ALGORITMO DE METROPOLIS
    #
    # ----------------------------------------------------
    for n in range(N+numerosSaida):
        xhat = x + epsilon*(R(*Rargs,size=x.shape))
        # Se função é definida, pula se a função rejeita o estado atual
        if FuncaoRejeicao and FuncaoRejeicao(xhat):
            continue
        jhat = FuncaoProbabilidade(xhat)
        sorteio = np.random.uniform(0,1)
        # Se a prob de ir pro estado novo é maior que ir pro estado atual, 
        # ele sempre vai para o estado novo.
        if sorteio < np.min([1,(jhat/j)]): 
            x = xhat.copy()
            j = jhat.copy()
    # -----------------------------------------------------
    # -----------------------------------------------------
            # Salva se o valor encontrado for o menor
            if j < jMin:
                xMin = x.copy()
                jMin = j
            
        # Salva o progresso a cada múltiplo de iterações
        if passoSalvarProgressao and (n % np.floor(N/passoSalvarProgressao)) == 0 and indiceSalvar < xProg.shape[0]:
            xProg[indiceSalvar] = x
            jProg[indiceSalvar] = j
            indiceSalvar += 1
        
        # Salva os numeroSaida números finais
        if n >= N:
            xOut[n-N] = x.copy()
            jOut[n-N] = j.copy()

    results = {
        'xOut':xOut,
        'jOut':jOut,
        'xMin':xMin,
        'jMin':jMin,
        'xProg':xProg,
        'jProg':jProg
    }
    return results

if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    import scipy.stats as stats
    import matplotlib.pyplot as plt

