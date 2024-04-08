
template<T>
class Metropolis {
    
    public:
        // Construtor
        Metropolis((double *)(T*), (void)(T*), T, double);
        // Realiza N iterações do algoritmo sem retornar resultados
        void Iterate(int);
        // Realiza N iteracoes do algoritmo e as escreve sequencialmente em T*
        void Generate(int,T*);

    private:
        // Ponteiro pra função de custo
        (double *CostFunction)(T*);
        // Ponteiro pra função de pertubação
        (void *DisturbFunction)(T*,double);
        // Estado atual
        T CurrentState;
        double CurrentCost;
        // Valor da pertubação
        double epsilon;
        // Variáveis auxiliares para a geração de números aleatórios
        std::random_device rd;
        std::mt19937 gen;
        std::uniform_real_distribution<double> uniform;
}
