#include <random>
#include "SimulatedAnnealing.hpp"

template<T>
Metropolis::Metropolis(double *CostFunction(T*),void *DisturbFunction(T*),double epsilon) :
    CostFunction(CostFunction), 
    DisturbFunction(DisturbFunction), 
    epsilon(epsilon),
    gen(rd()),
    uniform(0,1){};

void Metropolis::Iterate(int N){
    int n;
    double sorteio, candidateCost;
    T candidate;

    for (n=0; n<N; n++){
        candidate = DisturbFunction(this->CurrentState);
        candidateCost;
        sorteio = uniform(gen);
        if (sorteio < candidateCost/currentCost){
            currentState = candidate;
            currentCost = candidateCost;
        }
    }
};

void Metropolis::Generate(int N, T*){
    
}
