"""!@package docstring 
sga.py Implementation of genetic algorithms
"""

import numpy as np 
import pandas as pd 

from ea import EvolutionaryAlgorithm
from abc import ABC, abstractmethod 

class GeneticAlgorithm(EvolutionaryAlgorithm):

    """! Class that dictates the behaviour of a genetic algorithm"""
    def __init__(self,parameters):
        super().__init__()
        self.sgaParams = ['populationSize','bitsSize','crossoverProbability','mutationProbability','numberOfSelectedParents','numberOfRecombinedOffspring']
        self.AddParameters(self.sgaParams,parameters)   
        self.GenerateIndividuals()

    def SaveOptimumValue(self,aptitudeList,children):
        if (currentAptitude := np.max(aptitudeList)) > self.bestFitAptitude:
            self.bestFitAptitude = currentAptitude
            self.bestFitIndividual = children[np.argmax(aptitudeList)]

    def GenerateIndividuals(self):
        self.individuals = np.random.choice([1,0],size=(self.populationSize,self.bitsSize))

    def ParentSelection(self, aptitudeList):
        """Selects from the aptitude list based on their probability."""
        # Calculates probability based on aptitude results
        aptitudeProbabilityTerms = (aptitudeList-(np.min([np.min(aptitudeList),0]))+0.1)
        aptitudeProbability = aptitudeProbabilityTerms/np.sum(aptitudeProbabilityTerms)
        print(aptitudeProbability)
        selectionIndex = np.random.choice(aptitudeList.shape[0],(self.numberOfSelectedParents,),p=aptitudeProbability,replace=False)
        # Assert that selection matches the number of parents
        assert self.numberOfSelectedParents <= self.populationSize
        assert selectionIndex.shape[0] == self.numberOfSelectedParents
        return selectionIndex


    def Recombination(self,individuals):
        pc = self.crossoverProbability
        newIndividuals = individuals.copy()
        # Selects a particular number of offsprings to emerge
        recombinationChoice = np.concatenate([
                np.ones(self.numberOfRecombinedOffspring//2),
                np.zeros(individuals.shape[0]**2-self.numberOfRecombinedOffspring//2)
            ])
        np.random.shuffle(recombinationChoice)
        recombinationChoice = recombinationChoice.reshape(individuals.shape[0],individuals.shape[0]).astype(np.int64)
        for (parent1,parent2) in np.argwhere(recombinationChoice):
            sliceSize = int(np.ceil(np.abs(np.random.normal(0,1))))
            sliceStart = int(np.random.randint(0,individuals.shape[0]-sliceSize))
            sliceEnd = sliceStart+sliceSize
            if np.random.uniform(0,1) < pc:
                newIndividuals[parent1,sliceStart:sliceEnd] = individuals[parent2,sliceStart:sliceEnd]
                newIndividuals[parent2,sliceStart:sliceEnd] = individuals[parent1,sliceStart:sliceEnd]
        return newIndividuals

    def Mutation(self,individuals):
        pm = self.mutationProbability
        mutationChoice = np.random.choice([1,0],p=[pm,1-pm],size=(individuals.shape[0],1))
        bitInversionProbability = np.random.choice([1,0],p=[pm,1-pm],size=individuals.shape)
        mutationMatrix = mutationChoice & bitInversionProbability
        return individuals ^ mutationMatrix

    def Replacement(self,aptitudeList):
        aptitudeSort = pd.Series(aptitudeList)
        aptitudeSort = aptitudeSort.sort_values(ascending=False)
        return aptitudeSort.index[:self.populationSize]

if __name__ == "__main__":
    individuals = np.eye(8).astype(np.int64)
    aptitudeList = np.array([0.2,0.4,0.3,0.8,0.1])
    a = GeneticAlgorithm()
    print(aptitudeList)
    print("Generate individuals")
    print(newInd := a.GenerateIndividuals(10,8))
    print("Parent selection")
    print(a.ParentSelection(aptitudeList))
    print("Recombination")
    print(a.Recombination(individuals,0.4))
    print("Mutation")
    print(a.Mutation(individuals,0.4))
    print("Replacement")
    a.populationSize = 3 
    print(a.Replacement(aptitudeList))
