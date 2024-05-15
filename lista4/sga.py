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
        assert self.numberOfRecombinedOffspring <= self.populationSize**2
        assert self.numberOfSelectedParents <= self.populationSize
        assert self.crossoverProbability <= 1
        assert self.crossoverProbability >= 0
        assert self.mutationProbability <= 1
        assert self.mutationProbability >= 0


    def SaveOptimumValue(self,aptitudeList,children):
        if (currentAptitude := np.max(aptitudeList)) > self.bestFitAptitude:
            self.bestFitAptitude = currentAptitude
            self.bestFitIndividual = children[np.argmax(aptitudeList)]

    def GenerateIndividuals(self):
        self.individuals = np.random.choice([1,0],size=(self.populationSize,self.bitsSize))

    def ParentSelection(self, individuals):
        """Selects from the aptitude list based on their probability."""
        aptitudeList = self.CalculateAptitude(individuals)
        # Calculates probability based on aptitude results
        aptitudeProbabilityTerms = (aptitudeList-(np.min([np.min(aptitudeList),0]))+0.1)
        aptitudeProbability = aptitudeProbabilityTerms/np.sum(aptitudeProbabilityTerms)
        selectionIndex = np.random.choice(aptitudeList.shape[0],(self.numberOfSelectedParents,),p=aptitudeProbability,replace=False)
        # Assert that selection matches the number of parents
        return selectionIndex.astype(np.int64)


    def Recombination(self,individuals):
        pc = self.crossoverProbability
        resultIndividuals = np.zeros((self.numberOfRecombinedOffspring,self.bitsSize))
        # Selects a particular number of offsprings to emerge
        recombinationChoice = np.concatenate([
                np.ones(self.numberOfRecombinedOffspring//2),
                np.zeros(individuals.shape[0]**2-self.numberOfRecombinedOffspring//2)
            ])
        np.random.shuffle(recombinationChoice)
        recombinationChoice = recombinationChoice.reshape(individuals.shape[0],individuals.shape[0]).astype(np.int64)
        for index,(parent1,parent2) in enumerate(np.argwhere(recombinationChoice)):
            resultIndividuals[2*index:2*index+2,:] = self.RecombineParents(individuals[parent1],individuals[parent2])

        return resultIndividuals.astype(np.int64)
    
    def RecombineParents(self,parent1,parent2):
        resultIndividuals = np.zeros((2,self.bitsSize))

        sliceSize = int(np.ceil(np.abs(np.random.normal(0,1))))
        sliceStart = int(np.random.randint(0,parent1.shape[0]-sliceSize))
        sliceEnd = sliceStart+sliceSize
        
        pc = self.crossoverProbability
        # Parent definition
        resultIndividuals[0] = parent1
        resultIndividuals[1] = parent2
        if np.random.uniform(0,1) < pc:
            resultIndividuals[0,sliceStart:sliceEnd] = parent2[sliceStart:sliceEnd]
            resultIndividuals[1,sliceStart:sliceEnd] = parent1[sliceStart:sliceEnd]
        return resultIndividuals

    def MutateIndividual(self,individual):
        pm = self.mutationProbability
        bitInversionMap = np.random.choice([1,0],p=[pm,1-pm],size=individual.shape)
        return individual ^ bitInversionMap

    def Mutation(self,individuals):
        pm = self.mutationProbability
        mutationChoice = np.random.choice([1,0],p=[pm,1-pm],size=(individuals.shape[0],1))
        for individualIndex in np.argwhere(mutationChoice):
            individuals[individualIndex] = self.MutateIndividual(individuals[individualIndex])

        return individuals 

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
