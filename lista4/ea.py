import json
import time as tm 
from abc import ABC, abstractmethod 

import numpy as np
from tqdm import tqdm 


class EvolutionaryAlgorithm:
    
    def __init__(self):
        self.parameterList = []
        self.bestFitIndividual = []
        self.bestFitAptitude = -np.inf
        self.saveSize = 10
        self.executionCount = 0
        self.progress = True
        self.creationTime = tm.time()

    def AddParameters(self,parameterList,kwargs):
        """Adds initialization parameter in parameter list"""
        for parameter in parameterList:
            if not parameter in kwargs.keys():
                raise Exception(f"Parameter '{parameter}' not given")
            setattr(self,parameter,kwargs[parameter])
            self.parameterList.append(parameter)
        

    def ExportParameters(self):
        result = {}
        for item in self.parameterList:
            result[item] = getattr(self,item,None)
        return result

    def Execute(self,N):
        
        self.aptitudeEvolution = np.zeros((N,))
        startTime = tm.time()
        stepsIterator = tqdm(range(N)) if self.progress else range(N)
        for n in stepsIterator:
            selectionIndex = self.ParentSelection(self.individuals)
            # Assertions for the selection value
            assert selectionIndex.shape[0] == self.numberOfSelectedParents
            assert len(selectionIndex.shape) == 1
            
            parents = self.individuals[selectionIndex]
            
            children = self.Recombination(parents)
            # Assertions for the children state
            assert self.numberOfRecombinedOffspring == children.shape[0]
            assert self.bitsSize ==children.shape[1]

            children = self.Mutation(children)
            # Assertions for mutation results
            assert self.numberOfRecombinedOffspring == children.shape[0]
            assert self.bitsSize ==children.shape[1]
            
            aptitudeList = self.CalculateAptitude(children)
             
            self.SaveOptimumValue(aptitudeList,children)
            
            survivorsIndex = self.Replacement(aptitudeList)
            assert survivorsIndex.shape[0] == self.populationSize
            assert len(survivorsIndex.shape) == 1
            self.individuals = children[survivorsIndex]

            if self.StopConditionsMet():
                break
            self.aptitudeEvolution[n] = self.bestFitAptitude
        self.executeDuration = tm.time() - startTime
        self.executionCount += 1


    def StopConditionsMet(self):
        """Checks if any stop condition was met during execution and halts if conditions were met."""
        return False

    @abstractmethod 
    def GenerateIndividuals(self,individualsNumber,bitSize):
        """Generate individuals"""
        pass 

    @abstractmethod 
    def CalculateAptitude(self,individuals):
        """Calculate the aptitude for a list of individuals"""
        pass 

    @abstractmethod 
    def Recombination(self):
        """Recombines the genotypes of the parents to create children"""
        pass 

    @abstractmethod 
    def Mutation(self):
        """Alter the genotype of the children"""
        pass 

    @abstractmethod 
    def Replacement(self):
        """Select the children that were best fit"""
        pass 

