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
        for n in tqdm(range(N)):
            parents = self.individuals[self.ParentSelection(self.individuals)]
            children = self.Recombination(parents)
            children = self.Mutation(children)
            aptitudeList = self.CalculateAptitude(children)
            self.SaveOptimumValue(aptitudeList,children)
            self.individuals = children[self.Replacement(aptitudeList)]
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

