import json
import numpy as np
import matplotlib.pyplot as plt 

from sga import GeneticAlgorithm

def ConvertMatrixArray(array):
    Y,_ = np.meshgrid(np.arange(array.shape[1]),np.arange(array.shape[0]))
    Y = 2**Y
    result = np.where(array,Y,0)
    return np.sum(result,axis=1)


class MinimizeFunction(GeneticAlgorithm):
    
    def CalculateFenotypes(self,individuals):
        values = ConvertMatrixArray(individuals)
        values = values-2**(self.bitsSize - 1)
        values = values/2**(self.bitsSize - 2)
        return values

    def CalculateAptitude(self,individuals):
        values = self.CalculateFenotypes(individuals)
        return - (np.power(values,2)-0.3*np.cos(10*np.pi*values))
     
    def Report(self):
        report = {}
        report['Creation time'] = float(self.creationTime)
        report['Current execution'] = self.executeDuration
        report['SGA Params'] = {}
        for param in self.sgaParams:
            report['SGA Params'][param] = float(getattr(self,param))
        report['Results'] = {}
        report["Results"]['Best Individual'] = [int(i) for i in self.bestFitIndividual]
        report['Results']['Best Individual value'] = float(self.CalculateFenotypes(self.bestFitIndividual[np.newaxis,:])[0])
        report['Results']["Best aptitude"] = float(self.bestFitAptitude)
        return report 

if __name__ == "__main__":

    test = np.array([[1,1,0],[1,0,0],[1,0,1]])
    a = MinimizeFunction({
        "populationSize": 30,
        "bitsSize": 18,
        "numBits":5,
        "mutationProbability": 0.7,
        "crossoverProbability": 0.3,
        "numberOfSelectedParents":15,
        "numberOfRecombinedOffspring":60
    })
    a.Execute(1000)
    report = a.Report()
    print(json.dumps(report, indent=2))
    plt.show()
