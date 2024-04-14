import numpy as np


def GerarProblemaTSP(numeroCidades,raio=1):
    a = np.linspace(0,2*np.pi,numeroCidades+1)
    return np.stack([raio*np.sin(a),raio*np.cos(a)])


