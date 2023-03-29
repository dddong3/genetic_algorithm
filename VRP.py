# from TSP import *
from matplotlib import pyplot as plt
import numpy as np
from enum import Enum
from ga import *

class vrp:
    class mutation_type(Enum):
        Inversion = 0
        Swap = 1
        Scramble = 2
        Displacement = 3
        Insertion = 4
        Heuristic = 5

    class selection_type(Enum):
        Deterministic = 0
        Roulette = 1
        Tournament = 2
        Rank = 3
        Elitism = 4

    class crossover_type(Enum):
        Order = 0
        PartiallyMapped = 1
        Cycle = 2
        Position = 3
        Heuristic = 4
        PMX = 5

    class crossover:
        def __init__(self) -> None:
            pass
        def order(self):
            pass
    
    class mutation:
        def __init__(self) -> None:
            pass
        def inversion(self):
            pass
    
    class selection:
        def __init__(self) -> None:
            pass
        def roulette(self):
            pass

    def __init__(self, fileName: str):
        f = open(fileName, 'r', newline='')
        n = int(f.readline())
        # cityCoord = [list(map(int, x.split()[1::])) for x in [cc for cc in f.readlines()]]
        self.cityCoord = [list(map(int, x.split()[1::])) for x in [cc for cc in f.readlines()]]
        self.table = np.array(self.cityCoord)
        self.table = np.sqrt(np.sum((self.table[:, None] - self.table[None, :]) ** 2, axis=-1))
        self.len = n
        print(self.table)
        # print((cityCoord))

    def initialize(self, ga: genetic_algorithm):
        ga.chromosome = np.zeros((ga.popSize, ga.geneSize), dtype=int)
        for i in range(ga.popSize):
            ga.chromosome[i] = np.random.permutation(ga.geneSize)
        ga.bestSol = ga.chromosome[0]
        # ga.bestSolTimes = ga.problem.compute_times(ga.bestSol)

    def draw_point(self):
        x, y = zip(*self.cityCoord)
        plt.plot(x, y, 'ro')
        plt.show()