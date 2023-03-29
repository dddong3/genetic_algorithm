import csv
import copy
import random
import numpy as np
# from JAP import *
# from TSP import *
from VRP import *
from enum import Enum
# from visual_data import *
# from scipy.optimize import linear_sum_assignment

class genetic_algorithm():
    def __init__(self, parameter):
        parameterlist = ['problem','popSize','geneSize','crossoverRate','mutationRate','mutationType','selectionType','crossoverType']
        [setattr(self, s, parameter[s]) for s in parameterlist]
        self.problem.initialize(self)
        # self.plt = visual_data()
        # self.chromosome = np.zeros((self.popSize, self.geneSize), dtype=int)
        # for i in range(self.popSize):
        #     self.chromosome[i] = np.random.permutation(self.geneSize)
        # self.bestSol = self.chromosome[0]
        # self.bestSolTimes = self.problem.compute_times(self.bestSol)
    
    def run(self) -> None:
        self.crossover()
        self.mutation()
        self.selection()
        self.update_best()
        # self.plt.append('best', self.bestSolTimes)
        

    def initialize(self):
        # self.plt = visual_data()
        self.chromosome = np.zeros((self.popSize, self.geneSize), dtype=int)
        for i in range(self.popSize):
            self.chromosome[i] = np.random.permutation(self.geneSize)
        self.bestSol = self.chromosome[0]
        self.bestSolTimes = self.problem.compute_times(self.bestSol)

    def update_best(self):
        mn = 0xffffffff
        mx = -1
        for i in range(self.popSize):
            curTime = self.problem.compute_times(self.chromosome[i])
            mn = min(mn, curTime)
            mx = max(mx, curTime)
            if curTime < self.bestSolTimes:
                self.bestSolTimes = curTime
                self.bestSol = self.chromosome[i]
        self.plt.append('min', mn)
        self.plt.append('max', mx)
        self.plt.append('dif', mx - mn)

    def get_rand_range(self):
        if self.geneSize <= 1:
            return 0, self.geneSize
        leftBound = random.randint(0, self.geneSize - 1)
        rightBound = random.randint(leftBound, self.geneSize - 1)
        return leftBound, rightBound

    def crossover(self):
        self.childList = []
        rng = self.popSize - self.popSize % 2
        self.chromosome = np.random.permutation(self.chromosome)
        for i in range(0, rng, 2):
            if random.randint(0, 100) <= self.crossoverRate * 100:
                # print(self.chromosome[i], self.chromosome[i + 1])
                self.problem.crossover(i, i + 1, self)
        self.chromosome = np.concatenate((self.chromosome, np.array(self.childList)))

    def mutation(self):
        for i in range(np.size(self.chromosome, 0)):
            if random.randint(0, 100) <= self.mutationRate * 100:
                self.problem.mutation(i, self)

    def selection(self):
        self.problem.compute_fitness(self)
        self.problem.selection(self)