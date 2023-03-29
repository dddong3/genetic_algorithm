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
        # self.problem.initialize(self)
        # self.plt = visual_data()
        # self.chromosome = np.zeros((self.popSize, self.geneSize), dtype=int)
        # for i in range(self.popSize):
        #     self.chromosome[i] = np.random.permutation(self.geneSize)
        # self.bestSol = self.chromosome[0]
        # self.bestSolTimes = self.problem.compute_times(self.bestSol)

        # self.chromosome = np.zeros((self.popSize, self.geneSize), dtype=int)
        # for i in range(self.popSize):
        #     self.chromosome[i] = np.random.permutation(self.geneSize)
        # self.bestSol = self.chromosome[0]
        # self.bestSolTimes = self.problem.fitness(self.bestSol)
    
    def run(self) -> None:
        # pass
        # print(f'city: {self.problem.len}, car num: {self.problem.vehicleNum}')
        self.crossover()
        self.mutation()
        self.selection()
        self.update_best()

        # print("cd:  ",np.shape(self.chromosome))
        # self.plt.append('best', self.bestSolTimes)
        

    def initialize(self):
        # self.plt = visual_data()
        pass
        # print(self.chromosome)

    def update_best(self):
        # print('ff')
        # print(np.shape(self.chromosome))
        mn = self.problem.fitness(self.chromosome[0])
        mx = copy.deepcopy(mn)
        for i in range(self.popSize):
            curTime = self.problem.fitness(self.chromosome[i])
            mn = min(mn, curTime)
            mx = max(mx, curTime)
            if curTime < self.bestSolTimes:
                self.bestSolTimes = curTime
                self.bestSol = self.chromosome[i]
        # self.plt.append('min', mn)
        # self.plt.append('max', mx)
        # self.plt.append('dif', mx - mn)

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

        # print("c  ", self.chromosome)
        # print("ch  ",self.childList)
        if len(self.childList) > 0:
            self.chromosome = np.concatenate((self.chromosome, np.array(self.childList)), axis=0)
        # self.chromosome = np.concatenate((self.chromosome, np.array(self.childList)), axis=0)
        

    def mutation(self):
        for i in range(np.size(self.chromosome, 0)):
            if random.randint(0, 100) <= self.mutationRate * 100:
                self.problem.mutation(i, self)

    def selection(self):
        # self.problem.fitness(self)
        self.problem.selection(self)