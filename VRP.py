# from TSP import *
from matplotlib import pyplot as plt
import numpy as np
from enum import Enum
from ga import *
import copy
import matplotlib.patches as mpatches
import random
import os
import seaborn as sns

class vrp():
    class mutation_type(Enum):
        Inversion = 0 #
        Swap = 1 #
        Scramble = 2
        Displacement = 3
        Insertion = 4
        Heuristic = 5
        Permutation = 6

    class selection_type(Enum):
        Deterministic = 0
        RouletteWheel = 1
        Tournament = 2
        Rank = 3
        Elitism = 4

    class crossover_type(Enum):
        Order = 0  #
        PartiallyMapped = 1 #
        Cycle = 2
        Position = 3
        Heuristic = 4

    class crossover:
        def __init__(self, p1:int, p2:int, ga):
            if ga.crossoverType == ga.problem.crossover_type.PartiallyMapped:
                self.PartiallyMapped(p1, p2, ga)
            if ga.crossoverType == ga.problem.crossover_type.Order:
                self.OrderCrossover(p1, p2, ga)

        def PartiallyMapped(self, p1, p2, ga):
            childList = ga.childList
            leftBound, rightBound = ga.get_rand_range()

            # c1 = copy.deepcopy(ga.chromosome[p1])
            # c2 = copy.deepcopy(ga.chromosome[p2])
            # for i in range(leftBound, rightBound + 1):
            #     c1[i], c2[i] = c2[i], c1[i]

            c1, c2 = ga.chromosome[p1][:], ga.chromosome[p2][:]
            # c1[leftBound:rightBound], c2[leftBound:rightBound] = c2[leftBound:rightBound], c1[leftBound:rightBound]

            # print(c1)
            # print(c2)

            for i in range(leftBound, rightBound + 1):
                c1[i], c2[i] = c2[i], c1[i]

            #change c1[leftBound:rightBound+1] to c2[leftBound:rightBound+1] and c2[leftBound:rightBound+1] to c1[leftBound:rightBound+1] without for loop
            # c1[leftBound:rightBound+1], c2[leftBound:rightBound+1] = c2[leftBound:rightBound+1], c1[leftBound:rightBound+1]

            

            cxNumToIdx = [{}, {}]
            cxRepeat = [[], []]

            for i in range(ga.geneSize):
                if c1[i] not in cxNumToIdx[0]:
                    cxNumToIdx[0][c1[i]] = i
                elif leftBound <= i <= rightBound:
                    cxRepeat[0].append(cxNumToIdx[0][c1[i]])
                else:
                    cxRepeat[0].append(i)

                if c2[i] not in cxNumToIdx[1]:
                    cxNumToIdx[1][c2[i]] = i
                elif leftBound <= i <= rightBound:
                    cxRepeat[1].append(cxNumToIdx[1][c2[i]])
                else:
                    cxRepeat[1].append(i)
            offset = random.randint(0, len(cxRepeat[1]))
            # if offset == -1 and len(cxRepeat[1]) > 0:
            #     offset = random.randint(0, len(cxRepeat[1]) - 1)
            for i in range(len(cxRepeat[0])):
                if len(cxRepeat[1]) == 0:
                    break
                a = cxRepeat[0][i]
                b = cxRepeat[1][(i + offset) % len(cxRepeat[1])]
                c1[a], c2[b] = c2[b], c1[a]

            childList.append(c1)
            childList.append(c2)
        
        def OrderCrossover(self, p1:int, p2:int, ga):
            if np.size(p1, 0) == 0 or np.size(p2, 0) == 0:
                return
            childList = ga.childList
            leftBound, rightBound = ga.getRndRange()

            c1 = copy.deepcopy(ga.chromosome[p1])
            c2 = copy.deepcopy(ga.chromosome[p2])

            for i in range(leftBound, rightBound + 1):
                c1[i], c2[i] = c2[i], c1[i]

            cxNumToIdx = [dict() for _ in range(2)]
            cxRepeat = [list() for _ in range(2)]

            for i in range(ga.geneSize):
                if c1[i] not in cxNumToIdx[0]:
                    cxNumToIdx[0][c1[i]] = i
                elif leftBound <= i <= rightBound:
                    cxRepeat[0].append(cxNumToIdx[0][c1[i]])
                else:
                    cxRepeat[0].append(i)

                if c2[i] not in cxNumToIdx[1]:
                    cxNumToIdx[1][c2[i]] = i
                elif leftBound <= i <= rightBound:
                    cxRepeat[1].append(cxNumToIdx[1][c2[i]])
                else:
                    cxRepeat[1].append(i)

            for i in range(len(cxRepeat[0])):
                a = cxRepeat[0][i]
                b = cxRepeat[1][i]
                c1[a], c2[b] = c2[b], c1[a]

            childList.append(c1)
            childList.append(c2)
    
    class mutation:
        def __init__(self, idx:int, ga) -> None:
            if ga.mutationType == ga.problem.mutation_type.Inversion:
                self.Inversion(idx, ga)
            if ga.mutationType == ga.problem.mutation_type.Swap:
                self.Swap(idx, ga)
            if ga.mutationType == ga.problem.mutation_type.Permutation:
                self.Permutation(idx, ga)
        def Inversion(self, idx:int, ga):
            leftBound, rightBound = ga.get_rand_range()
            ga.chromosome[idx][leftBound:rightBound + 1] = ga.chromosome[idx][leftBound:rightBound + 1][::-1]
    
        def Swap(self, idx:int, ga):
            sa = random.randint(0, ga.geneSize - 2)
            sb = random.randint(sa + 1, ga.geneSize - 1)
            gene = ga.chromosome[idx]
            gene[sa], gene[sb] = gene[sb], gene[sa]
        def Permutation(self, idx:int, ga):
            leftBound, rightBound = ga.getRndRange()
            gene = ga.chromosome[idx]
            gene[leftBound:rightBound + 1] = np.random.permutation(gene[leftBound:rightBound + 1])


    class selection:
        def __init__(self, ga) -> None:
            if ga.selectionType == ga.problem.selection_type.Deterministic:
                self.Deterministic(ga)
            if ga.selectionType == ga.problem.selection_type.RouletteWheel:
                self.RouletteWheel(ga)
        def RouletteWheel(self, ga):
            ga.fitness = ga.fitness / np.sum(ga.fitness)
            ga.chromosome = ga.chromosome[np.random.choice(np.size(ga.chromosome, 0), ga.popSize, replace=False, p=ga.fitness)]
        def Deterministic(self, ga):
            # lst = np.argsort([ga.problem.fitness(i) for i in ga.chromosome])
            # print(f"lst {lst}")
            # lst = np.argsort(lst)
            # print(f"lst {lst}")
            ga.chromosome = ga.chromosome[np.argsort([ga.problem.fitness(i) for i in ga.chromosome])]
            # lst = [ga.problem.fitness(i) for i in ga.chromosome]
            # lst = np.argsort(lst)
            # lst = None
            # print(f"lst {lst}")
            # a = [ga.chromosome[i] for i in lst[::-1]]
            # ga.chromosome = np.array(a[:ga.popSize])

    def vehiclePath(self, chromosome: list[int]) -> list[list[int]]:
        #[7, 3, 2, 5, 1, 4, 6, 0] car = [5, 6, 7] -> [[7, 3, 2], [5, 1, 4], [6, 0]]
        vehiclePath, curVehiclePath = [], [self.home]
        for i in range(len(chromosome)):
            if chromosome[i] >= self.carPoint:
                vehiclePath.append(curVehiclePath + [self.home])
                curVehiclePath = [self.home]
            else:
                curVehiclePath.append(chromosome[i])
        vehiclePath.append(curVehiclePath + [self.home])
        return vehiclePath


    def fitness(self, chromosome: list[int]) -> float:
        vehiclePath = self.vehiclePath(chromosome)

        #compute the path without for loop
        # print(vehiclePath)
        # vehicleLen = [0.0 for _ in range(self.vehicleNum)]
        # for i in range(self.vehicleNum):
        #     # print(vehiclePath[i])
        #     for j in range(1, len(vehiclePath[i])):
        #         # print(f'{i}  {j}')
        #         vehicleLen[i] += self.table[vehiclePath[i][j-1]][vehiclePath[i][j]]

        # vehiclePath = vehiclePath[1:]
        # print(chromosome)
        # print(vehiclePath)

        #compute the path with for loop
        vehicleLen = [0.0 for _ in range(self.vehicleNum)]
        for i in range(self.vehicleNum):
            for j in range(1, len(vehiclePath[i])):
                vehicleLen[i] += self.table[vehiclePath[i][j-1] - 1][vehiclePath[i][j] - 1]


        # vehicleLen = [sum(self.table[a][b] for a, b in zip(path[:-1], path[1:])) for path in vehiclePath]
        return sum(vehicleLen)

        # @staticmethod
        # def countVehiclePass(chromosome: int) -> list[int]:
        #     # chromosome = 10261
        #     vehiclePassCnt = [0 for _ in range(self.vehicleNum + 1)]
        #     chro = ''.join([str(i) for i in chromosome])
        #     chro = int(chro, 2)
        #     mask = 2**self.len - 1
        #     for i in range(self.vehicleNum):
        #         vehiclePassCnt[i] = bin(mask & chro).count('1')
        #         mask <<= self.len
        #     return vehiclePassCnt[::-1]
        # def count_path_length(path: list[int]) -> float:
        #     length =  0.0
        #     for i in range(1, len(path)):
        #         length += self.table[path[i-1]][path[i]]
        #     return length
        # # len = [sum([self.table[chromosome[0][i]][chromosome[0][i + 1]] for i in range(chromosome[1][j] - 1)]) for j in range(self.vehicleNum)]
        # vehicleLen = [0.0 for _ in range(self.vehicleNum)]
        # print(self.len)
        # # for i in range(self.len):
        # #     len[i // self.vehicleNum] +=
        
        # cnt = 0
        # curVehicle = 0
        # home = chromosome[2]
        # vehiclePassCnt = countVehiclePass(chromosome[1])
        # vehiclePassCnt = [sum(vehiclePassCnt[:i + 1]) for i in range(self.vehicleNum + 1)]
        # print(vehiclePassCnt)
        # for i in range(1, len(vehiclePassCnt)):
            # let list = home + chromosome[0][vehiclePassCnt[i-1]:vehiclePassCnt[i]] + home
            # then calculate the length of list
            # tmp = deepc
            # tmp = [home] + chromosome[0][vehiclePassCnt[i-1]:vehiclePassCnt[i]].tolist() + [home]
            # pass
            # print('f    ', [home] + chromosome[0][vehiclePassCnt[i-1]:vehiclePassCnt[i]] + [home])
            # vehicleLen[i-1] = count_path_length([home] + chromosome[0][vehiclePassCnt[i-1]:vehiclePassCnt[i]].tolist() + [home])
        # c = [0,1,2,3,4,5,6,7,8]
        # print(chromosome[0][2:5])
        # for i in range(self.len):
        #     if i == cnt: # first city
        #         len[curVehicle] += self.table[home][chromosome[0][i]]
        #         print(f'{home} -> {chromosome[0][i]}, add v{ curVehicle}')
        #     elif i == cnt + vehiclePassCnt[curVehicle] - 1: # last city
        #         len[curVehicle] += self.table[chromosome[0][i]][home]
        #         curVehicle += 1
        #         print(f'{chromosome[0][i]} -> {home}, add v{ curVehicle}')
        #         cnt += vehiclePassCnt[curVehicle]
        #     else: # middle city
        #         len[curVehicle] += self.table[chromosome[0][i]][chromosome[0][i - 1]]
        #         print(f'{chromosome[0][i - 1]} -> {chromosome[0][i]}, add v{ curVehicle}')

        # for i in range(self.vehicleNum):
        #     len[i] = count_path_length(chromosome[0][cnt:cnt + vehiclePassCnt[i]], home)

        # print(len)

        # print(self.vehicleNum)
        # for i in range(self.len):
        #     if i == cnt: # first city
        #         len[vehicle] += self.table[startPoint][chromosome[0][i]]
        #         print(f'{startPoint} -> {chromosome[0][i]}, add v{ vehicle}')
        #     elif i == cnt + chromosome[1][i // self.vehicleNum] - 1: # last city
        #         len[vehicle] += self.table[chromosome[0][i]][startPoint]
        #         vehicle += 1
        #         print(f'{chromosome[0][i]} -> {startPoint}, add v{ vehicle}')
        #         cnt += chromosome[1][i // self.vehicleNum]
        #     else: # middle city
        #         len[vehicle] += self.table[chromosome[0][i]][chromosome[0][i - 1]]
        #         print(f'{chromosome[0][i - 1]} -> {chromosome[0][i]}, add v{ vehicle}')


        # len = [len[i] + len[i] for i in range(self.vehicleNum)]
        # print(vehicleLen)

        # # print(self.table)
        
        # # print(chromosome)
        # print(vehiclePassCnt)
        # # pass
        # return sum(vehicleLen)

    def __init__(self, fileName: str, **args) -> None:
        [setattr(self, key, value) for key, value in args.items()]
        if 'vehicleNum' not in args:
            self.vehicleNum = 1
        
        if 'home' not in args:
            self.home = 0

        f = open(fileName, 'r', newline='')
        n = int(f.readline())
        # cityCoord = [list(map(int, x.split()[1::])) for x in [cc for cc in f.readlines()]]
        coord = [x.split() for x in [cc for cc in f.readlines()]]
        self.cityCoord = {int(x[0]): list(map(float, x[1::])) for x in coord}
        minimumIdx = min(self.cityCoord.keys())
        self.table = np.array([self.cityCoord[i] for i in range(minimumIdx, minimumIdx + n)])
        self.table = np.sqrt(np.sum((self.table[:, None] - self.table[None, :]) ** 2, axis=-1))
        # self.table = np.sqrt(np.sum((self.table[:, None] - self.table[None, :]) ** 2, axis=-1))
        self.len = n
        print(self.table)
        # self.table = np.array(self.cityCoord)
        # self.table = np.sqrt(np.sum((self.table[:, None] - self.table[None, :]) ** 2, axis=-1))
        # self.len = n
        # print(self.table)
        # print(self.cityCoord.get(52, None))

    def initialize(self, ga: genetic_algorithm):
        # s = (2**(self.len*self.vehicleNum) - 1) ^ (2**(self.len*(self.vehicleNum-1)) - 1)
        # s = list(str(bin(s))[2:])
        # random.shuffle(s)
        # print(np.random.permutation(s))
        # ga.chromosome = [[np.random.permutation(ga.geneSize), np.random.permutation(s), self.home]for _ in range(ga.popSize)]
        # ga.chromosome = np.random.permutation(ga.geneSize)
        # ga.chromosome += 1
        # print('key    ', self.cityCoord.keys())
        cityList = list(self.cityCoord.keys())
        cityList.remove(self.home)
        cityList += [ _ + self.len + 1 for _ in range(self.vehicleNum - 1)]
        ga.geneSize = len(cityList)
        self.carPoint = self.len + 1
        self.len = len(cityList)
        ga.chromosome = [np.random.permutation(cityList) for _ in range(ga.popSize)]
        ga.bestSol = ga.chromosome[0]
        ga.bestSolTimes = ga.problem.fitness(ga.bestSol)
        # print(ga.chromosome)
        # ga.chromosome = [[np.random.permutation(ga.geneSize), np.random.permutation(2**(self.len*self.vehicleNum) - 1), self.home]for _ in range(ga.popSize)]

        #     a = (2**(self.len*self.vehicleNum) - 1) ^ (2**(self.len*(self.vehicleNum-1)) - 1)
        # a = list(str(bin(a))[2:])
        # random.shuffle(a)
        # a = int(''.join(a),2)
        # print(a)
        # print(ga.chromosome, len(ga.chromosome))
        # print(self.len, self.vehicleNum)
        # print(ga.geneSize)
        # tt = ga.chromosome[0][1]
        # print(2**(self.len*self.vehicleNum) - 1)
        # print(tt)
        '''
        [
            #0 -> [0, 1, 2, 3, 4, ..., n], city permutation
            #1 -> 2^city * vehicleNum - 1
            #2 -> [x]                   vehicle start point
        ]
            #####1 -> [v1, v2, v3, ..., vn]  vehicle capacity v1 + v2 + v3 + ... + vn = n
        '''

        ga.bestSol = ga.chromosome
        # ga.bestSolTimes = ga.problem.compute_times(ga.bestSol)

    def draw_vehicle_paths(self, cityCoords, vehiclePath, vehicleIdx, vehicleColors) -> None:
        for i in range(len(vehiclePath) - 1):
            start = cityCoords[vehiclePath[i]]
            end = cityCoords[vehiclePath[i+1]]
            plt.arrow(start[0], start[1], end[0]-start[0], end[1]-start[1], length_includes_head=True, head_width=0.5, head_length=0.5, fc=f'C{vehicleIdx+1}', ec=f'C{vehicleIdx+1}')
        vehicleColors.append(mpatches.Patch(color=f'C{vehicleIdx+1}', label=f'Vehicle {vehicleIdx}'))

    def draw_point(self, vehiclePaths, save_path='vehicle_paths') -> None:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # x, y = zip(*self.cityCoord)[1::]

        x, y = zip(*self.cityCoord.values())

        plt.figure(figsize=(10, 6))
        plt.plot(x, y, 'ro')
        plt.plot(self.cityCoord[self.home][0], self.cityCoord[self.home][1], 'bo')

        vehicleColors = sns.color_palette("hls", self.vehicleNum)
        legend = []

        for idx, coord in self.cityCoord.items():
            plt.text(coord[0] + 0.5, coord[1] + 0.5, str(idx))
        for idx, vehicle_path in enumerate(vehiclePaths):
            # self.draw_vehicle_paths(self.cityCoord, vehicle_path, idx, vehicleColors)
            for i in range(len(vehicle_path) - 1):
                # print(vehicle_path)
                start = self.cityCoord[vehicle_path[i]]
                end = self.cityCoord[vehicle_path[i+1]]
                plt.arrow(start[0], start[1], end[0]-start[0], end[1]-start[1], \
                          length_includes_head=True, head_width=0.5, head_length=0.5, fc=vehicleColors[idx] , ec=vehicleColors[idx])
        
            legend.append(mpatches.Patch(color=vehicleColors[idx], label=f'Vehicle {idx}'))

        plt.legend(handles=legend)
        # plt.show()
        plt.savefig(os.path.join(save_path, f'all_vehicle_path.png'))
        plt.close()

        for vehicle_idx, vehicle_path in enumerate(vehiclePaths):
            plt.figure(figsize=(10, 6))
            plt.plot(x, y, 'ro')

            for i, coord in self.cityCoord.items():
                plt.text(coord[0]+0.5, coord[1]+0.5, f'{i}', fontsize=12)

            for i in range(len(vehicle_path) - 1):
                start = self.cityCoord[vehicle_path[i]]
                end = self.cityCoord[vehicle_path[i+1]]
                plt.arrow(start[0], start[1], end[0]-start[0], end[1]-start[1],
                        length_includes_head=True, head_width=0.5, head_length=0.5,
                        fc=vehicleColors[vehicle_idx], ec=vehicleColors[vehicle_idx])
            
            plt.savefig(os.path.join(save_path, f'vehicle_{vehicle_idx}_path.png'))
            plt.close()