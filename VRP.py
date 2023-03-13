# from TSP import *
from matplotlib import pyplot as plt

class vrp():
    def __init__(self, fileName: str):
        f = open(fileName, 'r', newline='')
        n = int(f.readline())
        cityCoord = [list(map(int, x.split()[1::])) for x in [cc for cc in f.readlines()]]
        self.cityCoord = cityCoord
        # self.table = [ sublist for sublist in cityCoord]
        # print(self.table)
        self.cityCount = n
        # print(len(cityCoord))

    def draw_point(self):
        x, y = zip(*self.cityCoord)
        plt.plot(x, y, 'ro')
        plt.show()