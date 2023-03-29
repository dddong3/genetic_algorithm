from VRP import *
from ga import *
from threading import Thread
from multiprocessing import Process, Manager
import time
import tqdm 
# def run_ga(solution: genetic_algorithm) -> None:
#     for _ in range(gaParameter['liveLoops']):
#         solution.crossover()
#         solution.mutation()
#         solution.selection()
#         solution.update_best()
#         solution.plt.append('best', solution.bestSolTimes)

def run_ga(gaParameter:dict, data:list) -> None:
    solution = genetic_algorithm(gaParameter)
    solution.problem.initialize(solution)
    for _ in range(gaParameter['liveLoops']):
        # print(1)
        solution.run()
        # solution.crossover()
        # solution.mutation()
        # solution.selection()
        # solution.update_best()
    #     solution.plt.append('best', solution.bestSolTimes)
    # print(f'loop: {i} have best sol: {solution.bestSolTimes} with time: {solution.bestSol}')
    # print()
    # print(solution.problem.fitness(solution.chromosome))
    # print(solution.chromosome)
    # print(solution)
    data.append(solution)

if __name__ == '__main__':
    # problem = vrp('test.txt',home=2,vehicleNum=3)###5 car
    problem = vrp('Berlin52.txt',home=1,vehicleNum=5)###5 car
    gaParameter = {
            'liveLoops' : 100,
            'problem' : problem,
            'popSize' : 100,
            'geneSize' : problem.len + problem.vehicleNum - 1,
            'mutationRate' : 0.2,
            'crossoverRate' : 0.8,
            'mutationType' : problem.mutation_type.Swap,
            'selectionType' : problem.selection_type.Deterministic,
            'crossoverType' : problem.crossover_type.PartiallyMapped
    }

    loops = 100

    with Manager() as manager:
        data = manager.list()
        # thread = [Thread(target=run_ga, args=(gaParameter,data)) for _ in range(loops)]
        thread = [Process(target=run_ga, args=(gaParameter,data)) for _ in range(loops)]
        start = time.time()
        [_.start() for _ in thread]

        for _ in tqdm.tqdm(thread):
            _.join()

        # [_.join() for _ in thread]
        end = time.time()
        data = list(data)
    
    # print('f   ', data)
    totalBestIndex, totalBest = 0, data[0].bestSolTimes
    for i in range(loops):
        print(f'loop: {i} have best sol: {data[i].bestSol} with time: {data[i].bestSolTimes}')
        if data[i].bestSolTimes < totalBest:
            totalBest, totalBestIndex = data[i].bestSolTimes, i
        vehiclePath = data[i].problem.vehiclePath(data[i].bestSol)
        for j in range(data[i].problem.vehicleNum):
            print(f'vehicle {j}: {vehiclePath[j]}')
        print('\n\n---------------------\n\n')

    print(f'best everage: {sum([_.bestSolTimes for _ in data]) / loops} best std: {np.std([_.bestSolTimes for _ in data])}')
    print(f'total best: {totalBest} in loop: {totalBestIndex} \n with path: {data[totalBestIndex].bestSol}\nwith vehicle path:')
    bestVehiclePath = data[totalBestIndex].problem.vehiclePath(data[totalBestIndex].bestSol)
    print(*[f'vehicle {i}: {bestVehiclePath[i]}' for i in range(data[totalBestIndex].problem.vehicleNum)], sep='\n')
    print(f'run time: {end - start}')
    problem.draw_point(bestVehiclePath)