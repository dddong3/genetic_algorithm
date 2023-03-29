from VRP import *
from ga import *
from threading import Thread

# def run_ga(solution: genetic_algorithm) -> None:
#     for _ in range(gaParameter['liveLoops']):
#         solution.crossover()
#         solution.mutation()
#         solution.selection()
#         solution.update_best()
#         solution.plt.append('best', solution.bestSolTimes)

def run_ga(i:int, gaParameter:dict, data:list) -> None:
    solution = genetic_algorithm(gaParameter)
    # for _ in range(gaParameter['liveLoops']):
        # solution.crossover()
        # solution.mutation()
        # solution.selection()
        # solution.update_best()
    #     solution.plt.append('best', solution.bestSolTimes)
    data[i] = solution

if __name__ == '__main__':
    problem = vrp('Oliver30.txt')
    gaParameter = {
            'liveLoops' : 100,
            'problem' : problem,
            'popSize' : 500,
            'geneSize' : problem.len,
            'mutationRate' : 0.2,
            'crossoverRate' : 0.8,
            'mutationType' : problem.mutation_type.Inversion,
            'selectionType' : problem.selection_type.Deterministic,
            'crossoverType' : problem.crossover_type.PMX
    }

    loops = 10
    data = [ None for _ in range(loops)]
    # thread = [thread.append(Thread(target= thread_func, args=(i,gaParameter, data))) for i in range(loops)]
    # for i in range(loops):
    thread = [Thread(target=run_ga, args=(i,gaParameter,data)) for i in range(loops)]
    # run thread without using for loop
    [_.start() for _ in thread]
    [_.join() for _ in thread]