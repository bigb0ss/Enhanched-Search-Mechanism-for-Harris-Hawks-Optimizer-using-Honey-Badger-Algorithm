import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
import ObjectiveFunctions as obj
from cec2017.functions import all_functions
from cec2017.functions import *
from HHO import HHO
from HBA import HBA
from HBHHO import HBHHO


import warnings
warnings.filterwarnings('ignore')


def runExperiment(experimentResults):
    hho_results = []
    hba_results = []
    hbhho_results = []

    hho_explore = []
    hba_explore = []
    hbhho_explore = []

    hho_exploit = []
    hba_exploit = []
    hbhho_exploit = []

    hho_fitness_list = []
    hba_fitness_list = []
    hbhho_fitness_list = []

    for run in tqdm(range(RUNS), ascii=True):
        hho = HHO(objectiveFunction=OBJECTIVE, lb=LB,
                  ub=UB, dim=DIM, N=N, nIters=ITERS, random_seed=SEED+run)
        hba = HBA(objectiveFunction=OBJECTIVE, lb=LB,
                  ub=UB, dim=DIM, N=N, nIters=ITERS, random_seed=SEED+run)
        hbhho = HBHHO(objectiveFunction=OBJECTIVE, lb=LB,
                      ub=UB, dim=DIM, N=N, nIters=ITERS, random_seed=SEED+run)

        hho_position, hho_fitness, hho_convergence, (
            hho_exploration, hho_exploitation) = hho.optimize()
        hba_position, hba_fitness, hba_convergence, (
            hba_exploration, hba_exploitation) = hba.optimize()
        hbhho_position, hbhho_fitness, hbhho_convergence, (
            hbhho_exploration, hbhho_exploitation) = hbhho.optimize()

        hho_fitness_list.append(hho_fitness)
        hba_fitness_list.append(hba_fitness)
        hbhho_fitness_list.append(hbhho_fitness)

        hho_results.append(hho_convergence)
        hba_results.append(hba_convergence)
        hbhho_results.append(hbhho_convergence)

        hho_explore.append(hho_exploration)
        hho_exploit.append(hho_exploitation)

        hba_explore.append(hba_exploration)
        hba_exploit.append(hba_exploitation)

        hbhho_explore.append(hbhho_exploration)
        hbhho_exploit.append(hbhho_exploitation)

    output_directory = OBJECTIVE.__name__+'/'
    if OBJECTIVE.__name__ not in os.listdir():
        os.system('mkdir '+OBJECTIVE.__name__)

    fig, ax = plt.subplots(1,figsize=(10,6))

    ax.plot(np.mean(hho_results, axis=0))
    ax.plot(np.mean(hba_results, axis=0))
    ax.plot(np.mean(hbhho_results, axis=0))

    plt.title("HHO vs HBA vs HBHHO - Convergence Graph : "+OBJECTIVE.__name__)
    ax.set_xlabel("Iterations")
    ax.set_yscale('log')
    ax.set_ylabel("Fitness value")
    ax.legend(['HHO', 'HBA', 'HBHHO'])
    # plt.show()
    plt.savefig(output_directory + "Convergence graph.svg")
    plt.savefig(OBJECTIVE.__name__+"1.jpg")

    # Convergence Graph
    fig, ax = plt.subplots(1, 3, figsize=(30, 6))
    fig.suptitle("Exploration  - Exloitation ")

    ax[0].plot(np.median(hho_explore, axis=0))
    ax[0].plot(np.median(hho_exploit, axis=0))
    ax[0].set_ylim(-10, 110)
    ax[0].set_title("HHO")
    ax[0].set_xlabel("Iterations")
    ax[0].set_ylabel("Percentage")

    ax[1].plot(np.median(hba_explore, axis=0))
    ax[1].plot(np.median(hba_exploit, axis=0))
    ax[1].set_ylim(-10, 110)
    ax[1].set_title("HBA")
    ax[1].set_xlabel("Iterations")
    ax[1].set_ylabel("Percentage")

    ax[2].plot(np.median(hbhho_explore, axis=0))
    ax[2].plot(np.median(hbhho_exploit, axis=0))
    ax[2].set_ylim(-10, 110)
    ax[2].set_title("HBHHO")
    ax[2].set_xlabel("Iterations")
    ax[2].set_ylabel("Percentage")

    ax[0].legend(["Exploration", "Exploitation"])
    ax[1].legend(["Exploration", "Exploitation"])
    ax[2].legend(["Exploration", "Exploitation"])
    plt.savefig(output_directory + "exploration-exploitation.svg")

    results = {}
    results['function'] = OBJECTIVE.__name__
    results['hho_mean'] = np.mean(hho_fitness_list)
    results['hho_std'] = np.std(hho_fitness_list)

    results['hba_mean'] = np.mean(hba_fitness_list)
    results['hba_std'] = np.std(hba_fitness_list)

    results['hbhho_mean'] = np.mean(hbhho_fitness_list)
    results['hbhho_std'] = np.std(hbhho_fitness_list)

    fig, ax = plt.subplots(1)
    x = ['HHO', 'HBA', 'HBHHO']

    # print(np.mean(hho_explore))
    ax.bar(x, [np.mean(hho_explore), np.mean(
        hba_explore), np.mean(hbhho_explore)], color='b')
    ax.bar(x, [100-np.mean(hho_explore), 100 -
           np.mean(hba_explore), 100-np.mean(hbhho_explore)], bottom=[np.mean(hho_explore), np.mean(
               hba_explore), np.mean(hbhho_explore)], color='tab:orange')

    y_offset = -5
    for bar in ax.patches:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + bar.get_y() + y_offset,
            round(bar.get_height()),
            ha='center',
            color='w',
            weight='bold',
            size=15
        )

    ax.set_ylabel('Percentage')
    ax.set_xlabel('Algorithms')
    ax.legend(['Exploration', 'Exploitation'], bbox_to_anchor=(1, 0.5))
    # plt.show()
    # plt.savefig(output_directory + "Barchart.svg", bbox_inches="tight")
    # plt.show()
    experimentResults = experimentResults.append(results, ignore_index=True)
    return experimentResults

#########################################


# Testing with specific function and parameters
OBJECTIVE = f19
LB = -100
UB = 100
DIM = 50
N = 5
ITERS = 1000
RUNS = 30

# Setting the random seed for all the algorithms
SEED = 5


# pos, fit, c, (ex, exp) = HHO(objectiveFunction=OBJECTIVE, lb=LB,
#                              ub=UB, dim=DIM, N=N, nIters=ITERS).optimize()
# print(fit)

experimentResults = pd.DataFrame()
# experimentResults = runExperiment(experimentResults)

##########################################


# # Testing with all functions

# [obj.f1, obj.f4, obj.f7, obj.f10, obj.f13, obj.f16]
# [f1, f6, f13, f21, f26]
for function in tqdm([f3,f10,f21,f27]):
    if function.__name__ not in ['f17', 'f20', 'f29']:
        OBJECTIVE = function
        experimentResults = runExperiment(experimentResults)


# experimentResults.to_csv('output.csv')
