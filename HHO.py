import numpy as np
import math
import random


class HHO:
    def __init__(self, objectiveFunction, lb, ub, dim, N, nIters, random_seed):
        self.fitnessFunction = objectiveFunction
        self.lb = lb
        self.ub = ub
        self.dim = dim
        self.noSearchAgents = N
        self.nIters = nIters
        self.random_seed = random_seed

    def LevyFlight(self, dim):
        beta = 1.5
        sigma = (math.gamma(1+beta)*math.sin(math.pi*beta/2) /
                 (math.gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta)
        u = 0.01*np.random.randn(dim)*sigma
        v = np.random.randn(dim)
        zz = np.power(np.abs(v), (1/beta))
        step = np.divide(u, zz)
        return step

    def optimize(self):
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)

        # Setting the zeros as the starting position of the rabbit
        rabbit_location = np.zeros(self.dim)
        rabbit_energy = np.PINF

        hawks = np.random.uniform(
            0, 1, (self.noSearchAgents, self.dim)) * (self.ub-self.lb) + self.lb

        curve = np.zeros(self.nIters)
        # Number of iterataions for the optimization to take place
        deviation = []
        exploration = []
        exploitation = []
        for iter in range(self.nIters):

            for index in range(self.noSearchAgents):

                hawks[index, :] = np.clip(hawks[index, :], self.lb, self.ub)

                # fitness of the hawk location
                fitness = self.fitnessFunction(hawks[index, :])

                if fitness < rabbit_energy:
                    rabbit_energy = fitness
                    rabbit_location = hawks[index, :].copy()

            E1 = 2 * (1 - (iter/self.nIters))
            # print(rabbit_energy)
            # Updation of the Hawk Location
            for index in range(self.noSearchAgents):
                E0 = 2*np.random.random() - 1  # -1  < E0 <1
                escape_energy = E1 * E0

                if np.abs(escape_energy) >= 1:
                    # Exploration Phase
                    q = np.random.random()
                    random_Hawk_index = np.floor(
                        self.noSearchAgents * np.random.random())
                    random_hawk = hawks[random_Hawk_index.astype('int'), :]

                    if q < 0.5:
                        hawks[index, :] = random_hawk - np.random.random() * np.abs(
                            random_hawk - 2 * np.random.random() * hawks[index, :])

                    elif q >= 0.5:
                        hawks[index, :] = (rabbit_location - hawks.mean(0)) - np.random.random() * (
                            (self.ub-self.lb) * np.random.random() + self.lb)

                elif np.abs(escape_energy) < 1:
                    # Exploitation Phase
                    r = np.random.random()

                    if r >= 0.5 and np.abs(escape_energy) < 0.5:  # Hard Besiege
                        hawks[index, :] = (rabbit_location - hawks[index, :]) - \
                            escape_energy * \
                            np.abs(rabbit_location - hawks[index, :])

                    elif r >= 0.5 and np.abs(escape_energy) >= 0.5:  # Soft Besiege
                        jump_strength = 2 * (1-np.random.random())
                        hawks[index, :] = (rabbit_location - hawks[index, :]) - escape_energy * np.abs(
                            jump_strength * rabbit_location - hawks[index, :])

                    # Soft Besiege Dive
                    elif r < 0.5 and np.abs(escape_energy) >= 0.5:
                        jump_strength = 2 * (1 - np.random.random())
                        new_location = rabbit_location - escape_energy * \
                            np.abs(jump_strength *
                                   rabbit_location - hawks[index, :])

                        if self.fitnessFunction(new_location) < fitness:
                            # Update the new position if it improves its fitness
                            hawks[index, :] = new_location
                        else:
                            levyFlight_position = rabbit_location - escape_energy * \
                                np.abs(jump_strength*rabbit_location - hawks[index, :]) + np.multiply(
                                    np.random.randn(self.dim), self.LevyFlight(self.dim))

                            if self.fitnessFunction(levyFlight_position) < fitness:
                                hawks[index, :] = levyFlight_position

                    elif r < 0.5 and np.abs(escape_energy) < 0.5:  # Hard Besiege Dive

                        jump_strength = 2*(1-np.random.random())
                        new_position = rabbit_location - escape_energy * \
                            np.abs(jump_strength *
                                   rabbit_location - np.mean(hawks, axis=0))

                        if self.fitnessFunction(new_position) < fitness:
                            hawks[index, :] = new_position
                        else:
                            levyFlight_position = rabbit_location - escape_energy * \
                                np.abs(jump_strength * rabbit_location - np.mean(hawks, axis=0)) + np.multiply(
                                    np.random.randn(self.dim), self.LevyFlight(self.dim))

                            if self.fitnessFunction(levyFlight_position) < fitness:
                                hawks[index, :] = levyFlight_position

                hawks[index, :] = np.clip(hawks[index, :], self.lb, self.ub)
            # print("Fitness : ",rabbit_energy)
            median_position = np.median(hawks, axis=0)
            sumation = 0
            for hawk in hawks:
                sumation += np.abs(median_position - hawk)
            divj = sumation/self.noSearchAgents
            deviation.append(sum(divj)/self.dim)
            curve[iter] = rabbit_energy

        for divt in deviation:
            normalized_divt = (divt - np.min(deviation)) / \
                (np.max(deviation) - np.min(deviation))
            # exploration.append(normalized_divt * 100)
            # exploitation.append((1-normalized_divt) * 100)
            exploration.append(divt / np.max(deviation) * 100)
            exploitation.append(
                np.abs(divt - np.max(deviation))/np.max(deviation) * 100)

        return rabbit_location, rabbit_energy, curve, (exploration, exploitation)
