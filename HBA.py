import numpy as np

from numpy import linalg as LA
import math
import random



class HBA:
    # This function is to initialize the Honey Badger population.
    def __init__(self, objectiveFunction, dim, lb, ub, nIters, N, random_seed,beta=6, C=2):
        self.fitnessFunction = objectiveFunction
        self.dim = dim
        self.lb = lb * np.ones([dim, 1])
        self.ub = ub * np.ones([dim, 1])
        self.nIters = nIters
        self.N = N
        self.random_seed = random_seed

        self.beta = beta
        self.C = C

    def initial(self, pop, dim, ub, lb):
        X = np.zeros([pop, dim])
        for i in range(pop):
            for j in range(dim):
                X[i, j] = np.random.random()*(ub[j] - lb[j]) + lb[j]
        return X

    # Calculate fitness values for each Honey Badger.
    def CaculateFitness1(self, X, fun):
        fitness = fun(X)
        return fitness

    # Sort fitness.
    def SortFitness(self, Fit):
        fitness = np.sort(Fit, axis=0)
        index = np.argsort(Fit, axis=0)
        return fitness, index

    # Sort the position of the Honey Badger according to fitness.
    def SortPosition(self, X, index):
        Xnew = np.zeros(X.shape)
        for i in range(X.shape[0]):
            Xnew[i, :] = X[index[i], :]
        return Xnew

    # Boundary detection function.
    def BorderCheck1(self, X, lb, ub, dim):
        for j in range(dim):
            if X[j] < lb[j]:
                X[j] = ub[j]
            elif X[j] > ub[j]:
                X[j] = lb[j]
        return X

    def Intensity(self, pop, GbestPositon, X):
        epsilon = 0.00000000000000022204
        di = np.zeros(pop)
        S = np.zeros(pop)
        I = np.zeros(pop)
        for j in range(pop):
            if (j <= pop):
                di[j] = LA.norm([[X[j, :]-GbestPositon+epsilon]])
                S[j] = LA.norm([X[j, :]-X[j+1, :]+epsilon])
                di[j] = np.power(di[j], 2)
                S[j] = np.power(S[j], 2)
            else:
                print("check")
                di[j] = [LA.norm[[X[pop, :]-GbestPositon+epsilon]]]
                S[j] = [LA.norm[[X[pop, :]-X[1, :]+epsilon]]]
                di[j] = np.power(di[j], 2)
                S[j] = np.power(S[j], 2)

            for i in range(pop):
                n = np.random.random()
                # print(4*np.pi*di[i])
                I[i] = n*S[i]/[4*np.pi*di[i]]
            return I

    def optimize(self):

        np.random.seed(self.random_seed)
        random.seed(self.random_seed)

        # Initialize the number of honey badgers
        X = self.initial(self.N, self.dim, self.lb, self.ub)
        fitness = np.zeros([self.N, 1])
        for i in range(self.N):
            fitness[i] = self.CaculateFitness1(X[i, :], self.fitnessFunction)
        # Sort the fitness values of honey badger.
        fitness, sortIndex = self.SortFitness(fitness)
        # Sort the honey badger.
        X = self.SortPosition(X, sortIndex)
        # The optimal value for the current iteration.
        GbestScore = fitness[0]
        GbestPositon = np.zeros([1, self.dim])
        GbestPositon[0, :] = X[0, :]
        Curve = np.zeros([self.nIters, 1])
        C = 2                                          # constant in Eq. (3)
        # the ability of HB to get the food  Eq.(4)
        beta = 6
        vec_flag = [1, -1]
        vec_flag = np.array(vec_flag)
        Xnew = np.zeros([self.N, self.dim])
        deviation = []
        exploration = []
        exploitation = []
        for t in range(self.nIters):
            #print("iteration: ",t)
            # density factor in Eq. (3)
            alpha = C*np.exp(-t/self.nIters)
            # intensity in Eq. (2)
            I = self.Intensity(self.N, GbestPositon, X)
            Vs = np.random.random()
            for i in range(self.N):
                Vs = np.random.random()
                F = vec_flag[math.floor((2*np.random.random()))]
                for j in range(self.dim):
                    di = GbestPositon[0, j]-X[i, j]
                    if (Vs < 0.5):                           # Digging phase Eq. (4)
                        r3 = np.random.random()
                        r4 = np.random.randn()
                        r5 = np.random.randn()
                        Xnew[i, j] = GbestPositon[0, j] + F*beta*I[i] * GbestPositon[0, j] + \
                            F*r3*alpha*(di)*np.abs(np.cos(2*np.pi*r4)
                                                   * (1-np.cos(2*np.pi*r5)))
                    else:
                        r7 = np.random.random()
                        Xnew[i, j] = GbestPositon[0, j]+F*r7 * \
                            alpha*di    # Honey phase Eq. (6)
                # print(di)
                Xnew[i, :] = self.BorderCheck1(
                    Xnew[i, :], self.lb, self.ub, self.dim)
                tempFitness = self.CaculateFitness1(
                    Xnew[i, :], self.fitnessFunction)
                if (tempFitness <= fitness[i]):
                    fitness[i] = tempFitness
                    X[i, :] = Xnew[i, :]
            for i in range(self.N):
                X[i, :] = self.BorderCheck1(
                    X[i, :], self.lb, self.ub, self.dim)
            # Sort fitness values.
            Ybest, index = self.SortFitness(fitness)
            if (Ybest[0] <= GbestScore):
                GbestScore = Ybest[0]     # Update the global optimal solution.
                # Sort fitness values
                GbestPositon[0, :] = X[index[0], :]
            Curve[t] = GbestScore

            median = np.median(X, axis=0)
            sumation = 0
            for x in X:
                sumation += np.abs(median-x)
            divj = sumation / self.N
            deviation.append(sum(divj)/self.dim)

        for divt in deviation:
            exploration.append(divt / np.max(deviation) * 100)
            exploitation.append(
                np.abs(divt - np.max(deviation)) / np.max(deviation) * 100)

        return GbestPositon, GbestScore, Curve, (exploration, exploitation)
