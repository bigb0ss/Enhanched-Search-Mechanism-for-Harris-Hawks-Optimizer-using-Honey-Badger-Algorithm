from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder,  StandardScaler
from imblearn.over_sampling import SMOTE
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tabnanny import verbose
from HHO import HHO
from HBA import HBA
from HBHHO import HBHHO

import warnings
warnings.filterwarnings('ignore')


LB = 0
UB = 1
DIM = 2  # We are trying to optimize 2 hyperparameters
N = 10
ITERS = 10
RUNS = 10
# Setting the random seed for all the algorithms
SEED = 208001


def optimzeHyperParameters():

    hho_fitness_list = []
    hba_fitness_list = []
    hbhho_fitness_list = []

    hho_results = []
    hba_results = []
    hbhho_results = []

    hho_explore = []
    hba_explore = []
    hbhho_explore = []

    hho_exploit = []
    hba_exploit = []
    hbhho_exploit = []
    for run in tqdm(range(RUNS)):
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

    fig, ax = plt.subplots(1, figsize=(10, 6))
    ax.plot(np.mean(hho_results, axis=0))
    ax.plot(np.mean(hba_results, axis=0))
    ax.plot(np.mean(hbhho_results, axis=0))

    plt.title("HHO vs HBA vs HBHHO - Convergence Graph : "+OBJECTIVE.__name__)
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Fitness value")
    ax.set_yscale('log')
    ax.legend(['HHO', 'HBA', 'HBHHO'])
    # plt.savefig(output_directory + "Convergence graph.svg")
    plt.savefig(OBJECTIVE.__name__+".svg")
    # plt.show()
    # print(hho_position, hho_fitness)
    # print(hba_position, hba_fitness)
    # print(hbhho_position, hbhho_fitness)
    return hho_position, hba_position, hbhho_position


trainData = pd.read_csv('train.csv')
# Droping unwanted features
trainData.drop(columns=['PassengerId', 'Name',
               'Ticket', 'Cabin'], inplace=True)

encoder = LabelEncoder()
encoder.fit(trainData['Sex'])
trainData['Sex'] = encoder.transform(trainData['Sex'])


encoder.fit(trainData['Embarked'])
trainData['Embarked'] = encoder.transform(trainData['Embarked'])

trainData.fillna(trainData.median(), inplace=True)


scaler = StandardScaler()

xtrain = trainData.iloc[:, 1:].values
ytrain = trainData.iloc[:, 0].values

scaler.fit(xtrain)
xtrain = scaler.transform(xtrain)

xtrain, xval, ytrain, yval = train_test_split(
    xtrain, ytrain, test_size=0.2, random_state=SEED)

sm = SMOTE(sampling_strategy='minority', random_state=SEED)
sampledXtrain, sampledytrain = sm.fit_resample(xtrain, ytrain)

# Objective function for SVM hyper parameter tuning


def svmObjective(x):

    if np.isnan(x[0]):
        return 1
    if np.isnan(x[1]):
        return 1

    param1 = 'rbf' if round(x[0]) == 1 else 'linear'
    param2 = x[1] % (10-1)+1

    # print(param2)

    clf = SVC(kernel=param1, C=param2)

    clf.fit(sampledXtrain, sampledytrain)

    preds = clf.predict(xval)
    acc = accuracy_score(yval, preds)

    return 1-acc

# Evaluation funtion for SVM


def svmEvaluate(x):

    if np.isnan(x[0]):
        return 1
    if np.isnan(x[1]):
        return 1

    param1 = 'rbf' if round(x[0]) == 1 else 'linear'
    param2 = x[1] % (10-1)+1

    print("SVM")
    print("kernel: ", param1, "C: ", param2)

    clf = SVC(kernel=param1, C=param2)

    clf.fit(sampledXtrain, sampledytrain)

    preds = clf.predict(xval)
    acc = accuracy_score(yval, preds)

    return acc

# Objective function for random forest hyper parameter tuning


def rfObjective(x):

    if np.isnan(x[0]):
        return 1
    if np.isnan(x[1]):
        return 1

    param1 = 'gini' if round(x[0]) == 1 else 'entropy'
    param2 = round(x[1] % (100-10)+10)

    # print(param2)

    clf = RandomForestClassifier(criterion=param1, n_estimators=param2)

    clf.fit(sampledXtrain, sampledytrain)

    preds = clf.predict(xval)
    acc = accuracy_score(yval, preds)

    return 1-acc

# Evaluation function for Random forest


def rfEvaluate(x):

    if np.isnan(x[0]):
        return 1
    if np.isnan(x[1]):
        return 1

    param1 = 'gini' if round(x[0]) == 1 else 'entropy'
    param2 = round(x[1] % (100-10)+10)

    print("Random Forest")
    print("criterion: ", param1, "n_esitomators: ", param2)

    clf = RandomForestClassifier(criterion=param1, n_estimators=param2)

    clf.fit(sampledXtrain, sampledytrain)

    preds = clf.predict(xval)
    acc = accuracy_score(yval, preds)

    return acc

# Objective function for decision tree hyper parameter tuning


def dtObjective(x):

    if np.isnan(x[0]):
        return 1
    if np.isnan(x[1]):
        return 1

    param1 = 'gini' if round(x[0]) == 1 else 'entropy'
    param2 = round(x[1] * (10-1)+1)

    if param2 <= 0:
        return 1

    clf = DecisionTreeClassifier(criterion=param1, max_depth=param2)

    clf.fit(sampledXtrain, sampledytrain)

    preds = clf.predict(xval)
    acc = accuracy_score(yval, preds)

    return 1-acc


# Evaluate Function for decision tree
def dtEvaluate(x):

    if np.isnan(x[0]):
        return 1
    if np.isnan(x[1]):
        return 1

    param1 = 'gini' if round(x[0]) == 1 else 'entropy'
    param2 = round(x[1] * (10-1)+1)

    if param2 <= 0:
        return 1

    print("Decision Tree")
    print("criterion: ", param1, "max_depth: ", param2)
    clf = DecisionTreeClassifier(criterion=param1, max_depth=param2)

    clf.fit(sampledXtrain, sampledytrain)

    preds = clf.predict(xval)
    acc = accuracy_score(yval, preds)

    return acc

# Objective function for xgboost hyperparameter tuning


def xgbObjective(x):

    if np.isnan(x[0]):
        return 1
    if np.isnan(x[1]):
        return 1

    param1 = 'gblinear' if round(x[0]) == 1 else 'gbtree'
    param2 = round(x[1] % (10-1)+1)

    # print(param2)

    clf = XGBClassifier(booster=param1, max_depth=param2, verbosity=0)

    clf.fit(sampledXtrain, sampledytrain)

    preds = clf.predict(xval)
    acc = accuracy_score(yval, preds)

    return 1-acc


# SVM Objective
svmObjective.__name__ = "SVM"
rfObjective.__name__ = "Random Forest"
dtObjective.__name__ = "Decision Tree"


OBJECTIVE = svmObjective
hho_pos, hba_pos, hbhho_pos = optimzeHyperParameters()

svm_acc1=svmEvaluate(hho_pos)
svm_acc2=svmEvaluate(hba_pos[0])
svm_acc3=svmEvaluate(hbhho_pos)

print("HHO : ",svm_acc1)
print("HBA : ",svm_acc2)
print("HBHHO : ",svm_acc3)

OBJECTIVE = rfObjective
hho_pos, hba_pos, hbhho_pos =optimzeHyperParameters()

rf_acc1=svmEvaluate(hho_pos)
rf_acc2=svmEvaluate(hba_pos[0])
rf_acc3=svmEvaluate(hbhho_pos)


print("HHO : ",rf_acc1)
print("HBA : ",rf_acc2)
print("HBHHO : ",rf_acc3)


OBJECTIVE = dtObjective
hho_pos, hba_pos, hbhho_pos =optimzeHyperParameters()


dt_acc1=svmEvaluate(hho_pos)
dt_acc2=svmEvaluate(hba_pos[0])
dt_acc3=svmEvaluate(hbhho_pos)


print("HHO : ",dt_acc1)
print("HBA : ",dt_acc2)
print("HBHHO : ",dt_acc3)


# OBJECTIVE = xgbObjective
# optimzeHyperParameters()
