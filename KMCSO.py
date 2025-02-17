# Import necessary libraries
import os
from opfunu.cec_based.cec2022 import *
import numpy as np
from copy import deepcopy
from sklearn.cluster import KMeans

PopSize = 200
DimSize = 100
LB = [-100] * DimSize
UB = [100] * DimSize
TrialRuns = 20
MaxFEs = 1000 * DimSize

Pop = np.zeros((PopSize, DimSize))
Velocity = np.zeros((PopSize, DimSize))
FitPop = np.zeros(PopSize)
curFEs = 0
FuncNum = 1
curIter = 0
MaxIter = int(MaxFEs / PopSize * 2)
phi = 0.1
K = 2  # Number of clusters for K-Means

# Initialize population
def Initialization(func):
    global Pop, Velocity, FitPop
    Velocity = np.zeros((PopSize, DimSize))
    for i in range(PopSize):
        for j in range(DimSize):
            Pop[i][j] = LB[j] + (UB[j] - LB[j]) * np.random.rand()
        FitPop[i] = func(Pop[i])

# Check boundaries
def Check(indi):
    global LB, UB
    for i in range(DimSize):
        range_width = UB[i] - LB[i]
        if indi[i] > UB[i]:
            n = int((indi[i] - UB[i]) / range_width)
            mirrorRange = (indi[i] - UB[i]) - (n * range_width)
            indi[i] = UB[i] - mirrorRange
        elif indi[i] < LB[i]:
            n = int((LB[i] - indi[i]) / range_width)
            mirrorRange = (LB[i] - indi[i]) - (n * range_width)
            indi[i] = LB[i] + mirrorRange
    return indi

# KMCSO algorithm with modified global competition
def KMCSO(func):
    global Pop, Velocity, FitPop, phi, K

    # Perform K-Means clustering
    kmeans = KMeans(n_clusters=K, random_state=0).fit(Pop)
    labels = kmeans.labels_
    best_points = []  # Store the best individual in each cluster

    Off = deepcopy(Pop)
    FitOff = deepcopy(FitPop)

    # Intra-cluster competition (excluding best individual in each cluster)
    for cluster in range(K):
        cluster_indices = np.where(labels == cluster)[0]
        best_idx = cluster_indices[np.argmin(FitPop[cluster_indices])]
        best_points.append(best_idx)  # Store the best point in this cluster

        # Exclude the best point from local competition
        cluster_indices = np.delete(cluster_indices, np.where(cluster_indices == best_idx))

        np.random.shuffle(cluster_indices)
        for i in range(1, len(cluster_indices), 2):
            idx1, idx2 = cluster_indices[i-1], cluster_indices[i]
            if FitPop[idx1] < FitPop[idx2]:
                Off[idx1] = deepcopy(Pop[idx1])
                FitOff[idx1] = FitPop[idx1]
                Velocity[idx2] = np.random.rand(DimSize) * Velocity[idx2] + np.random.rand(DimSize) * (
                        Pop[idx1] - Pop[idx2]) + phi * (Pop[best_idx] - Pop[idx2])
                Off[idx2] = Pop[idx2] + Velocity[idx2]
                Off[idx2] = Check(Off[idx2])
                FitOff[idx2] = func(Off[idx2])
            else:
                Off[idx2] = deepcopy(Pop[idx2])
                FitOff[idx2] = FitPop[idx2]
                Velocity[idx1] = np.random.rand(DimSize) * Velocity[idx1] + np.random.rand(DimSize) * (
                            Pop[idx2] - Pop[idx1]) + phi * (Pop[best_idx] - Pop[idx1])
                Off[idx1] = Pop[idx1] + Velocity[idx1]
                Off[idx1] = Check(Off[idx1])
                FitOff[idx1] = func(Off[idx1])

    # Global competition using the best individuals from each cluster
    for i in range(0, K - 1, 2):
        idx1, idx2 = best_points[i], best_points[i + 1]
        if FitPop[idx1] < FitPop[idx2]:
            Velocity[idx2] = np.random.rand(DimSize) * Velocity[idx2] + np.random.rand(DimSize) * (
                    Pop[idx1] - Pop[idx2]) + phi * (Pop[idx1] - Pop[idx2])
            Off[idx2] = Pop[idx2] + Velocity[idx2]
            Off[idx2] = Check(Off[idx2])
            FitOff[idx2] = func(Off[idx2])
        else:
            Velocity[idx1] = np.random.rand(DimSize) * Velocity[idx1] + np.random.rand(DimSize) * (
                    Pop[idx2] - Pop[idx1]) + phi * (Pop[idx2] - Pop[idx1])
            Off[idx1] = Pop[idx1] + Velocity[idx1]
            Off[idx1] = Check(Off[idx1])
            FitOff[idx1] = func(Off[idx1])

    Pop = deepcopy(Off)
    FitPop = deepcopy(FitOff)

# Run KMCSO algorithm
def RunKMCSO(func):
    global FitPop, curIter, TrialRuns, DimSize
    All_Trial_Best = []
    for i in range(TrialRuns):
        BestList = []
        curIter = 0
        np.random.seed(2023 + 23 * i)
        Initialization(func)
        BestList.append(min(FitPop))
        while curIter < MaxIter:
            KMCSO(func)
            curIter += 1
            BestList.append(min(FitPop))
        All_Trial_Best.append(BestList)
    np.savetxt("./KMCSO2/KMCSO2_Data/CEC2022/F" + str(FuncNum + 1) + "_" + str(DimSize) + "D.csv", All_Trial_Best, delimiter=",")

# Main function
def main(dim):
    global FuncNum, DimSize, MaxFEs, MaxIter, Pop, LB, UB
    DimSize = dim
    Pop = np.zeros((PopSize, dim))
    MaxFEs = dim * 1000
    MaxIter = int(MaxFEs / PopSize * 2)
    LB = [-100] * dim
    UB = [100] * dim

    CEC2022 = [F12022(Dim), F22022(Dim), F32022(Dim), F42022(Dim), F52022(Dim), F62022(Dim),
               F72022(Dim), F82022(Dim), F92022(Dim), F102022(Dim), F112022(Dim), F122022(Dim)]

    for i in range(len(CEC2022)):
        FuncNum = i
        RunKMCSO(CEC2022[i].evaluate)

if __name__ == "__main__":
    if os.path.exists('./KMCSO2/KMCSO2_Data/CEC2022') == False:
        os.makedirs('./KMCSO2/KMCSO2_Data/CEC2022')
    Dims = [10, 20]
    for Dim in Dims:
        main(Dim)
