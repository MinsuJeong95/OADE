import pickle
import matplotlib.pyplot as plt
import os

import torch
import torch.nn as nn
import numpy as np

from sklearn import svm
from sklearn.preprocessing import MinMaxScaler
from collections import defaultdict

def rankClaculate(datasetType, modelType, Fold):
    path = './' + datasetType + '\\' + 'testResult' + '\\' + modelType + '\\' + Fold
    pickleFilePaths = os.listdir(path + '/epochTermTest')

    for pickleFilePath in pickleFilePaths:
        with open(path + '/epochTermTest' + '/' + pickleFilePath, 'rb') as fr:
            loadReIDdict = pickle.load(fr)

        folderPath = path + '/testResultGraph' + '/' + pickleFilePath.split('.')[0] + '/RankScore'
        try:
            if not os.path.exists(folderPath):
                os.makedirs(folderPath)

        except OSError:
            print('Error')

        f = open(folderPath + "/RankScore" + ".txt", 'w')
        f.write(pickleFilePath.split('.')[0] + '\n')
        rankWide = 20
        Rank = [0]*rankWide
        classRank = []
        rankScore = 0
        tmpcnt = 0
        ReIDdict = loadReIDdict
        lastScore = []
        m = nn.Softmax(dim=1)

        for key in ReIDdict:
            score = ReIDdict[key]

            for i in range(len(score)):
                reScore = torch.tensor(score[i][0])
                reScore = reScore.reshape((1, 2))
                reScore = m(reScore).squeeze()
                score[i][0] = reScore.tolist()[1]

            score.sort(key=lambda x : -x[0])

            for rankCnt in range(rankWide) :
                if score[rankCnt][1] == 1:
                    for i in range(rankCnt, rankWide) :
                        Rank[i] = Rank[i] + 1
                    break

        for i in range(rankWide):
            Rank[i] = Rank[i]/(len(ReIDdict))

        rankResult = [0]*rankWide
        for i in range(rankWide):
            rankResult[i] = int(Rank[i]*100*1000)
            rankResult[i] = rankResult[i]*0.001
            f.write('Rank'+str(i+1)+' : '+str(rankResult[i]) + '\n')

        print(pickleFilePath)

        foldType = Fold

        plt.figure()
        plt.axis([1, rankWide, 0, 100])
        plt.grid(True, axis='y')
        plt.plot(range(1, rankWide+1), rankResult, 'ro', label=modelType)
        plt.plot(range(1, rankWide+1), rankResult, 'r')
        plt.xlabel('Rank')
        plt.ylabel('TPR')
        plt.xticks(range(1,rankWide+1))
        plt.title('RankScore_' + foldType + '_' + modelType)
        plt.legend(loc='lower right')
        plt.savefig(folderPath + '/RankScore' + '.png')

        plt.close()
        f.close()
