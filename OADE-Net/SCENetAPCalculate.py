import pickle
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os
import torch

from sklearn import svm
from sklearn.preprocessing import MinMaxScaler
from collections import defaultdict


def apCalculate(datasetType, modelType, Fold):

    path = './' + datasetType + '\\' + 'testResult' + '\\' + modelType + '\\' + Fold
    pickleFilePaths = os.listdir(path + '/epochTermTest')
    for pickleFilePath in pickleFilePaths:
        folderPath = path + '/testResultGraph' + '/' + pickleFilePath.split('.')[0] + '/apCalculate'
        try:
            if not os.path.exists(folderPath):
                os.makedirs(folderPath)
        except OSError:
            print('Error')

        m = nn.Softmax(dim=1)
        mAP = 0

        with open(path + '/epochTermTest' + '/' + pickleFilePath, 'rb') as fr:
            loadReIDdict = pickle.load(fr)
        ReIDdict = loadReIDdict

        for i, (key, value) in enumerate(ReIDdict.items()):
            for valueSize in range(len(value)):
                score = torch.tensor(value[valueSize][0])
                score = score.reshape((1, 2))
                score = m(score).squeeze()
                score = score[1]
                ReIDdict[key][valueSize][0] = score

        for i, (key, value) in enumerate(ReIDdict.items()):

            classPath = folderPath + '/' + str(key)
            try:
                if not os.path.exists(classPath):
                    os.makedirs(classPath)
            except OSError:
                print('Error')

            graphPath = classPath + '/PRCurve'
            try:
                if not os.path.exists(graphPath):
                    os.makedirs(graphPath)
            except OSError:
                print('Error')

            uncorrectedPath = classPath + '/uncorrectedData'
            try:
                if not os.path.exists(uncorrectedPath):
                    os.makedirs(uncorrectedPath)
            except OSError:
                print('Error')

            numpyPath = classPath + '/PRdatas'
            try:
                if not os.path.exists(numpyPath):
                    os.makedirs(numpyPath)
            except OSError:
                print('Error')

            allReIdValues = []

            threshold = 0.5

            for valueSize in range(len(value)):
                score = value[valueSize][0]
                label = value[valueSize][1]
                imgName = value[valueSize][2]
                # if score >= threshold and label == 0:
                #     f.write(imgName + ' (FP)' + '\n')
                # elif score < threshold and label == 1:
                #     f.write(imgName + ' (FN)' + '\n')

                allReIdValues.append(value[valueSize])
            allReIdValues.sort(key=lambda x: -x[0])

            TP = 0
            FP = 0
            Recall = []
            Precision = []
            trueLabelCnt = 0

            for i, ReIDvalue in enumerate(allReIdValues):
                label = ReIDvalue[1]
                if label == 1:
                    trueLabelCnt = trueLabelCnt + 1

            for t, ReIDvalue in enumerate(allReIdValues):
                label = ReIDvalue[1]
                if label == 1:
                    TP = TP + 1
                elif label == 0:
                    FP = FP + 1

                if TP + FP == 0:
                    Precision.append(0)
                else:
                    Precision.append(TP / (TP + FP))

                if trueLabelCnt == 0:
                    Recall.append(0)
                else:
                    Recall.append(TP / trueLabelCnt)

            apPrecision = Precision
            apRecall = Recall

            plt.figure()
            plt.axis([min(apRecall), max(apRecall), 0, 1])
            plt.grid(True)
            plt.plot(apRecall, apPrecision)

            # AP = 0
            # for i in range(11):  # 11point 보간법
            #     for recall_i in range(len(apRecall)):
            #         if int(apRecall[recall_i] * 10) == i:
            #             precisionTmp = apPrecision[recall_i:]
            #             maxValue = max(precisionTmp)
            #
            #             AP = AP + maxValue
            #             break
            # AP = AP / 11
            AP = 0
            for apRecall_i in range(len(apRecall)):
                if apRecall_i == 0:
                    AP = AP + (apPrecision[apRecall_i] * apRecall[apRecall_i])
                else:
                    AP = AP + (apPrecision[apRecall_i] * (apRecall[apRecall_i] - apRecall[apRecall_i - 1]))

            mAP = mAP + AP

            AP = int(AP * (10000))
            AP = float(AP) / (100)

            foldType = Fold

            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(
                'PRCurve_' + foldType + '_' + str(key) + ' AP : ' + str(AP))
            plt.fill_between(apRecall[0:len(apRecall)], apPrecision[0:len(apPrecision)], alpha=0.2)

            plt.savefig(
                graphPath + '/PRCurve_' + foldType + '_' + modelType + '_' + str(key) + '_' + str(
                    AP) + '.png')
            np.save(numpyPath + '/apRecall_' + foldType + '_' + modelType + '_' + str(key),
                    np.array(apRecall))
            np.save(numpyPath + '/apPrecision_' + foldType + '_' + modelType + '_' + str(key),
                    np.array(apPrecision))

            plt.close()

        f = open(folderPath + "/mAP" + ".txt", 'w')
        f.write(pickleFilePath.split('.')[0] + '\n')
        mAP = mAP / len(ReIDdict)
        f.write('mAP : ' + str(mAP) + '\n')
        f.close()
