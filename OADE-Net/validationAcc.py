import pickle
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import shutil

def mySoftmax(a) :
    c = np.max(a)
    exp_a = np.exp(a-c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y

def accCalculate(datasetType, modelType, Fold, imgType) :

    path = '.\\' + datasetType + '\\valResult\\' + modelType + '\\' + Fold + '\\' + imgType
    pickleFilePath = os.listdir(path + '\\epochTermValidation')

    # 피클list
    modelNum = []
    for i in range(len(pickleFilePath)):  # 피클list 재배열
        modelName = pickleFilePath[i].split('_')
        modelNum.append(int(modelName[len(modelName) - 1].split('.')[0]))
    modelNum.sort()
    for num_i in range(len(modelNum)):  # 재배열 적용
        modelName = pickleFilePath[num_i].split('_')
        del modelName[len(modelName) - 1]
        modelName.append(str(modelNum[num_i]) + '.pickle')
        saveName = ''
        for name_i in range(len(modelName)):  # 나눠져 있는 list 재 연결
            if name_i == (len(modelName) - 1):
                saveName = saveName + modelName[name_i]
            else:
                saveName = saveName + modelName[name_i] + '_'
        pickleFilePath[num_i] = saveName

    print(len(pickleFilePath))

    thresholding = [0.5]
    Accuracy = []
    lossArray = []
    for fileCnt in range(len(pickleFilePath)):
        with open(path + '\\epochTermValidation'+'/'+pickleFilePath[fileCnt], 'rb') as fr:
            loadReIDdict = pickle.load(fr)
        ReIDdict = loadReIDdict

        folderPath = path + '\\valResultGraph' + '/' + pickleFilePath[fileCnt].split('.')[0]
        try:
            if not os.path.exists(folderPath):
                os.makedirs(folderPath)
        except OSError:
            print('Error')

        f = open(folderPath+"/uncorrected_" + pickleFilePath[fileCnt].split('.')[0] + ".txt", 'w')

        for t, threshold in enumerate(thresholding):
            TP = 0
            TN = 0
            FP = 0
            FN = 0
            Recall = 0
            Precision = 0
            totalLoss = 0
            f.write('threshold : ' + str(threshold) + '\n')

            for i, (key, value) in enumerate(ReIDdict.items()):
                for valueSize in range(len(value)):
                    score = np.array(value[valueSize][0])
                    score = mySoftmax(score)
                    score = score[1]
                    loss = value[valueSize][1]

                    label = value[valueSize][2]
                    imageName = value[valueSize][3]

                    if score >= threshold:
                        predict = 1
                    else:
                        predict = 0

                    if predict == 1 and label == 1:
                        TP = TP + 1
                    elif predict == 0 and label == 0:
                        TN = TN + 1
                    elif predict == 1 and label == 0:
                        FP = FP + 1
                        f.write(imageName + '\n')
                    elif predict == 0 and label == 1:
                        FN = FN + 1
                        f.write(imageName + '\n')

                    totalLoss += loss

            acc = (TP + TN) / (TP + FN + FP + TN)
            Accuracy.append(acc)
            print('Accuracy : ', acc)
            f.write('Accuracy : ' + str(acc) + '\n')

            loss = totalLoss / (TP + FN + FP + TN)
            lossArray.append(loss)
            print('Loss : ', loss)
            f.write('Loss : ' + str(loss) + '\n')

        plt.close()
        f.close()

    plt.figure()
    plt.axis([1, len(pickleFilePath), 0, 1])
    plt.grid(True)
    plt.plot(range(1, len(pickleFilePath) + 1), Accuracy, label='Accuracy')
    plt.plot(range(1, len(pickleFilePath) + 1), lossArray, label='Loss')

    labelType = imgType
    foldType = Fold

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy_Loss')
    plt.title('Validation_' + foldType + '_' + labelType + '_' + modelType)
    plt.legend(loc='center right', ncol=1)

    folderPath = '.\\' + datasetType + '\\valResult\\' + modelType + '\\' + Fold + '\\' + imgType + '\\valResultGraph'
    if not os.path.isdir(folderPath):
        os.makedirs(folderPath)
    plt.savefig(folderPath + '/Val_' + foldType + '_' + labelType + '_' + modelType + '.png')
    plt.close()

    np.save(folderPath + '/Val_' + foldType + '_' + labelType + '_' + modelType, np.array(Accuracy))

    # 모델 select
    accMax = max(Accuracy)
    modelSelect = 0
    for i in range(len(Accuracy)):
        if Accuracy[i] == accMax:
            modelSelect = i
            break
    trainPath = '.\\' + datasetType + '\\trainModels\\' + modelType + '\\' + Fold + '\\' + imgType + '\epochTermModel'
    modelPath = os.listdir(trainPath)

    # 모델list
    modelNum = []
    for i in range(len(modelPath)):  # 모델list 재배열
        modelName = modelPath[i].split('_')
        modelNum.append(int(modelName[len(modelName) - 1].split('.')[0]))
    modelNum.sort()
    for num_i in range(len(modelNum)):  # 재배열 적용
        modelName = modelPath[num_i].split('_')
        del modelName[len(modelName) - 1]
        modelName.append(str(modelNum[num_i]) + '.pth')
        saveName = ''
        for name_i in range(len(modelName)):  # 나눠져 있는 list 재 연결
            if name_i == (len(modelName) - 1):
                saveName = saveName + modelName[name_i]
            else:
                saveName = saveName + modelName[name_i] + '_'
        modelPath[num_i] = saveName
    selectTrainPath = path + '\\selectEpoch\\'
    if not os.path.isdir(selectTrainPath):
        os.makedirs(selectTrainPath)
    shutil.copyfile(trainPath + '\\' + modelPath[modelSelect], selectTrainPath + modelPath[modelSelect])









