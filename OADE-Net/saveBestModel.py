import os
from collections import defaultdict
import shutil
# Folds = ['Fold1',
#         'Fold2']
# imgTypes = ['IPVT1',
#            'IPVT2',
#            'IIPVT']
# modelTypes = ['original',
#              'attention']
# datasetTypes = ['RegDB_thermal',
#                'SYSU-MM01_thermal']

def saveBestModel(Folds, imgTypes, modelTypes, datasetTypes):
    rankScore = defaultdict(list)
    scoreTmp = []
    for datasetType in datasetTypes:
        for modelType in modelTypes:
            for imgType in imgTypes:
                for Fold in Folds:
                    path = './' + datasetType + '\\' + 'testResult' + '\\' + modelType + '\\' + Fold + '\\' + imgType
                    pickleFilePath = os.listdir(path + '/epochTermTest')[0]

                    folderPath = path + '/testResultGraph' + '/' + pickleFilePath.split('.')[0] + '/RankScore'
                    f = open(folderPath + "/RankScore" + ".txt", 'r')
                    readRank = f.readlines()
                    Rank1Score = readRank[1].split(' ')[2]
                    scoreTmp.append(float(Rank1Score.split('\n')[0]))
                    Rank10Score = readRank[10].split(' ')[2]
                    scoreTmp.append(float(Rank10Score.split('\n')[0]))
                    Rank20Score = readRank[20].split(' ')[2]
                    scoreTmp.append(float(Rank20Score.split('\n')[0]))

                    f.close()
                rankScore[imgType].append((scoreTmp[0] + scoreTmp[3])/2)
                rankScore[imgType].append((scoreTmp[1] + scoreTmp[4])/2)
                rankScore[imgType].append((scoreTmp[2] + scoreTmp[5])/2)
                scoreTmp = []
            sortRankScore = sorted(rankScore.items(), key=(lambda item: item[1]), reverse=True)
            bestType = sortRankScore[0][0]
            rankScore = defaultdict(list)

            #Save best model
            for Fold in Folds:
                loadPath = './' + datasetType + '\\' + 'valResult' + '\\' + modelType + '\\' + Fold + '\\' + bestType + \
                           '\\selectEpoch'
                loadModel = os.listdir(loadPath)
                savePath = './' + datasetType + '\\' + 'bestModel' + '\\' + modelType + '\\' + Fold + '\\' + bestType
                if not os.path.isdir(savePath):
                    os.makedirs(savePath)
                shutil.copyfile(loadPath + '\\' + loadModel[0], savePath + '\\' + loadModel[0])

