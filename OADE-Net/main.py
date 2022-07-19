import densenet
import densenetAttention
import saveImgs
import Training
import Validation
import validationAcc
import Test
import APCalculate
import RankCalculate

import saveBestModel
import saveScoreAndFeatureMap

import SCENet
import SCENetTraining
import SCENetValidation
import SCENetValidationAcc
import SCENetTest
import SCENetAPCalculate
import SCENetRankCalculate

def main():
    models = [densenet,
              densenetAttention]
    DBPath = 'I:\\JMS\\TrainingWork'
    Folds = ['Fold1',
            'Fold2']
    imgTypes = ['IPVT1',
               'IPVT2',
               'IIPVT']
    modelTypes = ['original',
                 'attention']
    datasetTypes = ['RegDB_thermal',
                   'SYSU-MM01_thermal']

    # Save training images before training
    for datasetType in datasetTypes:
        for imgType in imgTypes:
            for Fold in Folds:
                trainDBPath = DBPath + '\\' + datasetType + '\\' + Fold + '\\reTrain'
                saveImgs.positiveSave(imgType, trainDBPath)
                saveImgs.negativeSave(imgType, trainDBPath)

    # Run DenseNet
    for datasetType in datasetTypes:
        if datasetType == 'SYSU-MM01_thermal':
            geometricCentor = True
        else:
            geometricCentor = False

        for imgType in imgTypes:
            for model_i, modelType in enumerate(modelTypes):
                for Fold in Folds:
                    trainDBPath = DBPath + '\\' + datasetType + '\\' + Fold + '\\reTrain\\' + imgType
                    valDBPath = DBPath + '\\' + datasetType + '\\' + Fold + '\\reVal\\allcam'
                    testDBPath = DBPath + '\\' + datasetType + '\\' + Fold + '\\reTest\\allcam'

                    Training.training(datasetType, modelType, Fold, imgType, models[model_i], trainDBPath, numEpoch=10, startEpoch=0, lr=1e-3, wd=1e-4)
                    Validation.validation(datasetType, modelType, Fold, imgType, models[model_i], valDBPath, geometric=geometricCentor)
                    validationAcc.accCalculate(datasetType, modelType, Fold, imgType)
                    Test.test(datasetType, modelType, Fold, imgType, models[model_i], testDBPath)
                    APCalculate.apCalculate(datasetType, modelType, Fold, imgType)
                    RankCalculate.rankClaculate(datasetType, modelType, Fold, imgType)

    # Save best model
    saveBestModel.saveBestModel(Folds, imgTypes, modelTypes, datasetTypes)
    # Save SCE-Net Training Dataset
    saveScoreAndFeatureMap.saveScoreAndFeatureMap(datasetTypes, modelTypes, Folds, models, DBPath)

    for datasetType in datasetTypes:
        if datasetType == 'SYSU-MM01_thermal':
            geometricCentor = True
        else:
            geometricCentor = False

        for Fold in Folds:
            modelType = 'SCE-Net'
            sceNet = SCENet.SCENet()

            trainDBPath = DBPath + '\\' + datasetType + '\\' + Fold + '\\reTrain\\' + 'SCENetTrainData'
            valDBPath = DBPath + '\\' + datasetType + '\\' + Fold + '\\reVal\\' + 'allcam'
            testDBPath = DBPath + '\\' + datasetType + '\\' + Fold + '\\reTest\\' + 'allcam'

            SCENetTraining.training(datasetType, Fold, modelType, sceNet, trainDBPath, numEpoch=10, startEpoch=0, lr=1e-5, wd=1e-4)
            SCENetValidation.validation(datasetType, Fold, modelTypes, modelType, models, sceNet, valDBPath, geometricCentor)
            SCENetValidationAcc.accCalculate(datasetType, modelType, Fold)
            SCENetTest.test(datasetType, Fold, modelTypes, modelType, models, sceNet, testDBPath, geometricCentor)
            SCENetAPCalculate.apCalculate(datasetType, modelType, Fold)
            SCENetRankCalculate.rankClaculate(datasetType, modelType, Fold)

main()