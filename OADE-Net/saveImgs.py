import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import os

import imgPreprocess
import CustomDataset

def positiveSaveClassUp(camiter, device):
    camImgs = []
    camNames = []
    try:
        camData = camiter.next()
    except StopIteration:
        return [-1], [-1], -1

    camImgs.append(camData['image'].to(device))
    camNames.append(camData['filename'][0])
    camName = camNames[0].split('\\')
    path = str(camName[0]) + '\\' + str(camName[1]) + '\\' + str(camName[2]) + '\\' + str(
        camName[3]) + '\\' + \
           str(camName[4]) + '\\' + str(camName[5]) + '\\' + str(camName[6]) + '\\' + str(camName[7])
    camList = os.listdir(path)

    for camI in range(len(camList) - 1):
        camData = camiter.next()
        camImgs.append(camData['image'].to(device))
        camNames.append(camData['filename'][0])
    return camImgs, camNames, camName

def positiveSave(imgType, DBPath):
    imgPre = imgPreprocess.imgPreprocess()
    batchSize = 1
    loadDBPath = DBPath + '\\allcam'
    saveDBPath = DBPath + '\\' + imgType + '\\' + 'positive\\'
    if not os.path.isdir(saveDBPath):
        os.makedirs(saveDBPath)
    checkImages = os.listdir(saveDBPath)

    if len(checkImages) != 0:
        print('Positive images already have. Check your positive dataset')
        print('path : ' + saveDBPath)
        return 0

    trans = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Grayscale()])

    allCamProbTestset = CustomDataset.CustomDataset(root_dir=loadDBPath,
                                                    transforms=trans)
    allCamProbLoader = DataLoader(allCamProbTestset, batch_size=1, shuffle=False, pin_memory=True)
    allCamGalleryTestset = CustomDataset.CustomDataset(root_dir=loadDBPath,
                                                       transforms=trans)
    allCamGalleryLoader = DataLoader(allCamGalleryTestset, batch_size=batchSize, shuffle=False, pin_memory=True)

    device = torch.device("cpu")

    saveImgType = imgType

    imgCnt = 0
    probiter = iter(allCamProbLoader)
    galiter = iter(allCamGalleryLoader)

    probName = 0
    probImgs = []
    probNames = []
    probImgs, probNames, probName = positiveSaveClassUp(probiter, device)

    galName = 0
    galImgs = []
    galNames = []
    galImgs, galNames, galName = positiveSaveClassUp(galiter, device)

    while True:
        if int(probName[7]) == int(galName[7]):
            for prob_i in range(len(probImgs)):
                for gal_i in range(len(galImgs)):
                    if probNames[prob_i] == galNames[gal_i]:
                        continue
                    inputImgs = imgPre.preprocess(probImgs[prob_i], galImgs[gal_i], saveImgType, device)
                    # imgPre.viewTensorImg(inputImgs)
                    imgCnt = imgCnt + 1
                    probRealName = probNames[prob_i].split('\\')
                    probSaveName = probRealName[6] + '_' + probRealName[7] + '_' + probRealName[8].split('.')[0]
                    galRealName = galNames[gal_i].split('\\')
                    galSaveName = galRealName[6] + '_' + galRealName[7] + '_' + galRealName[8].split('.')[0]
                    imgPre.saveImg(inputImgs, saveDBPath, probSaveName, galSaveName, 1, imgCnt)

            probImgs = []
            probNames = []
            probImgs, probNames, probName = positiveSaveClassUp(probiter, device)

            galImgs = []
            galNames = []
            galImgs, galNames, galName = positiveSaveClassUp(galiter, device)

        elif int(probName[7]) < int(galName[7]):
            probImgs = []
            probNames = []
            probImgs, probNames, probName = positiveSaveClassUp(probiter, device)


        elif int(probName[7]) > int(galName[7]):
            galImgs = []
            galNames = []
            galImgs, galNames, galName = positiveSaveClassUp(galiter, device)

        if galName == -1 or probName == -1:
            break

def negativeSave(imgType, DBPath):
    imgPre = imgPreprocess.imgPreprocess()
    batchSize = 1
    trans = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Grayscale()])
    loadDBPath = DBPath + '\\allcam'
    saveDBPath = DBPath + '\\' + imgType + '\\' + 'negative\\'
    if not os.path.isdir(saveDBPath):
        os.makedirs(saveDBPath)
    checkImages = os.listdir(saveDBPath)

    if len(checkImages) != 0:
        print('Negative images already have. Check your negative dataset')
        print('path : ' + saveDBPath)
        return 0

    allCamProbTestset = CustomDataset.CustomDataset(root_dir=loadDBPath,
                                                    transforms=trans)
    allCamProbLoader = DataLoader(allCamProbTestset, batch_size=1, shuffle=True, pin_memory=True)
    allCamGalleryTestset = CustomDataset.CustomDataset(root_dir=loadDBPath,
                                                       transforms=trans)
    allCamGalleryLoader = DataLoader(allCamGalleryTestset, batch_size=batchSize, shuffle=True, pin_memory=True)

    device = torch.device("cpu")

    imgCnt = 0
    probiter = iter(allCamProbLoader)
    galiter = iter(allCamGalleryLoader)

    saveImgType = imgType
    path = DBPath + '\\' + imgType + '\\' + '\\positive'
    positiveList = os.listdir(path)
    print(len(positiveList))

    while imgCnt < len(positiveList):
        try:
            probData = probiter.next()
        except StopIteration:
            probiter = iter(allCamProbLoader)
            probData = probiter.next()

        probImgs = probData['image'].to(device)
        probName = probData['filename'][0].split('\\')
        probRealName = probName[7]
        probSaveName = probName[6] + '_' + probName[7] + '_' + probName[8].split('.')[0]

        try:
            galData = galiter.next()
        except StopIteration:
            galiter = iter(allCamGalleryLoader)
            galData = galiter.next()

        galImgs = galData['image'].to(device)
        galName = galData['filename'][0].split('\\')
        galRealName = galName[7]
        galSaveName = galName[6] + '_' + galName[7] + '_' + galName[8].split('.')[0]

        if galRealName == probRealName:
            continue

        inputImgs = imgPre.preprocess(probImgs, galImgs, saveImgType, device)
        imgCnt = imgCnt+1
        imgPre.saveImg(inputImgs, saveDBPath,probSaveName, galSaveName, 0, imgCnt)


