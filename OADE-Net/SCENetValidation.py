import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as nnf
import cv2
import os
import torch.nn.functional as F

import densenet
import densenetAttention
import SCENet
import pickle
import imgPreprocess
import CustomDataset


gradient1 = []
gradient2 = []
gradient_att = []
def save_gradient1(*args):
    # print("original : ", "Gradient saved!!!!")
    grad_input = args[1]
    grad_output = args[2]
    # print(grad_output)
    # self.gradient.append(grad_output[0]) #backward
    gradient1.append(grad_output)  # forward
    # print(self.gradient[0].size())

def save_gradient2(*args):
    # print("attention : ", "Gradient saved!!!!")
    grad_input = args[1]
    grad_output = args[2]
    # print(grad_output)
    # self.gradient.append(grad_output[0]) #backward
    gradient2.append(grad_output)  # forward
    # print(self.gradient[0].size())

def save_gradient_att(*args):
    # print("CBAM : ", "Gradient saved!!!!")
    grad_input = args[1]
    grad_output = args[2]
    # print(grad_output)
    # self.gradient.append(grad_output[0]) #backward
    gradient_att.append(grad_output)  # forward
    # print(self.gradient[0].size())

gradient = []
def save_gradient(*args):
    # print("distanceCal : ", "Gradient saved!!!!")
    grad_input = args[1]
    grad_output = args[2]
    # print(grad_output)
    # self.gradient.append(grad_output[0]) #backward
    gradient.append(grad_output)  # forward
    # print(self.gradient[0].size())

def distenseCal(camiter, model, device, DBPath):
    feature = []
    distenceResult = []
    camImgs = []
    camNames = []
    result = 0
    resultImg = 0
    resultName = 0

    camData = camiter.next()
    camImgs.append(camData['image'].to(device))
    camNames.append(camData['filename'][0])
    camName = camNames[0].split('\\')
    path = ''
    for i in range(len(camName)-1):
        path = path + str(camName[i]) + '\\'

    imageList = os.listdir(path)
    #print(model.features[-1])
    h = model.features[-1].register_forward_hook(save_gradient)

    #Img Feature extract
    for i in range(len(imageList)):
        inputImg = camImgs[i]

        if inputImg.shape[1] == 1: #IPVT2일 경우 Channel 맞춰줌
            reShape = torch.cat([inputImg, inputImg, inputImg], dim=1)
            inputImg = reShape

        outputs = model(inputImg)


        out = F.relu(gradient[i], inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = torch.squeeze(out)

        feature.append(out)
        if i >= len(imageList) - 1:
            break
        camData = camiter.next()
        camImgs.append(camData['image'].to(device))
        camNames.append(camData['filename'][0])

    #Calculate Images Center
    for center in range(len(feature)):
        for f_i in range(len(feature)):
            if center == f_i:
                continue
            distence = (feature[center] - feature[f_i]) * (feature[center] - feature[f_i]) #distance

            for d_i in range(len(distence)):
                result = result + distence[d_i]
        result = result**(1/2)
        distenceResult.append(result)
        result = 0
    pickImg = min(distenceResult)
    for d_i in range(len(distenceResult)):
        if pickImg == distenceResult[d_i]:
            resultImg = camImgs[d_i]
            resultName = camNames[d_i]
            break
    h.remove()
    gradient.clear()
    gradient1.clear()
    return resultImg, resultName

def validation(datasetType, Fold, modelTypes, modelType, models, sceNet, DBPath, geometric):
    batchSize = 100

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(torch.__version__)
    print(device)

    imgPre = imgPreprocess.imgPreprocess()

    model1Type = modelTypes[0]
    model2Type = modelTypes[1]

    bestModel1Path = './' + datasetType + '\\bestModel' + '\\' + model1Type + '\\' + Fold
    model1ImgType = os.listdir(bestModel1Path)[0]
    bestModel1Path += '\\' + model1ImgType
    bestModel1Name = os.listdir(bestModel1Path)[0]
    model1ValDBPath = DBPath

    bestModel2Path = './' + datasetType + '\\bestModel' + '\\' + model2Type + '\\' + Fold
    model2ImgType = os.listdir(bestModel2Path)[0]
    bestModel2Path += '\\' + model2ImgType
    bestModel2Name = os.listdir(bestModel2Path)[0]
    model2ValDBPath = DBPath

    trans = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Grayscale()])
    allCamProbValset = CustomDataset.CustomDataset(root_dir=model1ValDBPath,
                                                   transforms=trans)
    allCamProbLoader = DataLoader(allCamProbValset, batch_size=1, shuffle=False, pin_memory=True)
    allCamGalleryValset = CustomDataset.CustomDataset(root_dir=model2ValDBPath,
                                                      transforms=trans)
    allCamGalleryLoader = DataLoader(allCamGalleryValset, batch_size=batchSize, shuffle=False, pin_memory=True)

    # Class 개수 계산
    if geometric == True:
        allCamProbClassLen = len(allCamProbValset.classes)  # geometric
    else:
        allCamProbClassLen = len(allCamProbValset)  # No geometric
    totalLen = allCamProbClassLen * len(allCamGalleryValset)

    model1 = models[0].densenet161(pretrained=True)
    model1.classifier = nn.Linear(2208, 2)
    model2 = models[1].densenet161(pretrained=True)
    model2.classifier = nn.Linear(2208, 2)

    model1.load_state_dict(torch.load(
        bestModel1Path + '\\' + bestModel1Name),
        strict=False)
    model2.load_state_dict(torch.load(
        bestModel2Path + '\\' + bestModel2Name),
        strict=False)

    model1.eval()
    model1.to(device)
    model2.eval()
    model2.to(device)
    m = nn.Softmax(dim=1)
    criterion = nn.CrossEntropyLoss().to(device)

    h1 = model1.features[-1].register_forward_hook(save_gradient1)
    h2 = model2.features[-1].register_forward_hook(save_gradient2)
    h_att = model2.features[-3].SpatialGate.spaSig.register_forward_hook(save_gradient_att)

    print('Fold : ' + Fold)
    path = '.\\' + datasetType + '\\trainModels\\' + modelType + '\\' + Fold + '\\' + 'epochTermModel'
    modelPaths = os.listdir(path)

    # 모델list
    modelNum = []
    for i in range(len(modelPaths)):  # 모델list 재배열
        modelName = modelPaths[i].split('_')
        modelNum.append(int(modelName[len(modelName) - 1].split('.')[0]))
    modelNum.sort()
    for num_i in range(len(modelNum)):  # 재배열 적용
        modelName = modelPaths[num_i].split('_')
        del modelName[len(modelName) - 1]
        modelName.append(str(modelNum[num_i]) + '.pth')
        saveName = ''
        for name_i in range(len(modelName)):  # 나눠져 있는 list 재 연결
            if name_i == (len(modelName) - 1):
                saveName = saveName + modelName[name_i]
            else:
                saveName = saveName + modelName[name_i] + '_'
        modelPaths[num_i] = saveName

    print(len(modelPaths))

    for modelPath in modelPaths:
        ReIDdict = {}
        progress = 0
        sceNet.load_state_dict(torch.load(path + '\\' + modelPath))
        sceNet.eval()  # 평가모드
        sceNet.to(device)

        with torch.no_grad():
            ReIDresult = []
            keyTmp = 0

            # allCamProb : 인식 / allCamGallery : 등록
            allCamProbiter = iter(allCamProbLoader)
            for probI in range(allCamProbClassLen):
                if geometric == True:
                    # geometric
                    probCenterImg, probCenterName = distenseCal(allCamProbiter, model1, device, DBPath)
                    probCenterName = probCenterName.split('\\')
                    probCenterNameLen = len(probCenterName)
                    probRealLabel = probCenterName[probCenterNameLen - 2]
                    probSaveName = probCenterName[probCenterNameLen - 3] + '_' + \
                                   probCenterName[probCenterNameLen - 2] + '_' + \
                                   probCenterName[probCenterNameLen - 1].split('.')[0]
                else:
                    # No geometric
                    probeData = allCamProbiter.next()
                    probImg = probeData['image'].to(device)
                    probName = probeData['filename'][0].split('\\')
                    probNameLen = len(probName)
                    probRealLabel = probName[probNameLen - 2]
                    probSaveName = probName[probNameLen - 3] + '_' + \
                                   probName[probNameLen - 2] + '_' + \
                                   probName[probNameLen - 1].split('.')[0]

                for gallertyI, gallertyData in enumerate(allCamGalleryLoader):
                    finalLabels = []
                    galleryLabels = []
                    gallerySaveNames = []
                    galleryImgs = gallertyData['image'].to(device)
                    galleryNames = gallertyData['filename']

                    for i in range(len(galleryNames)):
                        galleryRealName = galleryNames[i].split('\\')
                        galleryRealNameLen = len(galleryRealName)
                        galleryRealLabel = galleryRealName[galleryRealNameLen - 2]
                        gallerySaveName = galleryRealName[galleryRealNameLen - 3] + '_' + \
                                          galleryRealName[galleryRealNameLen - 2] + '_' + \
                                          galleryRealName[galleryRealNameLen - 1].split('.')[0]
                        if probSaveName == gallerySaveName:
                            tmp = galleryImgs.tolist()
                            tmp.pop(i)
                            galleryImgs = torch.tensor(tmp).to(device)

                            continue
                        galleryLabels.append(int(galleryRealLabel))
                        gallerySaveNames.append(gallerySaveName)
                    galleryLabels = torch.tensor(galleryLabels)

                    for i in range(len(galleryLabels)):
                        if galleryLabels[i] != int(probRealLabel):
                            finalLabels.append(0)
                        else:
                            finalLabels.append(1)
                    finalLabels = torch.tensor(finalLabels).to(device)

                    if geometric == True:
                        catProbImgs = probCenterImg  # geometric
                        for i in range(galleryImgs.shape[0] - 1):
                            catProbImgs = torch.cat([catProbImgs, probCenterImg], dim=0)
                    else:
                        catProbImgs = probImg  # No geometric
                        for i in range(galleryImgs.shape[0] - 1):
                            catProbImgs = torch.cat([catProbImgs, probImg], dim=0)

                    inputImgs1 = imgPre.preprocess(catProbImgs, galleryImgs, model1ImgType, device)
                    if model1ImgType == 'IPVT2':
                        reShape = torch.cat([inputImgs1, inputImgs1, inputImgs1], dim=1)
                        inputImgs1 = reShape

                    inputImgs2 = imgPre.preprocess(catProbImgs, galleryImgs, model2ImgType, device)
                    if model2ImgType == 'IPVT2':
                        reShape = torch.cat([inputImgs2, inputImgs2, inputImgs2], dim=1)
                        inputImgs2 = reShape

                    outputs1 = model1(inputImgs1)
                    outputs1 = m(outputs1)

                    outputs2 = model2(inputImgs2)
                    outputs2 = m(outputs2)

                    outputData1 = torch.zeros(1, 1, 7, 7).to(device)  # output 7x7로 변환
                    for output_i in range(len(outputs1)):
                        tmp_w = []
                        tmp_h = torch.zeros(1, 7).to(device)

                        for output_w in range(7):
                            tmp_w.append(outputs1[output_i][1])
                        tmp_w = torch.tensor(tmp_w).to(device)
                        tmp_w = torch.reshape(tmp_w, (1, 7))
                        for output_h in range(7):
                            tmp_h = torch.cat([tmp_h, tmp_w], dim=0)
                        tmp_h = tmp_h[1:, :]
                        tmp_h = torch.reshape(tmp_h, (1, 1, 7, 7))
                        outputData1 = torch.cat([outputData1, tmp_h], dim=0)

                    outputData2 = torch.zeros(1, 1, 7, 7).to(device)  # output 7x7로 변환
                    for output_i in range(len(outputs2)):
                        tmp_w = []
                        tmp_h = torch.zeros(1, 7).to(device)

                        for output_w in range(7):
                            tmp_w.append(outputs2[output_i][1])
                        tmp_w = torch.tensor(tmp_w).to(device)
                        tmp_w = torch.reshape(tmp_w, (1, 7))
                        for output_h in range(7):
                            tmp_h = torch.cat([tmp_h, tmp_w], dim=0)
                        tmp_h = tmp_h[1:, :]
                        tmp_h = torch.reshape(tmp_h, (1, 1, 7, 7))
                        outputData2 = torch.cat([outputData2, tmp_h], dim=0)

                    outputData1 = outputData1[1:, :]
                    outputData2 = outputData2[1:, :]

                    totalData = torch.cat([outputData1, outputData2, gradient1[0], gradient2[0], gradient_att[0]],
                                          dim=1)
                    gradient1.clear()
                    gradient2.clear()
                    gradient_att.clear()

                    outputs = sceNet(totalData)
                    outputs = m(outputs)
                    loss = criterion(outputs, finalLabels)

                    progress = progress + outputs.shape[0]
                    print("load : %.5f%%" % ((progress * 100) / totalLen))

                    for labelCnt in range(outputs.shape[0]):
                        label = finalLabels[labelCnt].item()

                        ReIDkey = probRealLabel

                        if keyTmp != ReIDkey and keyTmp != 0:
                            value = ReIDdict.get(keyTmp)
                            if value != None:
                                for valueCnt in range(len(ReIDresult)):
                                    value.append(ReIDresult[valueCnt])
                                ReIDdict[keyTmp] = value
                            else:
                                ReIDdict[keyTmp] = ReIDresult

                            ReIDresult = []
                        ReIDresult.append(
                            [[outputs[labelCnt][0].item(), outputs[labelCnt][1].item()], loss.item(), label,
                             '[' + str(probSaveName) + '-' + str(gallerySaveNames[labelCnt]) + ']'])
                        keyTmp = ReIDkey

            value = ReIDdict.get(keyTmp)
            if value != None:
                for valueCnt in range(len(ReIDresult)):
                    value.append(ReIDresult[valueCnt])
                ReIDdict[keyTmp] = value
            else:
                ReIDdict[keyTmp] = ReIDresult

            saveValResultName = modelPath.split('.')[0]
            valResultPath = './' + datasetType + '\\' + 'valResult/' + modelType + '\\' + Fold + '/epochTermValidation/'
            if not os.path.isdir(valResultPath):
                os.makedirs(valResultPath)
            with open(valResultPath + 'valResult' + '_' + saveValResultName + '.pickle',
                      'wb') as fw:
                pickle.dump(ReIDdict, fw)

    del allCamProbValset
    del allCamProbLoader
    del allCamGalleryValset
    del allCamGalleryLoader
    torch.cuda.empty_cache()

