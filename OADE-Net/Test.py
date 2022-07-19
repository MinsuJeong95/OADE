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

import pickle
import imgPreprocess
import CustomDataset

gradient = []
def save_gradient(*args):
    # print("Gradient saved!!!!")
    grad_input = args[1]
    grad_output = args[2]
    # print(grad_output)
    # self.gradient.append(grad_output[0]) #backward
    gradient.append(grad_output)  # forward
    # print(self.gradient[0].size())

def distenseCal(camiter, model, device):
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
    for i in range(len(camName) - 1):
        path = path + str(camName[i]) + '\\'
    imageList = os.listdir(path)
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
            distence = (feature[center] - feature[f_i]) * (feature[center] - feature[f_i]) #L2 distance
            for d_i in range(len(distence)):
                result = result + distence[d_i]
        result = result ** (1 / 2)
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
    return resultImg, resultName

def test(datasetType, modelType, Fold, imgType, model, DBPath, geometric=False):
    batchSize = 300

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(torch.__version__)
    print(device)

    model = model.densenet161(pretrained=True)
    model.classifier = nn.Linear(2208, 2)

    trans = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Grayscale()])

    # allCamProb : 인식 / allCamGallery : 등록
    allCamProbTestset = CustomDataset.CustomDataset(root_dir=DBPath,
                                              transforms=trans)
    allCamProbLoader = DataLoader(allCamProbTestset, batch_size=1, shuffle=False, pin_memory=True)
    allCamGalleryTestset = CustomDataset.CustomDataset(root_dir=DBPath,
                                              transforms=trans)
    allCamGalleryLoader = DataLoader(allCamGalleryTestset, batch_size=batchSize, shuffle=False, pin_memory=True)

    imgPre = imgPreprocess.imgPreprocess()

    # Class 개수 계산
    if geometric == True:
        allCamProbClassLen = len(allCamProbTestset.classes)  # geometric
    else:
        allCamProbClassLen = len(allCamProbTestset)  # No geometric
    totalLen = allCamProbClassLen*len(allCamGalleryTestset)

    print('Fold : ' + Fold)
    print('Type : ' + imgType)
    path = '.\\' + datasetType + '\\' + 'valResult\\' + modelType + '\\' + Fold + '\\' + imgType + '\\' + 'selectEpoch'
    modelPath = os.listdir(path)

    ReIDdict = {}
    progress = 0

    model.load_state_dict(torch.load(path + '\\' + modelPath[0]))
    model.eval()  # 평가모드
    model.to(device)

    with torch.no_grad():
        ReIDresult = []
        keyTmp = 0

        # allCamProb : 인식 / allCamGallery : 등록
        allCamProbiter = iter(allCamProbLoader)
        for probI in range(allCamProbClassLen):

            if geometric == True:
                # geometric
                probCenterImg, probCenterName = distenseCal(allCamProbiter, model, device)
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
                probNameeLen = len(probName)
                probRealLabel = probName[probNameeLen - 2]
                probSaveName = probName[probNameeLen - 3] + '_' + \
                               probName[probNameeLen - 2] + '_' + \
                               probName[probNameeLen - 1].split('.')[0]

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
                finalLabels = torch.tensor(finalLabels)

                if geometric == True:
                    catProbImgs = probCenterImg  # geometric
                    for i in range(galleryImgs.shape[0] - 1):
                        catProbImgs = torch.cat([catProbImgs, probCenterImg], dim=0)
                else:
                    catProbImgs = probImg  # No geometric
                    for i in range(galleryImgs.shape[0] - 1):
                        catProbImgs = torch.cat([catProbImgs, probImg], dim=0)

                inputImgs = imgPre.preprocess(catProbImgs, galleryImgs, imgType, device)

                if imgType == 'IPVT2':
                    reShape = torch.cat([inputImgs, inputImgs, inputImgs], dim=1)
                    inputImgs = reShape

                outputs = model(inputImgs)

                progress = progress + outputs.shape[0]
                print("load : %.5f%%" % ( (progress * 100) / totalLen ))

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
                    ReIDresult.append([[outputs[labelCnt][0].item(), outputs[labelCnt][1].item()], label,
                                         '['+str(probSaveName)+'-'+str(gallerySaveNames[labelCnt])+']'])
                    keyTmp = ReIDkey

        value = ReIDdict.get(keyTmp)
        if value != None:
            for valueCnt in range(len(ReIDresult)):
                value.append(ReIDresult[valueCnt])
            ReIDdict[keyTmp] = value
        else:
            ReIDdict[keyTmp] = ReIDresult

    saveTestResultName = modelPath[0].split('.')[0]
    testPath = './' + datasetType + '\\' + 'testResult/' + modelType + '\\' + Fold + '\\' + imgType + '/epochTermTest/'
    if not os.path.isdir(testPath):
        os.makedirs(testPath)
    with open(testPath + 'ReID_test_result_' + saveTestResultName + '.pickle',
              'wb') as fw:
        pickle.dump(ReIDdict, fw)

    del allCamProbTestset
    del allCamProbLoader
    del allCamGalleryTestset
    del allCamGalleryLoader
    torch.cuda.empty_cache()