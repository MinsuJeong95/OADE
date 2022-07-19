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

import pickle
import imgPreprocess


def accuracy(out, yb):
    softmax = nn.Softmax(dim=1)
    out = softmax(out)
    compare = []
    for i in range(len(out)):
        if out[i][1] >= out[i][0]:
            compare.append(1)
        else:
            compare.append(0)

    yb = yb.cpu()
    compare = torch.Tensor(compare).long()

    return (compare == yb).float().mean()


def training(datasetType, modelType, Fold, imgType, model, DBPath, numEpoch=1, startEpoch=0, lr=1e-3, wd=1e-4):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(torch.__version__)
    print(device)

    batchSize = 16
    learningRate = lr
    wdecay = wd

    imgPre = imgPreprocess.imgPreprocess()

    print('Fold : ' + Fold)
    print('Type : ' + imgType)

    model = model.densenet161(pretrained=True)
    model.classifier = nn.Linear(2208, 2)

    pathLen = len(DBPath.split('\\'))

    trans = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    Trainset = torchvision.datasets.ImageFolder(root=DBPath, transform=trans)
    Loader = DataLoader(Trainset, batch_size=batchSize, shuffle=True, pin_memory=True)

    model.train()  # 학습모드
    model.to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learningRate, weight_decay=wdecay)

    writer = SummaryWriter()  # tensorboard
    iterCnt = 0
    fileName = 0
    lastEpoch = 0
    saveIterCnt = []
    saveLoss = []
    saveAccuracy = []

    for epoch in range(numEpoch - startEpoch):  # 데이터셋을 수차례 반복합니다.
        for i, Data in enumerate(Loader):
            inputImgs = Data[0].to(device)
            labels = Data[1].to(device)

            # 변화도(Gradient) 매개변수를 0으로 만들고
            optimizer.zero_grad()
            # 순전파 + 역전파 + 최적화를 한 후
            outputs = model(inputImgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 통계 출력
            acc = accuracy(outputs, labels)
            print('[%d, %5d] loss: %.6f' %
                  (epoch + 1, i + 1, loss.item()))
            writer.add_scalar('iter/loss', loss.item(), iterCnt)
            writer.add_scalar('iter/accuracy', acc, iterCnt)
            saveIterCnt.append(iterCnt)
            saveLoss.append(loss.item())
            saveAccuracy.append(acc)

            iterCnt = iterCnt + 1

        fileName = DBPath.split('\\')[pathLen-4] + '_' + DBPath.split('\\')[pathLen-3] + '_' + \
                   DBPath.split('\\')[pathLen-1]
        PATH = './' + datasetType + '/trainModels/' + modelType + '/' + \
               Fold + '/' + imgType + '/epochTermModel/'
        if not os.path.isdir(PATH):
            os.makedirs(PATH)
        trainPath = PATH + fileName + '_ReID_' + modelType + '_' + \
               str(epoch + startEpoch + 1) + '.pth'

        print(trainPath)
        torch.save(model.state_dict(), trainPath)

        lastEpoch = epoch + 1
        if lastEpoch % 3 == 0:
            learningRate = learningRate * 0.1
            optimizer = optim.Adam(model.parameters(), lr=learningRate, weight_decay=wdecay)

        trainInfoPath = './' + datasetType + '/trainModels/' + modelType + '/' + Fold + '/' + imgType \
                        + '/saveEpochInfo/'
        if not os.path.isdir(trainInfoPath):
            os.makedirs(trainInfoPath)

        with open(trainInfoPath + fileName + '_ReID_' + modelType + '_saveIterCnt_' + str(
                startEpoch + 1) + '-' + str(startEpoch + lastEpoch) + '.pickle', 'wb') as fw:
            pickle.dump(saveIterCnt, fw)
        with open(trainInfoPath + fileName + '_ReID_' + modelType + '_saveLoss_' + str(
                startEpoch + 1) + '-' + str(startEpoch + lastEpoch) + '.pickle', 'wb') as fw:
            pickle.dump(saveLoss, fw)
        with open(trainInfoPath + fileName + '_ReID_' + modelType + '_saveAccuracy_' + str(
                startEpoch + 1) + '-' + str(startEpoch + lastEpoch) + '.pickle', 'wb') as fw:
            pickle.dump(saveAccuracy, fw)

    del Trainset
    del Loader
    torch.cuda.empty_cache()
    writer.close()

    print('Finished Training')


