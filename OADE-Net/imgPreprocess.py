import cv2
import torch
import numpy as np
import random
from PIL import Image

import torch.nn.functional as F

class imgPreprocess:

    def preprocess(self, imgs1, imgs2, type, device):
        resizeImgs1 = F.interpolate(imgs1, size=(224, 112), mode='bicubic', align_corners=False).to(device)
        resizeImgs2 = F.interpolate(imgs2, size=(224, 112), mode='bicubic', align_corners=False).to(device)

        if type == 'IPVT1':
            zeroImgs = torch.zeros(imgs1.shape[0], imgs1.shape[1], imgs1.shape[2],
                                   resizeImgs1.shape[3] + resizeImgs2.shape[3]).to(device)
            outputImg = torch.cat([imgs1, imgs2, zeroImgs], dim=1)
        elif type == 'IPVT2':
            outputImg = torch.cat([resizeImgs1, resizeImgs2], dim=3)
        else:
            IPVT1 = torch.cat([imgs1, imgs2], dim=1)
            IPVT2 = torch.cat([resizeImgs1, resizeImgs2], dim=3)
            outputImg = torch.cat([IPVT1, IPVT2], dim=1)

        return outputImg

    def viewTensorImg(self, outputImgs):
        for i in range(outputImgs.shape[0]):
            viewImg = outputImgs[i].cpu()
            viewImg = viewImg.permute(1, 2, 0)
            viewImg = viewImg.numpy()

            cv2.imshow('view', viewImg)
            cv2.waitKey(0)

    def saveImg(self, img, save_image_path, registImageName, identifyImageName, ReIDLabels, imgCnt):
        img = img.squeeze()
        if img.shape[0] == 3:
            b = img[0]
            g = img[1]
            r = img[2]
            img = torch.cat([r.unsqueeze(0), g.unsqueeze(0), b.unsqueeze(0)], dim=0)
            img = img.permute(1, 2, 0)

        img = img.numpy()

        img *= 255.0
        cv2.imwrite(
            save_image_path + '[' + registImageName + '-' + identifyImageName + ']' + str(ReIDLabels) + '_' + str(imgCnt) + '.png', img)


