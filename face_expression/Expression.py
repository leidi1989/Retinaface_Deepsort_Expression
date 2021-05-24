'''
Description: 
Version: 
Author: Leidi
Date: 2021-03-11 15:34:15
LastEditors: Leidi
LastEditTime: 2021-05-24 10:23:20
'''
import numpy as np
from PIL import Image
from torchvision import transforms
from skimage.transform import resize
import time
from face_expression.models.vgg import *


class Face_Expression():
    def __init__(self, model_weight):
        self.face_expressionn_model = VGG('VGG19')
        checkpoint = torch.load(model_weight)
        self.face_expressionn_model.load_state_dict(checkpoint['net'])

        self.face_expressionn_model.cuda()
        self.face_expressionn_model.eval()
        cut_size = 44
        self.transform_test = transforms.Compose([
            transforms.TenCrop(cut_size),
            transforms.Lambda(lambda crops: torch.stack(
                [transforms.ToTensor()(crop) for crop in crops])),
        ])

    def rgb2gray(self, rgb):
        return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

    def bgr2gray(self, rgb):
        return np.dot(rgb[..., :3], [0.114, 0.587, 0.299])

    def detect(self, image):

        # if len(image):
        start_total = time.time()

        gray = self.bgr2gray(image)
        gray = resize(gray, (48, 48), mode='symmetric').astype(np.uint8)
        img = gray[:, :, np.newaxis]
        img = np.concatenate((img, img, img), axis=2)
        img = Image.fromarray(img)
        inputs = self.transform_test(img)
        ncrops, c, h, w = np.shape(inputs)
        inputs = inputs.view(-1, c, h, w)
        inputs = inputs.cuda()
        with torch.no_grad():
            # inputs = Variable(inputs, volatile=True)

            pre_process_time = time.time()
            print('Expression pre process time: {:.4f}'.format(
                pre_process_time - start_total))

            tic = time.time()

            outputs = self.face_expressionn_model(inputs)

            print('Expression net forward time: {:.4f}'.format(
                time.time() - tic))

            bef_process_time_start = time.time()

            outputs_avg = outputs.view(ncrops, -1).mean(0)  # avg over crops
            score = F.softmax(outputs_avg, dim=-1)

            bef_process_time_end = time.time()

            print('Expression bef process time: {:.4f}'.format(
                bef_process_time_end - bef_process_time_start))

            # print(score)
        return score.cpu().detach().numpy().tolist()
