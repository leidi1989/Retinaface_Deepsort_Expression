'''
Description: 
Version: 
Author: Leidi
Date: 2021-03-11 15:35:10
LastEditors: Leidi
LastEditTime: 2021-03-11 17:17:26
'''
from torchvision import datasets, models, transforms
import torchvision
import torch
import torch.nn as nn
import numpy as np
import cv2


class FairFace():
    def __init__(self, model_weight):
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

        self.model_fair = torchvision.models.resnet34(pretrained=False)
        self.model_fair.fc = nn.Linear(self.model_fair.fc.in_features, 18)
        self.model_fair.load_state_dict(torch.load(model_weight))
        self.model_fair = self.model_fair.to(self.device)
        self.model_fair.eval()

        self.trans = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225])
        ])

    def detect(self, image):
        race = ["White", "Black", "Latino_Hispanic", "East_Asian",
                "Southeast_Asian", "Indian", "Middle_Eastern"]
        gender = ["Male", "Female"]
        age = ['0-2', '3-9', '10-19', '20-29',
               '30-39', '40-49', '50-59', '60-69', '70+']
        image = self.trans(image)
        # reshape image to match model dimensions (1 batch size)
        image = image.view(1, 3, 224, 224)
        image = image.to(self.device)
        outputs = self.model_fair(image)
        outputs = outputs.cpu().detach().numpy()
        outputs = np.squeeze(outputs)

        race_outputs = outputs[:7]
        gender_outputs = outputs[7:9]
        age_outputs = outputs[9:18]

        race_score = np.exp(race_outputs) / np.sum(np.exp(race_outputs))
        gender_score = np.exp(gender_outputs) / np.sum(np.exp(gender_outputs))
        age_score = np.exp(age_outputs) / np.sum(np.exp(age_outputs))

        race_pred = np.argmax(race_score)
        gender_pred = np.argmax(gender_score)
        age_pred = np.argmax(age_score)
        result = {"race": race[race_pred],
                  "gender": gender[gender_pred], "age": age[age_pred]}
        print("result", result)
        return result


# face = FairFace("/home/py/code/github/FairFace/fair_face_models/res34_fair_align_multi_7_20190809.pt")
# image = cv2.imread("/home/py/code/pycharm/pyqt/Ceramic_detection/face_image/3.jpg")
# face.detect(image)
def main():

    det = FairFace("face_expression/weight/res34_fair_align_multi_7.pt")
    cap = cv2.VideoCapture(
        r'/home/user/workspace/detect_sample/Camera Roll/WIN_20210308_15_30_38_Pro.mp4')
    # cap = cv2.VideoCapture(0)

    while True:

        _, im = cap.read()
        if im is None:
            break
        im = cv2.resize(im, (640, 480))
        result = det.detect(im)

        cv2.imshow('face expression', im)
        cv2.waitKey(1)
        if cv2.getWindowProperty('face expression', cv2.WND_PROP_AUTOSIZE) < 1:
            # 点x退出
            break
        # except Exception as e:
        #     print(e)
        #     break

    cap.release()
    cv2.destroyAllWindows()
    # img_s, det_face = det.detect()

    return


if __name__ == '__main__':

    main()
