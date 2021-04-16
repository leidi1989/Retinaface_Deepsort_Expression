'''
Description:
Version:
Author: Leidi
Date: 2021-03-06 11:18:41
LastEditors: Leidi
LastEditTime: 2021-04-16 10:06:04
'''
from __future__ import print_function
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import time
import cv2

from Pytorch_Retinaface.data import cfg_mnet, cfg_re50
from Pytorch_Retinaface.layers.functions.prior_box import PriorBox
from Pytorch_Retinaface.utils.nms.py_cpu_nms import py_cpu_nms
from Pytorch_Retinaface.models.retinaface import RetinaFace
from Pytorch_Retinaface.utils.box_utils import decode, decode_landm
from face_expression.Expression import *

from utils.BaseDetector import baseDet
from utils.general import letterbox


cfg_mnet = {
    'name': 'mobilenet0.25',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 32,
    'ngpu': 1,
    'epoch': 250,
    'decay1': 190,
    'decay2': 220,
    'image_size': 640,
    'pretrain': True,
    'return_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3},
    'in_channel': 32,
    'out_channel': 64
}


cfg_re50 = {
    'name': 'Resnet50',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 24,
    'ngpu': 4,
    'epoch': 100,
    'decay1': 70,
    'decay2': 90,
    'image_size': 840,
    'pretrain': True,
    'return_layers': {'layer2': 1, 'layer3': 2, 'layer4': 3},
    'in_channel': 256,
    'out_channel': 256
}


expression_list = ['Angry', 'Disgust', 'Fear',
                   'Happy', 'Sad', 'Surprise', 'Neutral']


class Face_info():
    def __init__(self, bbox_in=None, expression_in=None):
        if len(bbox_in):
            self.id = bbox_in[4]
            self.bbox = bbox_in[0:4]
            # expression confidence score
            self.expression = expression_in


def plot_face_bbox(image_in, face_info_list, line_thickness=None):
    # Plots one bounding box on image img
    image = image_in
    if len(face_info_list):
        for face_info in face_info_list:
            color = (0, 255, 0)
            cls_id = None
            tl = line_thickness or round(
                0.002 * (image.shape[0] + image.shape[1]) / 2) + 1  # line/font thickness
            c1, c2 = (face_info.bbox[0], face_info.bbox[1]
                      ), (face_info.bbox[2], face_info.bbox[3])

            cv2.rectangle(image, c1, c2, color,
                          thickness=tl, lineType=cv2.LINE_AA)
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(
                cls_id, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(image, c1, c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(image, 'ID:{}'.format(face_info.id), (c1[0], c1[1] - 2), 0, tl / 3,
                        [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

            str_off_set = 0
            if face_info.expression != None:
                for one in zip(expression_list, face_info.expression):
                    str_line = ''
                    str_line += one[0] + ':' + str(one[1])
                    cv2.putText(image, '{}'.format(str_line), (face_info.bbox[0], face_info.bbox[3] + 15 + str_off_set), 0, tl / 6,
                                [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
                    str_off_set += 20

    return image


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    def f(x): return x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(
            pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(
            pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(
            pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


class Face_Detect(baseDet):

    def __init__(self, weight):
        super(Face_Detect, self).__init__()
        self.init_model(weight)
        self.build_config()

    def init_model(self, weight):

        # Face_Detect 权重输入
        # self.weights = 'weights/Resnet50_Final.pth'
        self.model = RetinaFace(cfg=cfg_re50, phase='test')
        self.model = load_model(self.model, weight, load_to_cpu=None)
        print('Finished loading model!')
        print(self.model)
        cudnn.benchmark = True
        device = torch.device("cuda")
        self.model = self.model.to(device)

    def preprocess(self, img):
        
        img = letterbox(img, new_shape=self.img_size)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(torch.device("cuda"))
        img = img.half()  # 半精度
        img /= 255.0  # 图像归一化
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        return img

    def detect(self, img):
        """[人脸检测]

        Args:
            img ([tensor]): [输入图片]

        Returns:
            img ([tensor]): [输出图片]
            dets ([list]): [人脸检测框及置信度]
        """        
        # 自定义参数
        # TODO
        # img = cv2.resize(img, (640, 480))
        device = torch.device('cuda')
        cfg = cfg_re50
        resize = 1
        confidence_threshold = 0.8
        top_k = 5000
        nms_threshold = 0.8
        keep_top_k = 750

        start_total = time.time()
        
        img = np.float32(img)
        im_height, im_width, _ = img.shape
        scale = torch.Tensor(
            [img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(torch.device("cuda"))
        scale = scale.to(torch.device("cuda"))

        pre_process_time = time.time()

        print('Face detect pre process time: {:.4f}'.format(
            pre_process_time - start_total))

        tic = time.time()

        loc, conf, landms = self.model(img)  # forward pass

        bef_process_time_start = time.time()

        print('Face detect net forward time: {:.4f}'.format(time.time() - tic))

        priorbox = PriorBox(cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0),
                              prior_data, cfg['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2]])
        scale1 = scale1.to(device)
        landms = landms * scale1 / resize
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(
            np.float32, copy=False)
        keep = py_cpu_nms(dets, nms_threshold)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        for one in dets:
            print('confidence:', one[4])

        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[:keep_top_k, :]
        # landms = landms[:keep_top_k, :]

        # dets = np.concatenate((dets, landms), axis=1)
        bef_process_time_end = time.time()
        print('Face detect bef process time: {:.4f}'.format(
            bef_process_time_end - bef_process_time_start))
        print('Face detect Total time: {:.4f}'.format(bef_process_time_end - start_total))

        return img, dets


class Face_detect_experssion():
    """[人脸,表情检测]
    """
    def __init__(self, face_detect_weight_path, face_expression_weight_path) -> None:
        """[Face_detect_experssion, 初始化函数]

        Args:
            face_detect_weight_path ([str]): [face_detect_weight_path]
            face_expression_weight_path ([str]): [face_expression_weight_path]
        """
        self.face_det = Face_Detect(face_detect_weight_path)
        self.exprs_det = Face_Expression(face_expression_weight_path)

    def detect_expression(self, im):
        """[表情检测器]

        Args:
            im ([type]): [输入图片]

        Returns:
            result_boxes ([list]): [人脸检测框,左上点和右下点坐标及ID]
            Face_info_list ([list]): [表情置信度列表]
        """
        print('*'*50)
        result_boxes = {}
        Face_info_list = []
        # TODO
        # im = cv2.resize(im, (640, 480))
        if im is None:
            return result_boxes, Face_info_list
        
        # 检测并跟踪,获取检测框位置及ID
        result = self.face_det.feedCap(im)
        result_boxes = result['boxes']

        # 对检测出的检测框进行表情识别
        if 0 != len(result_boxes):
            for one_face_img in result_boxes:
                face_expression = self.exprs_det.detect(im[one_face_img[1]:one_face_img[3],
                                                           one_face_img[0]:one_face_img[2]])
                Face_info_list.append(Face_info(one_face_img, face_expression))

        for one_face in Face_info_list:
            print('ID: {},\nexpression list: {},\nboxes: {}'.format(
                one_face.id, one_face.expression, one_face.bbox[0:4]))
        print('*'*50)

        return result_boxes, Face_info_list


def test(im, face_detect_weight_path, face_expression_weight_path):

    det = Face_detect_experssion(
        face_detect_weight_path, face_expression_weight_path)

    while True:
        if 0 == len(im):
            break
        im = cv2.resize(im, (640, 480))
        result_boxes, Face_info_list = det.detect_expression(im)
        # 图像输出
        # im = cv2.resize(im, (640, 480))
        # result_image = plot_face_bbox(im, Face_info_list)
        # cv2.imshow(name, result_image)
        # cv2.waitKey(1)

        # if cv2.getWindowProperty(name, cv2.WND_PROP_AUTOSIZE) < 1:
        #     # 点x退出
        #     return
        # 无图像输出
        print(result_boxes)
        print(Face_info_list)


if __name__ == '__main__':

    # Face_Detect_backbone_weights = r'weights/Resnet50_Final.pth'
    face_detect_weight_path = r'weights/Resnet50_Final.pth'
    face_expression_weight_path = r'face_expression/weight/PrivateTest_model.t7'
    cap = cv2.VideoCapture(
        r'WIN_20210308_15_32_55_Pro.mp4')
    # cap = cv2.VideoCapture(0)
    name = 'demo'

    _, im = cap.read()
    
    time_star = time.time()
    
    test(im, face_detect_weight_path, face_expression_weight_path)
    
    print('Total time: {:.4f}'.format(time.time() - time_star))
    
    cap.release()
    cv2.destroyAllWindows()
