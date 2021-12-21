# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 15:21:57 2021

@author: GWJIANG
"""

import logging
import time
from flask_app import app, api_bp
from .utils import *
import requests
import json
import numpy
import cv2
import os
import sys
import numpy as np
import base64
from flask import current_app, Blueprint, jsonify, request
import io
import PIL.Image as Image
from pathlib import Path
import torch

###############################################

logger = logging.getLogger(__name__)


os.chdir("/opt/flask_app/yolov5")
sys.path.append("/opt/flask_app/yolov5")

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, colorstr, non_max_suppression, \
    apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_sync


def RGB_IMAGE(img_bytes):
    image = Image.open(io.BytesIO(img_bytes))
    image = image.convert("RGB")
    filename = 'temp.jpg'  
    image.save(filename)
    image = open(filename, 'rb') #open binary file in read mode
    image_read = image.read()
    image_64_encode = base64.encodestring(image_read)
    img_bytes = base64.b64decode(image_64_encode)
    return img_bytes

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

class Yolov5Net():
    def __init__(self, model_weight_path, img_size=640, model_conf=0.7, iou_thres=0.45, max_det=1000):
        self.model_weight_path = model_weight_path
        self.img_size = [img_size,img_size]
        self.model_conf = model_conf
        self.iou_thres = iou_thres
        self.max_det = max_det
        # initialize
        self.device = select_device('')
        # load model
        classify, suffix = False, Path(self.model_weight_path).suffix.lower()
        stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults
        print('loading model ...')
        self.model = attempt_load(self.model_weight_path, map_location=self.device)  # load FP32 model
        stride = int(self.model.stride.max())  # model stride
        self.class_names = self.model.module.names if hasattr(self.model, 'module') else self.model.names  # get class names
        self.img_size = check_img_size(self.img_size, s=stride)
        print('self.class_names : ', self.class_names)
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, *self.img_size).to(self.device).type_as(next(self.model.parameters())))  # run once
    
    def detect(self, img):
        try:
            res = letterbox(img, new_shape=self.img_size)[0]
            # Convert
            res = res.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            res = np.ascontiguousarray(res)
            res_tensor = torch.from_numpy(res).to(self.device)
            res_tensor = res_tensor / 255.0  # 0 - 255 to 0.0 - 1.0
            if len(res_tensor.shape) == 3:
                res_tensor = res_tensor[None]  # expand for batch dim
            # predict
            pred = self.model(res_tensor, augment=False)[0]
            # NMS
            pred = non_max_suppression(pred, self.model_conf, self.iou_thres, classes=None, agnostic=False)
            # get bbox
            output_dict = {'center':[], 'people_start':[], 'people_end':[], 'label':[], 'confidence':[]}
            result_dict = {}
            return_object  = {}
            result = []
            
            for det in pred:
                if len(det):
                    # print('det shape : \n', det.shape)
                    t = time.time()
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(res_tensor.shape[2:], det[:, :4], img.shape).round()
                    for *xyxy, conf, classID in reversed(det):
                        x1 = int(xyxy[0])
                        y1 = int(xyxy[1])
                        x2 = int(xyxy[2])
                        y2 = int(xyxy[3])
                        xc = (x1+x2)/2
                        yc = (y1+y2)/2
                        cn = self.class_names[int(classID)]
                        # cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        # cv2.putText(img, cn, (x1, y1+10), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
                        # print('x1=',x1,',y1=',y1,',x2=',x2,',y2=',y2,',cn=',cn,',conf=',conf.item())
                        output_dict['center'].append([xc,yc])
                        output_dict['people_start'].append([x1,y1])
                        output_dict['people_end'].append([x2,y2])
                        output_dict['label'].append(cn)
                        output_dict['confidence'].append(conf.item())
            print(output_dict)
            if len(output_dict['label']) > 1 :
                index = output_dict['people_start'][:][1].index(min(output_dict['people_start'][:][1]))
            else:
                index = 0
            print(0)
            print(index)
            
            return_object['label'] = int(output_dict['label'][index])
            return_object['pose_rectangle'] =  {
                "top": int(output_dict['people_start'][index][1]),
                "left": int(output_dict['people_end'][index][0]),
                "width": int(output_dict['people_end'][index][0] - output_dict['people_start'][index][0]),
                "height": int(output_dict['people_end'][index][1] - output_dict['people_start'][index][1]) 
                }
            print(1)
            print(return_object)
            
            result.append(return_object)
            result_dict['pose_list'] = result
            result_dict['status'] = 0
            print(2)
            print(result_dict)
            #return jsonify(result_dict)
            return result_dict
        
        except Exception as err:
            #logger.error("fatal error in %s", err, exc_info=True)
            status = {"Fatal": str("No hand detected_")+str(err)}
            status['pose_list']= []
            status['status']= "100" 
            #return jsonify(status)
            return status  
        
        
print("load function.")
weight_file = "/opt/flask_app/yolov5/weights/gesture0123_yolov5s_img640_batch16_epoch100.pt"
yolov5net = Yolov5Net(weight_file)    
print(yolov5net)


@api_bp.route('/PredictPose', methods=['GET', 'POST'])
def Pose():
    try:

        
        img_bytes = request.files['image'].read() 
        
        img_bytes = RGB_IMAGE(img_bytes)
        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), -1)
        
        output = yolov5net.detect(img)
        return jsonify(output)
    

    except Exception as err:
        logger.error("Fatal error in %s", err, exc_info=True)
        status = {"Fatal": str(err)}
        status['pose']= []
        status['status']= "100" 

        return jsonify(status)
    
    
    
    
    
@api_bp.route('/test', methods=['GET', 'POST'])
def test():
    output = {"msg": "I'm the test endpoint from blueprint_x."}
    return jsonify(output)
