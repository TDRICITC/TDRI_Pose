# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 09:12:14 2021

@author: GWJIANG
"""
import base64
import cv2


with open(r"C:\Users\GWJIANG\Desktop\GW\python code\yolov4\DEMO\4.JPG", "rb") as img_file:
    b64_string = base64.b64encode(img_file.read())
print(b64_string)



lmList = []

from PIL import Image
import io
import cv2
import numpy as np
byteImgIO = io.BytesIO()
byteImg = Image.open(r"C:\Users\GWJIANG\Desktop\GW\python code\yolov4\DEMO\4.JPG")
byteImg.save(byteImgIO, "PNG")
byteImgIO.seek(0)
byteImg = byteImgIO.read()

img_bytes = byteImg

#img_bytes = request.files['image'].read() 
img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), -1)
detector_img = detector.findHands(img)
detector_img = detector.findHands(img)
lmList = detector.findPosition(detector_img, draw=False)
print (lmList)
