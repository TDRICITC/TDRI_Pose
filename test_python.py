# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 16:38:29 2021

@author: GWJIANG
"""
import threading
import tkinter as tk
from tkinter import *
import cv2
from PIL import Image,ImageTk
from random import choice
import cv2
import time
import json                                                                              
import requests
from PIL import Image
from io import BytesIO 
from cvzone.HandTrackingModule import HandDetector
import numpy as np    
import os
import mediapipe as mp 
import pandas as pd
###########input############

file_pathname = r"C:\Users\GWJIANG\20211125\acc\2021-12-31\camera01\1" #影像資料路徑
answer = pd.read_csv(r"C:\Users\GWJIANG\20211125\acc\1231.csv")        #答案csv路徑

############################
for filename in os.listdir(file_pathname):

    #filename = str(filename)+".jpg"
    
    return_object = []   
    res_object1 = []
    res_object2  = []
    result = 'null'
    image_result = 'null'
    yolo_result = 'null'
    
    try: 
        im_result = answer.loc[answer['request_id'].isin([filename.replace(".jpg","")])]
        print(im_result)
        image_result = im_result.iloc[0]['label']
    except Exception:
        print("no answer")
    
    img = cv2.imread(file_pathname+'/'+filename)

    try:
        cv2.putText(img, "api_answer = "+str(image_result), (5,150), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    except Exception as err:
        cv2.putText(img, "api_answer = null", (5,50 ), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    
    
    cv2.imshow("show",img)
    cv2.waitKey(10)
cv2.destroyAllWindows()   
                    
