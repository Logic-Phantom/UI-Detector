'''
Created on 2025. 5. 14.

@author: LCM
'''
from ultralytics import YOLO
model = YOLO("yolov8n.yaml")  # 혹은 yolov8n.pt
model.train(data="data.yaml", epochs=100)