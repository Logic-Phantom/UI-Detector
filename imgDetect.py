'''
Created on 2025. 5. 14.

@author: LCM
'''
from ultralytics import YOLO  # YOLO 모델을 가져옵니다.
import cv2  # OpenCV를 사용하여 이미지를 읽고 처리합니다.
import numpy as np  # Numpy를 사용하여 배열 처리합니다.
import pandas as pd  # pandas를 사용하여 DataFrame 처리합니다.

# YOLO 모델 로드
model = YOLO("runs/detect/train/weights/best.pt")  # 모델 로딩

# 이미지 로드
image_path = './screenshots/test.png'
img = cv2.imread(image_path)

# 객체 감지
results = model(img)

# 결과 시각화
results[0].show()  # 첫 번째 결과의 시각화를 표시

# 결과를 DataFrame으로 변환
detected_elements_df = results[0].pandas().xywh[0]  # pandas DataFrame 형식으로 변환
print(detected_elements_df)  # 감지된 요소 출력