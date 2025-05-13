'''
Created on 2025. 5. 13.

@author: LCM
'''
# yolo/detector.py
import torch
import json
import os

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True)

def detect(image_path: str, save_json_path: str = "output/detection.json"):
    results = model(image_path)

    # 결과를 Pandas DataFrame으로 추출
    df = results.pandas().xyxy[0]

    # 필요한 필드만 추출
    objects = []
    for _, row in df.iterrows():
        obj = {
            "class": row['name'],
            "confidence": float(row['confidence']),
            "x1": float(row['xmin']),
            "y1": float(row['ymin']),
            "x2": float(row['xmax']),
            "y2": float(row['ymax'])
        }
        objects.append(obj)

    # 디렉토리 없으면 생성
    os.makedirs(os.path.dirname(save_json_path), exist_ok=True)

    # JSON으로 저장
    with open(save_json_path, 'w', encoding='utf-8') as f:
        json.dump(objects, f, indent=2)

    print(f"Detection result saved to: {save_json_path}")
    return objects
