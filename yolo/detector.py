'''
Created on 2025. 5. 13.

@author: LCM
'''
import torch
import json
import cv2
from pathlib import Path

# YOLO 모델 로드
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

def detect_ui_elements(image_path):
    # 이미지 로드
    img = cv2.imread(image_path)
    results = model(img)
    
    # 결과를 JSON 형식으로 변환
    detected_elements = []
    for idx, det in enumerate(results.xywh[0]):
        # 감지된 요소의 클래스, 좌표, 크기
        class_id = int(det[5].item())  # 클래스 ID
        x, y, w, h = det[:4].tolist()  # 좌표 및 크기
        label = results.names[class_id]  # 클래스명 (예: 'button', 'textbox')

        # 감지된 요소 정보
        detected_elements.append({
            "type": label,
            "id": f"{label}-{idx}",
            "position": {
                "top": f"{int(y)}px",
                "left": f"{int(x)}px",
                "width": f"{int(w)}px",
                "height": f"{int(h)}px"
            },
            "children": []
        })
    
    # 계층형 JSON 구조로 포장
    ui_json = {
        "name": Path(image_path).stem,  # 이미지 이름을 UI 이름으로 사용
        "elements": detected_elements
    }

    return ui_json

# 이미지에서 UI 요소 감지 후 JSON 생성
image_path = '../img/test.png'
ui_json = detect_ui_elements(image_path)

# 결과 출력
print(json.dumps(ui_json, indent=2))
