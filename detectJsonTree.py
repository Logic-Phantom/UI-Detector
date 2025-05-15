import torch
from ultralytics import YOLO
import json
import uuid
from datetime import datetime, timezone
import os

class DetectionJsonBuilder:
    def __init__(self):
        # YOLO 모델 초기화
        self.model = YOLO("runs/detect/train4/weights/best.pt")  # 모델 로딩
        
    def generate_id(self):
        # Figma 스타일의 ID 생성 (예: "3:47")
        return f"{uuid.uuid4().int % 100}:{uuid.uuid4().int % 1000}"
    
    def create_base_structure(self):
        # 기본 JSON 구조 생성
        return {
            "document": {
                "id": "0:0",
                "name": "Document",
                "type": "DOCUMENT",
                "scrollBehavior": "SCROLLS",
                "children": []
            },
            "components": {},
            "componentSets": {},
            "schemaVersion": 0,
            "styles": {},
            "name": "Detection Results",
            "lastModified": datetime.now(timezone.utc).isoformat(),
            "version": str(uuid.uuid4().int),
            "role": "owner",
            "editorType": "detection"
        }
    
    def create_detection_group(self, detection, class_names, image_width, image_height):
        # 탐지된 객체에 대한 그룹 생성
        x1, y1, x2, y2 = detection.xyxy[0].tolist()
        conf = detection.conf[0]
        class_id = int(detection.cls[0])
        class_name = class_names[class_id]
        
        group_id = self.generate_id()
        
        return {
            "id": group_id,
            "name": f"Detection_{class_name}",
            "type": "GROUP",
            "scrollBehavior": "SCROLLS",
            "children": [
                {
                    "id": self.generate_id(),
                    "name": f"Rectangle_{class_name}",
                    "type": "RECTANGLE",
                    "scrollBehavior": "SCROLLS",
                    "blendMode": "PASS_THROUGH",
                    "absoluteBoundingBox": {
                        "x": x1,
                        "y": y1,
                        "width": x2 - x1,
                        "height": y2 - y1
                    },
                    "constraints": {
                        "vertical": "TOP",
                        "horizontal": "LEFT"
                    }
                },
                {
                    "id": self.generate_id(),
                    "name": f"{class_name}_{conf:.2f}",
                    "type": "TEXT",
                    "scrollBehavior": "SCROLLS",
                    "characters": f"{class_name} ({conf:.2f})",
                    "style": {
                        "fontFamily": "Arial",
                        "fontSize": 14,
                        "fontWeight": 600
                    }
                }
            ]
        }

    def detect_and_create_json(self, image_path):
        # 이미지에서 객체 탐지 수행
        results = self.model(image_path)
        
        # 기본 JSON 구조 생성
        json_data = self.create_base_structure()
        
        # Canvas 생성
        canvas = {
            "id": "0:1",
            "name": "Detection Canvas",
            "type": "CANVAS",
            "scrollBehavior": "SCROLLS",
            "children": []
        }
        
        # 각 탐지 결과에 대한 그룹 생성
        for result in results:
            image_width, image_height = result.orig_shape[1], result.orig_shape[0]
            class_names = result.names  # 클래스 이름 딕셔너리 가져오기
            
            for detection in result.boxes:
                detection_group = self.create_detection_group(detection, class_names, image_width, image_height)
                canvas["children"].append(detection_group)
        
        json_data["document"]["children"] = [canvas]
        return json_data

def main():
    detector = DetectionJsonBuilder()
    image_path = "./screenshots/test.png"  # 실제 이미지 경로
    json_result = detector.detect_and_create_json(image_path)
    
    # JSON 결과 저장
    with open("detection_results.json", "w", encoding="utf-8") as f:
        json.dump(json_result, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main() 