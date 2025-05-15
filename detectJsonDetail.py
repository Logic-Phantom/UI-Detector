import torch
from ultralytics import YOLO
import json
import uuid
from datetime import datetime, timezone
import os
from collections import defaultdict
import cv2
import numpy as np
from PIL import Image, ImageStat

class DetectionJsonBuilder:
    def __init__(self):
        # YOLO 모델 초기화
        self.model = YOLO("runs/detect/train4/weights/best.pt")  # 모델 로딩
        
    def generate_id(self):
        # Figma 스타일의 ID 생성 (예: "3:47")
        return f"{uuid.uuid4().int % 100}:{uuid.uuid4().int % 1000}"
    
    def extract_styles(self, image, bbox):
        """
        이미지에서 객체의 스타일 정보를 추출
        bbox: [x1, y1, x2, y2]
        """
        x1, y1, x2, y2 = map(int, bbox)
        
        # 바운딩 박스 영역 추출
        roi = image[y1:y2, x1:x2]
        if roi.size == 0:
            return None
            
        # PIL Image로 변환
        roi_pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        
        # 평균 색상 추출
        stat = ImageStat.Stat(roi_pil)
        r, g, b = stat.mean
        
        # 테두리 감지
        edges = cv2.Canny(roi, 100, 200)
        has_border = np.sum(edges) > 0
        
        # 배경색과 전경색 분리
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 전경 픽셀의 평균 색상
        fg_mask = binary > 0
        if np.any(fg_mask):
            fg_color = np.mean(roi[fg_mask], axis=0)
        else:
            fg_color = [r, g, b]
            
        # 배경 픽셀의 평균 색상
        bg_mask = binary == 0
        if np.any(bg_mask):
            bg_color = np.mean(roi[bg_mask], axis=0)
        else:
            bg_color = [r, g, b]
            
        return {
            "dominantColor": {
                "r": r / 255.0,
                "g": g / 255.0,
                "b": b / 255.0,
                "a": 1.0
            },
            "foregroundColor": {
                "r": fg_color[2] / 255.0,
                "g": fg_color[1] / 255.0,
                "b": fg_color[0] / 255.0,
                "a": 1.0
            },
            "backgroundColor": {
                "r": bg_color[2] / 255.0,
                "g": bg_color[1] / 255.0,
                "b": bg_color[0] / 255.0,
                "a": 1.0
            },
            "hasBorder": has_border,
            "borderThickness": 1.0 if has_border else 0.0
        }
    
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
            "styles": {
                "46:4": {
                    "key": "73a861256d166682302af8bab3eccf4cc5d45333",
                    "name": "Colors/Pink",
                    "styleType": "FILL",
                    "remote": True,
                    "description": ""
                }
            },
            "name": "Detection Results",
            "lastModified": datetime.now(timezone.utc).isoformat(),
            "version": str(uuid.uuid4().int),
            "role": "owner",
            "editorType": "detection"
        }
    
    def create_rectangle(self, x1, y1, x2, y2, class_name, styles):
        return {
            "id": self.generate_id(),
            "name": f"Rectangle {self.generate_id()}",
            "type": "RECTANGLE",
            "scrollBehavior": "SCROLLS",
            "boundVariables": {
                "fills": [{
                    "type": "VARIABLE_ALIAS",
                    "id": f"VariableID:{uuid.uuid4()}"
                }]
            },
            "blendMode": "PASS_THROUGH",
            "fills": [{
                "blendMode": "NORMAL",
                "type": "SOLID",
                "color": styles["dominantColor"]
            }],
            "strokes": [] if not styles["hasBorder"] else [{
                "blendMode": "NORMAL",
                "type": "SOLID",
                "color": styles["foregroundColor"],
                "thickness": styles["borderThickness"]
            }],
            "strokeWeight": styles["borderThickness"],
            "strokeAlign": "INSIDE",
            "styles": {
                "fill": "46:4"
            },
            "cornerRadius": 4.0,
            "cornerSmoothing": 0.0,
            "absoluteBoundingBox": {
                "x": x1,
                "y": y1,
                "width": x2 - x1,
                "height": y2 - y1
            },
            "absoluteRenderBounds": {
                "x": x1,
                "y": y1,
                "width": x2 - x1,
                "height": y2 - y1
            },
            "constraints": {
                "vertical": "TOP",
                "horizontal": "LEFT"
            },
            "effects": [],
            "interactions": []
        }

    def create_text(self, x1, y1, text, styles):
        return {
            "id": self.generate_id(),
            "name": text,
            "type": "TEXT",
            "scrollBehavior": "SCROLLS",
            "blendMode": "PASS_THROUGH",
            "fills": [{
                "blendMode": "NORMAL",
                "type": "SOLID",
                "color": styles["foregroundColor"]
            }],
            "strokes": [],
            "strokeWeight": 1.0,
            "strokeAlign": "OUTSIDE",
            "styles": {
                "fill": "3:43",
                "text": "3:44"
            },
            "absoluteBoundingBox": {
                "x": x1,
                "y": y1 - 20,
                "width": 100,
                "height": 20
            },
            "constraints": {
                "vertical": "TOP",
                "horizontal": "LEFT"
            },
            "characters": text,
            "style": {
                "fontFamily": "Pretendard",
                "fontPostScriptName": "Pretendard-SemiBold",
                "fontStyle": "SemiBold",
                "fontWeight": 600,
                "fontSize": 16.0,
                "textAlignHorizontal": "LEFT",
                "textAlignVertical": "CENTER",
                "letterSpacing": -0.2800000011920929,
                "lineHeightPx": 20.0,
                "lineHeightPercent": 104.17266082763672,
                "lineHeightPercentFontSize": 125.0,
                "lineHeightUnit": "PIXELS"
            }
        }

    def create_class_group(self, class_name, detections, image):
        group_id = self.generate_id()
        children = []
        
        for detection in detections:
            x1, y1, x2, y2 = detection.xyxy[0].tolist()
            conf = detection.conf[0]
            
            # 객체의 스타일 추출
            styles = self.extract_styles(image, [x1, y1, x2, y2])
            if styles is None:
                continue
                
            # Rectangle과 Text 추가
            rect = self.create_rectangle(x1, y1, x2, y2, class_name, styles)
            text = self.create_text(x1, y1, f"{conf:.2f}", styles)
            children.extend([rect, text])
        
        return {
            "id": group_id,
            "name": f"Group {class_name}",
            "type": "GROUP",
            "scrollBehavior": "SCROLLS",
            "children": children,
            "blendMode": "PASS_THROUGH",
            "clipsContent": False,
            "background": [],
            "fills": [],
            "strokes": [],
            "cornerRadius": 4.0,
            "cornerSmoothing": 0.0,
            "strokeWeight": 1.0,
            "strokeAlign": "INSIDE",
            "backgroundColor": {
                "r": 0.0,
                "g": 0.0,
                "b": 0.0,
                "a": 0.0
            },
            "effects": [],
            "interactions": []
        }

    def detect_and_create_json(self, image_path):
        # 이미지 로드
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
            
        results = self.model(image_path)
        json_data = self.create_base_structure()
        
        canvas = {
            "id": "0:1",
            "name": "Page 1",
            "type": "CANVAS",
            "scrollBehavior": "SCROLLS",
            "children": [],
            "backgroundColor": {
                "r": 0.9607843160629272,
                "g": 0.9607843160629272,
                "b": 0.9607843160629272,
                "a": 1.0
            }
        }
        
        # 클래스별로 탐지 결과 그룹화
        class_detections = defaultdict(list)
        for result in results:
            for detection in result.boxes:
                class_id = int(detection.cls[0])
                class_name = result.names[class_id]
                class_detections[class_name].append(detection)
        
        # 각 클래스별로 그룹 생성
        for class_name, detections in class_detections.items():
            class_group = self.create_class_group(class_name, detections, image)
            canvas["children"].append(class_group)
        
        json_data["document"]["children"] = [canvas]
        return json_data

def main():
    detector = DetectionJsonBuilder()
    image_path = "./screenshots/test.png"  # 실제 이미지 경로
    json_result = detector.detect_and_create_json(image_path)
    
    # JSON 결과 저장
    os.makedirs("json", exist_ok=True)
    with open("json/detection_results.json", "w", encoding="utf-8") as f:
        json.dump(json_result, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main() 