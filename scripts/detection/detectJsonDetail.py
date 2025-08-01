import json
import cv2
import numpy as np
from ultralytics import YOLO
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

class UIDetector:
    def __init__(self, model_path="runs/detect/train6/weights/best.pt"):
        """
        UI 요소 탐지기 초기화
        
        Args:
            model_path: 학습된 모델 경로
        """
        self.model_path = model_path
        if not os.path.exists(model_path):
            print(f"Warning: Custom model not found at {model_path}")
            print("Using default YOLOv5s model...")
            self.model = YOLO("yolov5s.pt")
        else:
            self.model = YOLO(model_path)
        
        print(f"Model loaded: {self.model_path}")
        print(f"Available classes: {self.model.names}")
    
    def detect_with_visualization(self, image_path, conf_threshold=0.25, save_result=True):
        """
        UI 요소를 탐지하고 결과를 시각화합니다.
        
        Args:
            image_path: 이미지 경로
            conf_threshold: 신뢰도 임계값
            save_result: 결과 이미지 저장 여부
        """
        # 이미지 로드
        if not os.path.exists(image_path):
            print(f"Error: Image file not found at {image_path}")
            return None
        
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Failed to load image from {image_path}")
            return None
        
        print(f"Image loaded: {img.shape}")
        
        # 탐지 수행
        results = self.model(img, conf=conf_threshold, verbose=True)
        
        # 결과 분석
        detected_elements = []
        result_img = img.copy()
        
        if len(results) > 0:
            result = results[0]
            if hasattr(result, 'boxes') and result.boxes is not None:
                boxes = result.boxes
                if len(boxes) > 0:
                    print(f"Detected {len(boxes)} objects")
                    
                    for i, box in enumerate(boxes):
                        # 박스 좌표
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        
                        # 클래스 정보
                        cls = int(box.cls[0].cpu().numpy())
                        conf = float(box.conf[0].cpu().numpy())
                        label = self.model.names[cls]
                        
                        print(f"  {i}: {label} (conf: {conf:.3f}) at ({x1}, {y1}, {x2}, {y2})")
                        
                        # JSON 데이터 생성
                        element = {
                            "type": label,
                            "id": f"{label}-{i}",
                            "confidence": conf,
                            "position": {
                                "top": f"{y1}px",
                                "left": f"{x1}px",
                                "width": f"{x2 - x1}px",
                                "height": f"{y2 - y1}px"
                            },
                            "bbox": [x1, y1, x2, y2],
                            "children": []
                        }
                        detected_elements.append(element)
                        
                        # 시각화
                        color = self._get_color(cls)
                        cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)
                        
                        # 라벨 텍스트
                        label_text = f"{label}: {conf:.2f}"
                        (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        cv2.rectangle(result_img, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
                        cv2.putText(result_img, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                else:
                    print("No objects detected with current threshold")
            else:
                print("No detection results found")
        else:
            print("No detection results")
        
        # 결과 저장
        if save_result and len(detected_elements) > 0:
            output_dir = Path("result")
            output_dir.mkdir(exist_ok=True)
            
            # 시각화 결과 저장
            result_path = output_dir / "detected_result.png"
            cv2.imwrite(str(result_path), result_img)
            print(f"Visualization saved to: {result_path}")
        
        # JSON 결과 생성
        ui_json = {
            "name": Path(image_path).stem,
            "image_path": image_path,
            "detection_settings": {
                "confidence_threshold": conf_threshold,
                "model_path": self.model_path
            },
            "elements": detected_elements,
            "summary": {
                "total_elements": len(detected_elements),
                "unique_types": len(set([elem["type"] for elem in detected_elements]))
            }
        }
        
        return ui_json, result_img
    
    def _get_color(self, class_id):
        """클래스별 색상 반환"""
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
            (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
            (64, 0, 0), (0, 64, 0), (0, 0, 64), (64, 64, 0),
            (64, 0, 64), (0, 64, 64), (192, 0, 0), (0, 192, 0),
            (0, 0, 192), (192, 192, 0), (192, 0, 192), (0, 192, 192)
        ]
        return colors[class_id % len(colors)]
    
    def analyze_detection_results(self, ui_json):
        """탐지 결과를 분석하고 통계를 출력합니다."""
        if not ui_json or "elements" not in ui_json:
            print("No detection results to analyze")
            return
        
        elements = ui_json["elements"]
        if len(elements) == 0:
            print("No elements detected")
            return
        
        # 타입별 통계
        type_counts = {}
        confidence_scores = []
        
        for elem in elements:
            elem_type = elem["type"]
            type_counts[elem_type] = type_counts.get(elem_type, 0) + 1
            confidence_scores.append(elem["confidence"])
        
        print("\n=== Detection Analysis ===")
        print(f"Total elements detected: {len(elements)}")
        print(f"Unique element types: {len(type_counts)}")
        print(f"Average confidence: {np.mean(confidence_scores):.3f}")
        print(f"Min confidence: {np.min(confidence_scores):.3f}")
        print(f"Max confidence: {np.max(confidence_scores):.3f}")
        
        print("\nElement types distribution:")
        for elem_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {elem_type}: {count}")
        
        # 신뢰도 분포
        print(f"\nConfidence distribution:")
        conf_ranges = [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0)]
        for low, high in conf_ranges:
            count = sum(1 for conf in confidence_scores if low <= conf < high)
            print(f"  {low:.1f}-{high:.1f}: {count} elements")

def main():
    """메인 실행 함수"""
    # 탐지기 초기화
    detector = UIDetector()
    
    # 테스트 이미지 경로
    image_path = './screenshots/test.png'
    
    print("Starting detailed UI detection...")
    print(f"Image path: {image_path}")
    
    # 탐지 수행
    ui_json, result_img = detector.detect_with_visualization(image_path, conf_threshold=0.25)
    
    if ui_json:
        # 결과 분석
        detector.analyze_detection_results(ui_json)
        
        # JSON 결과 저장
        output_path = './json/detection_results_detailed.json'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(ui_json, f, indent=2, ensure_ascii=False)
        print(f"\nDetailed results saved to: {output_path}")
        
        # JSON 결과 출력
        print("\n=== Detection Results ===")
        print(json.dumps(ui_json, indent=2, ensure_ascii=False))
    else:
        print("Detection failed!")

if __name__ == "__main__":
    main() 