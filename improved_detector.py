import json
import cv2
import numpy as np
from ultralytics import YOLO
import os
import matplotlib.pyplot as plt
from pathlib import Path

class ImprovedUIDetector:
    def __init__(self, model_path="runs/detect/train4/weights/best.pt"):
        """
        개선된 UI 요소 탐지기 초기화
        
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
        
        print(f"✅ Model loaded: {self.model_path}")
        print(f"📋 Available classes: {len(self.model.names)}")
    
    def detect_with_multiple_thresholds(self, image_path, save_visualization=True):
        """
        여러 임계값으로 탐지를 수행하고 최적의 결과를 선택합니다.
        
        Args:
            image_path: 이미지 경로
            save_visualization: 시각화 결과 저장 여부
        """
        if not os.path.exists(image_path):
            print(f"❌ Error: Image file not found at {image_path}")
            return None
        
        img = cv2.imread(image_path)
        if img is None:
            print(f"❌ Error: Failed to load image from {image_path}")
            return None
        
        print(f"📸 Image loaded: {img.shape}")
        
        # 다양한 임계값으로 탐지 시도
        thresholds = [0.05, 0.1, 0.15, 0.2, 0.25]
        best_result = None
        best_score = 0
        
        for conf_thresh in thresholds:
            print(f"\n🔍 Testing with confidence threshold: {conf_thresh}")
            
            try:
                results = self.model(img, conf=conf_thresh, verbose=False)
                
                if len(results) > 0:
                    result = results[0]
                    if hasattr(result, 'boxes') and result.boxes is not None:
                        boxes = result.boxes
                        if len(boxes) > 0:
                            # 평균 신뢰도 계산
                            confidences = [float(box.conf[0].cpu().numpy()) for box in boxes]
                            avg_confidence = np.mean(confidences)
                            
                            # 탐지 개수와 평균 신뢰도의 조합으로 점수 계산
                            score = len(boxes) * avg_confidence
                            
                            print(f"  Detected {len(boxes)} objects (avg conf: {avg_confidence:.3f}, score: {score:.3f})")
                            
                            if score > best_score:
                                best_score = score
                                best_result = {
                                    'threshold': conf_thresh,
                                    'results': results,
                                    'score': score,
                                    'num_detections': len(boxes),
                                    'avg_confidence': avg_confidence
                                }
                        else:
                            print("  No objects detected")
                    else:
                        print("  No detection boxes found")
                else:
                    print("  No detection results")
                    
            except Exception as e:
                print(f"  ❌ Error during inference: {e}")
        
        if best_result is None:
            print("❌ No successful detections found")
            return None
        
        print(f"\n🎯 Best detection result:")
        print(f"  Threshold: {best_result['threshold']}")
        print(f"  Detections: {best_result['num_detections']}")
        print(f"  Average confidence: {best_result['avg_confidence']:.3f}")
        print(f"  Score: {best_result['score']:.3f}")
        
        # 최적 결과로 JSON 생성
        ui_json = self._create_json_result(best_result['results'], image_path, best_result)
        
        # 시각화 생성
        if save_visualization:
            self._create_visualization(best_result['results'], img, image_path)
        
        return ui_json
    
    def _create_json_result(self, results, image_path, detection_info):
        """탐지 결과를 JSON 형식으로 변환"""
        detected_elements = []
        
        if len(results) > 0:
            result = results[0]
            if hasattr(result, 'boxes') and result.boxes is not None:
                boxes = result.boxes
                
                for i, box in enumerate(boxes):
                    # 박스 좌표
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # 클래스 정보
                    cls = int(box.cls[0].cpu().numpy())
                    conf = float(box.conf[0].cpu().numpy())
                    label = self.model.names[cls]
                    
                    # 요소 정보 생성
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
                        "area": (x2 - x1) * (y2 - y1),
                        "children": []
                    }
                    detected_elements.append(element)
        
        # JSON 구조 생성
        ui_json = {
            "name": Path(image_path).stem,
            "image_path": image_path,
            "detection_info": {
                "threshold": detection_info['threshold'],
                "total_detections": detection_info['num_detections'],
                "average_confidence": detection_info['avg_confidence'],
                "detection_score": detection_info['score'],
                "model_path": self.model_path
            },
            "elements": detected_elements,
            "summary": {
                "total_elements": len(detected_elements),
                "unique_types": len(set([elem["type"] for elem in detected_elements])),
                "total_area": sum([elem["area"] for elem in detected_elements])
            }
        }
        
        return ui_json
    
    def _create_visualization(self, results, img, image_path):
        """탐지 결과를 시각화하고 저장"""
        result_img = img.copy()
        
        if len(results) > 0:
            result = results[0]
            if hasattr(result, 'boxes') and result.boxes is not None:
                boxes = result.boxes
                
                for i, box in enumerate(boxes):
                    # 박스 좌표
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # 클래스 정보
                    cls = int(box.cls[0].cpu().numpy())
                    conf = float(box.conf[0].cpu().numpy())
                    label = self.model.names[cls]
                    
                    # 색상 선택
                    color = self._get_color(cls)
                    
                    # 박스 그리기
                    cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)
                    
                    # 라벨 텍스트
                    label_text = f"{label}: {conf:.2f}"
                    (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    
                    # 라벨 배경
                    cv2.rectangle(result_img, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
                    
                    # 라벨 텍스트
                    cv2.putText(result_img, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 결과 저장
        output_dir = Path("result")
        output_dir.mkdir(exist_ok=True)
        
        result_path = output_dir / "improved_detection_result.png"
        cv2.imwrite(str(result_path), result_img)
        print(f"📊 Visualization saved to: {result_path}")
    
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
    
    def analyze_results(self, ui_json):
        """탐지 결과를 분석하고 통계를 출력"""
        if not ui_json or "elements" not in ui_json:
            print("❌ No detection results to analyze")
            return
        
        elements = ui_json["elements"]
        if len(elements) == 0:
            print("❌ No elements detected")
            return
        
        print("\n📊 Detection Analysis:")
        print("=" * 50)
        
        # 기본 통계
        print(f"Total elements: {len(elements)}")
        print(f"Detection threshold: {ui_json['detection_info']['threshold']}")
        print(f"Average confidence: {ui_json['detection_info']['average_confidence']:.3f}")
        print(f"Detection score: {ui_json['detection_info']['detection_score']:.3f}")
        
        # 타입별 통계
        type_counts = {}
        confidence_scores = []
        areas = []
        
        for elem in elements:
            elem_type = elem["type"]
            type_counts[elem_type] = type_counts.get(elem_type, 0) + 1
            confidence_scores.append(elem["confidence"])
            areas.append(elem["area"])
        
        print(f"\nElement types ({len(type_counts)} unique):")
        for elem_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {elem_type}: {count}")
        
        print(f"\nConfidence statistics:")
        print(f"  Min: {np.min(confidence_scores):.3f}")
        print(f"  Max: {np.max(confidence_scores):.3f}")
        print(f"  Mean: {np.mean(confidence_scores):.3f}")
        print(f"  Std: {np.std(confidence_scores):.3f}")
        
        print(f"\nArea statistics:")
        print(f"  Total area: {sum(areas):,} pixels")
        print(f"  Average area: {np.mean(areas):.0f} pixels")
        print(f"  Largest element: {max(areas):,} pixels")

def main():
    """메인 실행 함수"""
    print("🚀 Improved UI Detector")
    print("=" * 50)
    
    # 탐지기 초기화
    detector = ImprovedUIDetector()
    
    # 테스트 이미지 경로
    image_path = './screenshots/test.png'
    
    print(f"📸 Processing image: {image_path}")
    
    # 탐지 수행
    ui_json = detector.detect_with_multiple_thresholds(image_path)
    
    if ui_json:
        # 결과 분석
        detector.analyze_results(ui_json)
        
        # JSON 결과 저장
        output_path = './json/improved_detection_results.json'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(ui_json, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Results saved to: {output_path}")
        
        # JSON 결과 출력
        print("\n📄 Detection Results:")
        print(json.dumps(ui_json, indent=2, ensure_ascii=False))
    else:
        print("❌ Detection failed!")

if __name__ == "__main__":
    main() 