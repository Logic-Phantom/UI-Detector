import json
import cv2
import numpy as np
from ultralytics import YOLO
import os

# YOLOv5 모델 로드 - train4 모델 사용 (가장 성능이 좋은 모델)
model_path = "runs/detect/train4/weights/best.pt"
if not os.path.exists(model_path):
    print(f"Error: Model file not found at {model_path}")
    exit(1)

model = YOLO(model_path)

def detect_ui_elements(image_path, conf_threshold=0.1, iou_threshold=0.45):
    """
    UI 요소를 감지하는 함수
    
    Args:
        image_path: 이미지 경로
        conf_threshold: 신뢰도 임계값 (기본값: 0.1 - 더 낮게 설정)
        iou_threshold: IoU 임계값 (기본값: 0.45)
    """
    # 이미지 존재 확인
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return None
    
    # 이미지 로드
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Failed to load image from {image_path}")
        return None
    
    print(f"Image loaded successfully: {img.shape}")
    
    # YOLOv5 모델로 이미지 분석 (임계값 설정)
    results = model(img, conf=conf_threshold, iou=iou_threshold, verbose=True)
    
    # 결과에서 pandas 형식으로 변환
    detected_elements_df = results[0].to_df()
    
    # 디버깅용: 감지된 객체 출력
    print(f"\nDetection Results:")
    print(f"Confidence threshold: {conf_threshold}")
    print(f"IoU threshold: {iou_threshold}")
    print(f"Total detections: {len(detected_elements_df)}")
    
    if len(detected_elements_df) == 0:
        print("No objects detected. Trying with even lower confidence threshold...")
        # 더 낮은 신뢰도로 재시도
        results = model(img, conf=0.05, iou=0.3, verbose=True)
        detected_elements_df = results[0].to_df()
        print(f"Retry with lower threshold - Total detections: {len(detected_elements_df)}")
    
    # 감지된 객체 상세 정보 출력
    if len(detected_elements_df) > 0:
        print("\nDetected Objects:")
        for idx, row in detected_elements_df.iterrows():
            print(f"  {idx}: {row['name']} (conf: {row['confidence']:.3f}) at {row['box']}")
    
    ui_json = {
        "name": os.path.splitext(os.path.basename(image_path))[0],
        "elements": []
    }
    
    # 각 감지된 요소를 JSON으로 변환
    for idx, row in detected_elements_df.iterrows():
        box = row['box']
        xmin, ymin, xmax, ymax = box['x1'], box['y1'], box['x2'], box['y2']
        
        element_id = f"{row['name']}-{idx}"
        
        element = {
            "type": row['name'],
            "id": element_id,
            "confidence": float(row['confidence']),
            "position": {
                "top": f"{int(ymin)}px",
                "left": f"{int(xmin)}px",
                "width": f"{int(xmax - xmin)}px",
                "height": f"{int(ymax - ymin)}px"
            },
            "children": []
        }
        
        ui_json['elements'].append(element)
    
    return json.dumps(ui_json, indent=2)

# 테스트 실행
if __name__ == "__main__":
    image_path = './screenshots/test.png'
    
    print("Starting UI Detection...")
    print(f"Model path: {model_path}")
    print(f"Image path: {image_path}")
    
    ui_json = detect_ui_elements(image_path)
    
    if ui_json:
        print("\nResulting UI JSON:")
        print(ui_json)
        
        # 결과를 파일로 저장
        output_path = './json/detection_results.json'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(ui_json)
        print(f"\nResults saved to: {output_path}")
    else:
        print("Detection failed!")
    
    # 모델 정보 출력
    print(f"\nModel Info:")
    print(f"Class names: {model.names}")
    print(f"Number of classes: {len(model.names)}")
