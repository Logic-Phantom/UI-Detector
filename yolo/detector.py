'''
Created on 2025. 5. 13.

@author: LCM
'''
import torch
import json
import cv2
import os
from pathlib import Path

# YOLO 모델 로드
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

def detect_ui_elements(image_path, conf_threshold=0.25):
    """
    UI 요소를 감지하는 함수
    
    Args:
        image_path: 이미지 경로
        conf_threshold: 신뢰도 임계값
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
    
    # YOLOv5 모델로 이미지 분석 (API 호환성을 위해 conf 파라미터 제거)
    results = model(img)
    
    # 결과를 JSON 형식으로 변환
    detected_elements = []
    
    if len(results.xywh[0]) > 0:
        print(f"Detected {len(results.xywh[0])} objects")
        
        for idx, det in enumerate(results.xywh[0]):
            # 감지된 요소의 클래스, 좌표, 크기
            class_id = int(det[5].item())  # 클래스 ID
            x, y, w, h = det[:4].tolist()  # 좌표 및 크기
            confidence = float(det[4].item())  # 신뢰도
            label = results.names[class_id]  # 클래스명
            
            # 신뢰도 임계값 필터링
            if confidence >= conf_threshold:
                print(f"  {idx}: {label} (conf: {confidence:.3f}) at ({x:.1f}, {y:.1f}, {w:.1f}, {h:.1f})")

                # 감지된 요소 정보
                detected_elements.append({
                    "type": label,
                    "id": f"{label}-{idx}",
                    "confidence": confidence,
                    "position": {
                        "top": f"{int(y)}px",
                        "left": f"{int(x)}px",
                        "width": f"{int(w)}px",
                        "height": f"{int(h)}px"
                    },
                    "children": []
                })
    else:
        print("No objects detected")
    
    # 계층형 JSON 구조로 포장
    ui_json = {
        "name": Path(image_path).stem,  # 이미지 이름을 UI 이름으로 사용
        "elements": detected_elements
    }

    return ui_json

if __name__ == "__main__":
    # 경로 수정 - 상대 경로를 올바르게 설정
    image_path = '../screenshots/test.png'
    
    # 절대 경로로 변환
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    image_path = os.path.join(project_root, 'screenshots', 'test.png')
    
    print(f"Current directory: {current_dir}")
    print(f"Project root: {project_root}")
    print(f"Image path: {image_path}")
    
    ui_json = detect_ui_elements(image_path)
    
    if ui_json:
        print("\nResulting UI JSON:")
        print(json.dumps(ui_json, indent=2))
        
        # 결과를 파일로 저장
        output_path = os.path.join(project_root, 'json', 'detection_results_yolo.json')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(ui_json, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {output_path}")
    else:
        print("Detection failed!")
