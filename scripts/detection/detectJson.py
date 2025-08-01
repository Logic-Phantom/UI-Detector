import json
import cv2
import numpy as np
from ultralytics import YOLO
import os
from pathlib import Path

# 실제 학습 데이터셋 기준 경로
DATASET_PATH = 'yolo/datasets/screenshots/start'
IMAGES_PATH = os.path.join(DATASET_PATH, 'images')
LABELS_PATH = os.path.join(DATASET_PATH, 'labels')
CLASSES_PATH = os.path.join(LABELS_PATH, 'classes.txt')

# 클래스명 로드
try:
    with open(CLASSES_PATH, 'r', encoding='utf-8') as f:
        CLASS_NAMES = [line.strip() for line in f if line.strip()]
except FileNotFoundError:
    # 대안 경로 시도
    alternative_paths = [
        'screenshots/start/labels/classes.txt',
        'output/result/classes.txt',
        'yolo/datasets/screenshots/val/labels/classes.txt'
    ]
    
    CLASS_NAMES = []
    for alt_path in alternative_paths:
        try:
            with open(alt_path, 'r', encoding='utf-8') as f:
                CLASS_NAMES = [line.strip() for line in f if line.strip()]
                print(f"✅ Loaded classes from: {alt_path}")
                break
        except FileNotFoundError:
            continue
    
    if not CLASS_NAMES:
        print("⚠️  Warning: Could not find classes.txt file. Using default class names.")
        CLASS_NAMES = ['Button', 'InputBox', 'TextArea', 'Group', 'Frame']

# 모델 경로
model_path = "runs/detect/train_aug_clean/weights/best.pt"
if not os.path.exists(model_path):
    print(f"Error: Model file not found at {model_path}")
    exit(1)
model = YOLO(model_path)

def detect_ui_elements(image_path, conf_threshold=0.05, iou_threshold=0.45):
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return None
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Failed to load image from {image_path}")
        return None
    print(f"Image loaded successfully: {img.shape}")
    results = model(img, conf=conf_threshold, iou=iou_threshold, verbose=True)
    detected_elements_df = results[0].to_df()
    print(f"\nDetection Results:")
    print(f"Confidence threshold: {conf_threshold}")
    print(f"IoU threshold: {iou_threshold}")
    print(f"Total detections: {len(detected_elements_df)}")
    if len(detected_elements_df) == 0:
        print("No objects detected. Trying with even lower confidence threshold...")
        results = model(img, conf=0.01, iou=0.3, verbose=True)
        detected_elements_df = results[0].to_df()
        print(f"Retry with lower threshold - Total detections: {len(detected_elements_df)}")
    if len(detected_elements_df) > 0:
        print("\nDetected Objects:")
        for idx, row in detected_elements_df.iterrows():
            print(f"  {idx}: {row['name']} (conf: {row['confidence']:.3f}) at {row['box']}")
    class_counter = {}
    ui_json = {
        "name": os.path.splitext(os.path.basename(image_path))[0],
        "elements": []
    }
    for idx, row in detected_elements_df.iterrows():
        box = row['box']
        xmin, ymin, xmax, ymax = box['x1'], box['y1'], box['x2'], box['y2']
        label = row['name']
        class_counter[label] = class_counter.get(label, 0) + 1
        element_id = f"{label}-{idx}"
        element = {
            "type": label,
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
    ui_json['summary'] = {
        "total_elements": len(ui_json['elements']),
        "unique_types": len(set([e['type'] for e in ui_json['elements']])),
        "class_counts": class_counter
    }
    return json.dumps(ui_json, indent=2)

if __name__ == "__main__":
    # 예시: 실제 학습 데이터셋의 이미지 사용
    image_path = os.path.join(IMAGES_PATH, 'workScr4.png')
    print("Starting UI Detection...")
    print(f"Model path: {model_path}")
    print(f"Image path: {image_path}")
    ui_json = detect_ui_elements(image_path)
    if ui_json:
        print("\nResulting UI JSON:")
        print(ui_json)
        output_path = './json/detection_results.json'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(ui_json)
        print(f"\nResults saved to: {output_path}")
    else:
        print("Detection failed!")
    print(f"\nModel Info:")
    print(f"Class names: {CLASS_NAMES}")
    print(f"Number of classes: {len(CLASS_NAMES)}")
