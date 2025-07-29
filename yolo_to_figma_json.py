import json
import cv2
import numpy as np
from ultralytics import YOLO
import os
from pathlib import Path
import datetime
import uuid

# 실제 학습 데이터셋 기준 경로
DATASET_PATH = 'yolo/datasets/screenshots/start'
IMAGES_PATH = os.path.join(DATASET_PATH, 'images')
LABELS_PATH = os.path.join(DATASET_PATH, 'labels')
CLASSES_PATH = os.path.join(LABELS_PATH, 'classes.txt')

# 클래스명 로드
with open(CLASSES_PATH, 'r', encoding='utf-8') as f:
    CLASS_NAMES = [line.strip() for line in f if line.strip()]

# YOLO 모델 로드
MODEL_PATH = "runs/detect/train_aug_clean/weights/best.pt"
if not os.path.exists(MODEL_PATH):
    print(f"Error: Model file not found at {MODEL_PATH}")
    exit(1)
model = YOLO(MODEL_PATH)

# YOLO 클래스명 → Figma type 매핑 예시
YOLO_TO_FIGMA_TYPE = {
    'Button': 'RECTANGLE',
    'InputBox': 'RECTANGLE',
    'TextArea': 'TEXT',
    'AppHeader': 'RECTANGLE',
    'ComboBox': 'RECTANGLE',
    'CheckBox': 'RECTANGLE',
    'CheckBoxGroup': 'GROUP',
    'RadioButton': 'RECTANGLE',
    'DateInput': 'RECTANGLE',
    'Output': 'RECTANGLE',
    'Group': 'GROUP',
    'Grid': 'RECTANGLE',
    'GridTitle': 'TEXT',
    'FormTitle': 'TEXT',
    # ... 필요시 추가
}

# 고유 ID 생성 (Figma 스타일)
def make_figma_id():
    return f"{np.random.randint(1, 100)}:{np.random.randint(1, 100000)}"

def yolo_to_figma_node(idx, row, image_w, image_h):
    """
    YOLO 감지 결과(row)를 Figma node(dict)로 변환
    """
    label = row['name']
    figma_type = YOLO_TO_FIGMA_TYPE.get(label, 'RECTANGLE')
    box = row['box']
    xmin, ymin, xmax, ymax = box['x1'], box['y1'], box['x2'], box['y2']
    node = {
        "id": make_figma_id(),
        "name": f"{label}-{idx}",
        "type": figma_type,
        "absoluteBoundingBox": {
            "x": float(xmin),
            "y": float(ymin),
            "width": float(xmax - xmin),
            "height": float(ymax - ymin)
        },
        "absoluteRenderBounds": {
            "x": float(xmin),
            "y": float(ymin),
            "width": float(xmax - xmin),
            "height": float(ymax - ymin)
        },
        "scrollBehavior": "SCROLLS",
        "blendMode": "PASS_THROUGH",
        "fills": [],
        "strokes": [],
        "strokeWeight": 1.0,
        "strokeAlign": "INSIDE",
        "children": []
    }
    # 텍스트 노드라면 characters 필드 샘플 추가
    if figma_type == 'TEXT':
        node["characters"] = label
        node["style"] = {
            "fontFamily": "Pretendard",
            "fontWeight": 400,
            "fontSize": 16.0,
            "textAlignHorizontal": "LEFT",
            "textAlignVertical": "CENTER"
        }
    return node

def yolo_results_to_figma_json(image_path, results):
    """
    YOLO 감지 결과를 Figma REST API 스타일 JSON으로 변환
    """
    img = cv2.imread(image_path)
    image_h, image_w = img.shape[:2]
    detected_elements_df = results[0].to_df()
    children = []
    for idx, row in detected_elements_df.iterrows():
        node = yolo_to_figma_node(idx, row, image_w, image_h)
        children.append(node)
    # Figma 최상위 구조
    figma_json = {
        "document": {
            "id": "0:0",
            "name": "Document",
            "type": "DOCUMENT",
            "scrollBehavior": "SCROLLS",
            "children": [
                {
                    "id": "0:1",
                    "name": Path(image_path).stem,
                    "type": "CANVAS",
                    "scrollBehavior": "SCROLLS",
                    "children": children,
                    "backgroundColor": {
                        "r": 1.0, "g": 1.0, "b": 1.0, "a": 1.0
                    }
                }
            ]
        },
        "components": {},
        "componentSets": {},
        "schemaVersion": 0,
        "styles": {},
        "name": Path(image_path).stem,
        "lastModified": datetime.datetime.now().isoformat(),
        "thumbnailUrl": "",
        "version": str(uuid.uuid4()),
        "role": "owner",
        "editorType": "figma",
        "linkAccess": "view"
    }
    return figma_json

def main():
    print("🚀 YOLO to Figma JSON Converter (screenshots/ 하위 이미지 전체)")
    print("=" * 50)
    today = datetime.datetime.now().strftime('%Y-%m-%d')
    figma_json_dir = f'./figma_json/{today}'
    os.makedirs(figma_json_dir, exist_ok=True)
    image_dir = './screenshots/'
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, file)
                print(f"\n📸 Processing image: {image_path}")
                img = cv2.imread(image_path)
                if img is None:
                    print(f"❌ Error: Failed to load image from {image_path}")
                    continue
                results = model(img, conf=0.05, verbose=False)
                figma_json = yolo_results_to_figma_json(image_path, results)
                output_json = os.path.join(figma_json_dir, f'{Path(file).stem}_figma.json')
                with open(output_json, 'w', encoding='utf-8') as f:
                    json.dump(figma_json, f, indent=2, ensure_ascii=False)
                print(f"💾 Figma JSON saved to: {output_json}")

if __name__ == "__main__":
    main() 