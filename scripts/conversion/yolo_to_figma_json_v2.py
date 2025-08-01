import json
import cv2
import numpy as np
from ultralytics import YOLO
import os
from pathlib import Path
import datetime
import uuid

# 실제 학습 데이터셋 기준 경로
DATASET_PATH = '../../yolo/datasets/screenshots/start'
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
        '../../screenshots/start/labels/classes.txt',
        '../../output/result/classes.txt',
        '../../yolo/datasets/screenshots/val/labels/classes.txt'
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

# YOLO 모델 로드
MODEL_PATH = "../../runs/detect/train_aug_clean/weights/best.pt"
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

def make_constraints():
    return {
        "vertical": "TOP",
        "horizontal": "LEFT"
    }

def make_fills(color=None):
    if color is None:
        color = {"r": 1.0, "g": 1.0, "b": 1.0, "a": 1.0}
    return [{
        "blendMode": "NORMAL",
        "type": "SOLID",
        "color": color
    }]

def yolo_to_figma_node(idx, row, image_w, image_h):
    label = row['name']
    figma_type = YOLO_TO_FIGMA_TYPE.get(label, 'RECTANGLE')
    box = row['box']
    xmin, ymin, xmax, ymax = box['x1'], box['y1'], box['x2'], box['y2']
    node = {
        "id": make_figma_id(),
        "name": f"{label}-{idx}",
        "type": figma_type,
        "scrollBehavior": "SCROLLS",
        "blendMode": "PASS_THROUGH",
        "fills": make_fills(),
        "strokes": [],
        "strokeWeight": 1.0,
        "strokeAlign": "INSIDE",
        "cornerRadius": 4.0,
        "cornerSmoothing": 0.0,
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
        "constraints": make_constraints(),
        "effects": [],
        "interactions": [],
        "children": []
    }
    # 텍스트 노드라면 characters, style 등 샘플 추가
    if figma_type == 'TEXT':
        node["characters"] = label
        node["style"] = {
            "fontFamily": "Pretendard",
            "fontPostScriptName": "Pretendard-SemiBold",
            "fontStyle": "SemiBold",
            "fontWeight": 600,
            "fontSize": 16.0,
            "textAlignHorizontal": "LEFT",
            "textAlignVertical": "CENTER",
            "letterSpacing": -0.28,
            "lineHeightPx": 20.0,
            "lineHeightPercent": 104.17,
            "lineHeightPercentFontSize": 125.0,
            "lineHeightUnit": "PIXELS"
        }
        node["characterStyleOverrides"] = []
        node["styleOverrideTable"] = {}
        node["lineTypes"] = ["NONE"]
        node["lineIndentations"] = [0]
        node["layoutVersion"] = 4
    return node

def yolo_results_to_figma_json(image_path, results):
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
                    "children": [
                        {
                            "id": "3:47",
                            "name": f"Group {np.random.randint(10000, 99999)}",
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
                                "r": 0.96, "g": 0.96, "b": 0.96, "a": 1.0
                            },
                            "absoluteBoundingBox": {
                                "x": 0.0, "y": 0.0, "width": float(image_w), "height": float(image_h)
                            },
                            "absoluteRenderBounds": {
                                "x": 0.0, "y": 0.0, "width": float(image_w), "height": float(image_h)
                            },
                            "constraints": make_constraints(),
                            "effects": [],
                            "interactions": []
                        }
                    ],
                    "backgroundColor": {
                        "r": 1.0, "g": 1.0, "b": 1.0, "a": 1.0
                    },
                    "prototypeStartNodeID": None,
                    "flowStartingPoints": [],
                    "prototypeDevice": {
                        "type": "NONE",
                        "rotation": "NONE"
                    }
                }
            ]
        },
        "components": {},
        "componentSets": {},
        "schemaVersion": 0,
        "styles": {
            "35:2": {
                "key": "2ac120ac6ac512b38beae31f1b6f8b0ac4bf77ea",
                "name": "ui/color/primary",
                "styleType": "FILL",
                "remote": False,
                "description": ""
            },
            "3:43": {
                "key": "e5de19beaa78ad890a802e18e6d0b29b56898a01",
                "name": "Default/White",
                "styleType": "FILL",
                "remote": True,
                "description": ""
            },
            "3:44": {
                "key": "8110f9c4a71c1e2520887bb6951a482e5a72f7f8",
                "name": "title/h6",
                "styleType": "TEXT",
                "remote": True,
                "description": ""
            },
            "3:45": {
                "key": "fb6858e368c06159ec3fddcc558ac7640024d5b7",
                "name": "body/2xl",
                "styleType": "TEXT",
                "remote": True,
                "description": ""
            },
            "3:46": {
                "key": "85b71e7fa6a1819e9fb52a839ba36b14082944ab",
                "name": "title/h9",
                "styleType": "TEXT",
                "remote": True,
                "description": ""
            }
        },
        "name": Path(image_path).stem,
        "lastModified": datetime.datetime.now().isoformat(),
        "thumbnailUrl": "",
        "version": str(uuid.uuid4()),
        "role": "owner",
        "editorType": "figma",
        "linkAccess": "view"
    }
    return figma_json, detected_elements_df

def main():
    print("🚀 YOLO to Figma JSON Converter v2 (screenshots/ 하위 이미지 전체)")
    print("=" * 50)
    today = datetime.datetime.now().strftime('%Y-%m-%d')
    figma_json_dir = f'./figma_json_v2/{today}'
    figma_png_dir = f'./result_v2/{today}'
    os.makedirs(figma_json_dir, exist_ok=True)
    os.makedirs(figma_png_dir, exist_ok=True)
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
                figma_json, detected_elements_df = yolo_results_to_figma_json(image_path, results)
                output_json = os.path.join(figma_json_dir, f'{Path(file).stem}_figma.json')
                with open(output_json, 'w', encoding='utf-8') as f:
                    json.dump(figma_json, f, indent=2, ensure_ascii=False)
                print(f"💾 Figma JSON saved to: {output_json}")
                # 시각화 PNG 저장
                result_img = img.copy()
                for idx, row in detected_elements_df.iterrows():
                    box = row['box']
                    xmin, ymin, xmax, ymax = int(box['x1']), int(box['y1']), int(box['x2']), int(box['y2'])
                    label = row['name']
                    color = (0, 255, 0)
                    cv2.rectangle(result_img, (xmin, ymin), (xmax, ymax), color, 2)
                    cv2.putText(result_img, label, (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                output_png = os.path.join(figma_png_dir, f'{Path(file).stem}_figma.png')
                cv2.imwrite(output_png, result_img)
                print(f"🖼️  Visualization saved to: {output_png}")

if __name__ == "__main__":
    main() 