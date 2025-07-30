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

# YOLO 클래스명 → Figma type 매핑
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
    'Frame': 'FRAME',
    'Container': 'FRAME',
    'Panel': 'FRAME',
    'Section': 'FRAME',
    'Header': 'FRAME',
    'Footer': 'FRAME',
    'Sidebar': 'FRAME',
    'MainContent': 'FRAME',
    'Navigation': 'FRAME',
    'SearchArea': 'FRAME',
    'FilterArea': 'FRAME',
    'ResultArea': 'FRAME',
    'FormArea': 'FRAME',
    'ButtonGroup': 'GROUP',
    'InputGroup': 'GROUP',
    'LabelGroup': 'GROUP',
    'ControlGroup': 'GROUP',
}

# 그룹 컨트롤로 인식할 클래스들
GROUP_CONTROLS = ['Group', 'CheckBoxGroup', 'ButtonGroup', 'InputGroup', 'LabelGroup', 'ControlGroup']

# 프레임 컨트롤로 인식할 클래스들
FRAME_CONTROLS = ['Frame', 'Container', 'Panel', 'Section', 'Header', 'Footer', 'Sidebar', 
                  'MainContent', 'Navigation', 'SearchArea', 'FilterArea', 'ResultArea', 'FormArea']

# 고유 ID 생성 (Figma 스타일)
def make_figma_id():
    return f"{np.random.randint(1, 100)}:{np.random.randint(1, 100000)}"

def is_contained(box1, box2):
    """
    box1이 box2를 완전히 포함하는지 확인
    """
    x1_1, y1_1, x2_1, y2_1 = box1['x1'], box1['y1'], box1['x2'], box1['y2']
    x1_2, y1_2, x2_2, y2_2 = box2['x1'], box2['y1'], box2['x2'], box2['y2']
    
    return (x1_1 <= x1_2 and y1_1 <= y1_2 and x2_1 >= x2_2 and y2_1 >= y2_2)

def yolo_to_figma_node(idx, row, image_w, image_h):
    """
    YOLO 감지 결과(row)를 Figma node(dict)로 변환 (rest.json 스타일)
    """
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
        "fills": [{
            "blendMode": "NORMAL",
            "type": "SOLID",
            "color": {
                "r": 1.0, "g": 1.0, "b": 1.0, "a": 1.0
            }
        }] if figma_type in ["RECTANGLE", "FRAME", "GROUP"] else [],
        "strokes": [],
        "strokeWeight": 1.0,
        "strokeAlign": "OUTSIDE",
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
        "constraints": {
            "vertical": "TOP",
            "horizontal": "LEFT"
        },
        "children": []
    }

    # TEXT 노드라면 rest.json 스타일 필드 추가
    if figma_type == 'TEXT':
        node["characters"] = label
        node["styles"] = {
            "fill": "1:3172",
            "text": "1:4305"
        }
        node["characterStyleOverrides"] = []
        node["style"] = {
            "fontFamily": "Pretendard",
            "fontWeight": 400,
            "fontSize": 16.0,
            "textAlignHorizontal": "LEFT",
            "textAlignVertical": "CENTER"
        }

    # FRAME 타입이라면 추가 속성 설정
    if figma_type == 'FRAME':
        node["clipsContent"] = False
        node["background"] = []
        node["backgroundColor"] = {
            "r": 1.0, "g": 1.0, "b": 1.0, "a": 1.0
        }
        node["layoutMode"] = "HORIZONTAL"
        node["counterAxisSizingMode"] = "FIXED"
        node["primaryAxisSizingMode"] = "FIXED"
        node["layoutWrap"] = "NO_WRAP"
        node["layoutAlign"] = "INHERIT"
        node["layoutGrow"] = 0.0
        node["layoutSizingHorizontal"] = "FIXED"
        node["layoutSizingVertical"] = "FIXED"
        node["effects"] = []
        node["interactions"] = []

    return node

def build_hierarchical_structure(detected_elements_df):
    """
    감지된 요소들을 복잡한 계층 구조로 정리
    """
    # 모든 요소를 노드로 변환
    nodes = []
    for idx, row in detected_elements_df.iterrows():
        node = yolo_to_figma_node(idx, row, 0, 0)
        node['original_row'] = row
        nodes.append(node)
    
    # 프레임, 그룹, 일반 요소 분리
    frame_nodes = []
    group_nodes = []
    regular_nodes = []
    
    for node in nodes:
        label = node['original_row']['name']
        if label in FRAME_CONTROLS:
            frame_nodes.append(node)
        elif label in GROUP_CONTROLS:
            group_nodes.append(node)
        else:
            regular_nodes.append(node)
    
    # 각 프레임에 포함되는 요소들 찾기 (그룹과 일반 요소 모두)
    for frame_node in frame_nodes:
        frame_box = frame_node['original_row']['box']
        
        # 이 프레임에 포함되는 그룹들 찾기
        contained_groups = []
        remaining_groups = []
        
        for group_node in group_nodes:
            if is_contained(frame_box, group_node['original_row']['box']):
                contained_groups.append(group_node)
            else:
                remaining_groups.append(group_node)
        
        # 이 프레임에 포함되는 일반 요소들 찾기
        contained_regulars = []
        remaining_regulars = []
        
        for regular_node in regular_nodes:
            if is_contained(frame_box, regular_node['original_row']['box']):
                contained_regulars.append(regular_node)
            else:
                remaining_regulars.append(regular_node)
        
        # 각 그룹에 포함되는 일반 요소들 찾기
        for group_node in contained_groups:
            group_box = group_node['original_row']['box']
            
            group_contained = []
            group_remaining = []
            
            for regular_node in contained_regulars:
                if is_contained(group_box, regular_node['original_row']['box']):
                    group_contained.append(regular_node)
                else:
                    group_remaining.append(regular_node)
            
            # 그룹의 children에 포함된 요소들 추가
            group_node['children'] = group_contained
            
            # 그룹에 포함되지 않은 요소들은 프레임의 직접 children으로
            contained_regulars = group_remaining
        
        # 프레임의 children에 그룹들과 포함되지 않은 일반 요소들 추가
        frame_node['children'] = contained_groups + contained_regulars
        
        # 포함되지 않은 요소들만 남김
        group_nodes = remaining_groups
        regular_nodes = remaining_regulars
    
    # 남은 그룹들에 포함되는 요소들 찾기
    for group_node in group_nodes:
        group_box = group_node['original_row']['box']
        
        contained_nodes = []
        remaining_nodes = []
        
        for regular_node in regular_nodes:
            if is_contained(group_box, regular_node['original_row']['box']):
                contained_nodes.append(regular_node)
            else:
                remaining_nodes.append(regular_node)
        
        group_node['children'] = contained_nodes
        regular_nodes = remaining_nodes
    
    # 최종 결과: 프레임들 + 포함되지 않은 그룹들 + 포함되지 않은 일반 요소들
    final_nodes = frame_nodes + group_nodes + regular_nodes
    
    # original_row 제거 (JSON 직렬화를 위해)
    def clean_node(node):
        if 'original_row' in node:
            del node['original_row']
        for child in node.get('children', []):
            clean_node(child)
    
    for node in final_nodes:
        clean_node(node)
    
    return final_nodes

def yolo_results_to_figma_json(image_path, results):
    """
    YOLO 감지 결과를 Figma REST API 스타일 JSON으로 변환 (복잡한 계층 구조 포함)
    """
    img = cv2.imread(image_path)
    image_h, image_w = img.shape[:2]
    detected_elements_df = results[0].to_df()
    
    # 계층 구조로 정리된 children 생성
    children = build_hierarchical_structure(detected_elements_df)
    
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

def save_detection_result_with_image(image_path, results, output_dir):
    """
    감지 결과를 JSON과 PNG로 저장
    """
    # JSON 저장
    figma_json = yolo_results_to_figma_json(image_path, results)
    json_filename = f'{Path(image_path).stem}_figma.json'
    json_path = os.path.join(output_dir, json_filename)
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(figma_json, f, indent=2, ensure_ascii=False)
    
    # PNG 저장 (감지 결과 시각화)
    img = cv2.imread(image_path)
    annotated_img = results[0].plot()
    
    png_filename = f'{Path(image_path).stem}_detection.png'
    png_path = os.path.join(output_dir, png_filename)
    
    cv2.imwrite(png_path, annotated_img)
    
    return json_path, png_path

def main():
    print("🚀 YOLO to Figma JSON Converter v3 (복잡한 계층 구조 + PNG 저장)")
    print("=" * 60)
    
    today = datetime.datetime.now().strftime('%Y-%m-%d')
    output_dir = f'./figma_json/{today}'
    os.makedirs(output_dir, exist_ok=True)
    
    image_dir = './screenshots/'
    processed_count = 0
    
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
                
                try:
                    json_path, png_path = save_detection_result_with_image(image_path, results, output_dir)
                    print(f"💾 JSON saved to: {json_path}")
                    print(f"🖼️  PNG saved to: {png_path}")
                    processed_count += 1
                except Exception as e:
                    print(f"❌ Error processing {image_path}: {str(e)}")
    
    print(f"\n✅ Processing complete! {processed_count} files processed.")
    print(f"📁 Output directory: {output_dir}")

if __name__ == "__main__":
    main() 