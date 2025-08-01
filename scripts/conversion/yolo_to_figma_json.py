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
    'Button': 'INSTANCE',  # 버튼은 INSTANCE로 설정하여 더 구체적인 타입 지정 가능
    'InputBox': 'INSTANCE',  # 입력박스도 INSTANCE로 설정
    'Input': 'INPUT',  # INPUT 타입 추가
    'TextArea': 'TEXT',
    'AppHeader': 'RECTANGLE',
    'ComboBox': 'INSTANCE',  # 콤보박스는 INSTANCE로 설정
    'CheckBox': 'RECTANGLE',
    'CheckBoxGroup': 'GROUP',
    'RadioButton': 'INSTANCE',  # 라디오버튼은 INSTANCE로 설정
    'DateInput': 'INSTANCE',  # 날짜입력도 INSTANCE로 설정
    'Output': 'TEXT',  # 출력은 TEXT로 설정
    'Group': 'GROUP',
    'Grid': 'RECTANGLE',
    'GridTitle': 'TEXT',
    'FormTitle': 'TEXT',
    'Pagination': 'INSTANCE',  # 페이지네이션 추가
    'SelectBox': 'INSTANCE',  # 셀렉트박스 추가
    'Table': 'FRAME',  # 테이블은 FRAME으로 설정 (Java에서 grid로 변환)
    'Title': 'FRAME',  # 제목은 FRAME으로 설정 (Java에서 udc로 변환)
    'Rectangle': 'RECTANGLE',  # 일반 사각형
    # ... 필요시 추가
}

# 그룹 컨트롤로 인식할 클래스들
GROUP_CONTROLS = ['Group', 'CheckBoxGroup', 'Grid']

# 컴포넌트 타입 매핑 (Java 코드의 cl: 타입들)
COMPONENT_TYPE_MAPPING = {
    'Button': 'button',
    'InputBox': 'inputbox', 
    'Input': 'inputbox',  # INPUT도 inputbox로 매핑
    'ComboBox': 'combobox',
    'SelectBox': 'combobox',
    'RadioButton': 'radiobutton',
    'DateInput': 'inputbox',
    'Pagination': 'pageindexer',
    'Output': 'output',
    'Table': 'grid',  # 테이블은 grid로 매핑
    'Title': 'udc',  # 제목은 udc로 매핑
    'Rectangle': 'group'  # 사각형은 group으로 매핑
}

def detect_component_type(label, parent_label=None):
    """
    Java 코드의 타입 감지 로직을 참고하여 컴포넌트 타입을 결정
    """
    lower_label = label.lower()
    lower_parent = parent_label.lower() if parent_label else ""
    
    # Table 감지 (Java에서 grid로 변환)
    if 'table' in lower_label:
        return 'grid'
    
    # Title Frame 감지 (Java에서 udc로 변환)
    if ('title' in lower_label and parent_label and 'frame' in parent_label.lower()):
        return 'udc'
    
    # ComboBox/SelectBox 감지
    if ('combobox' in lower_label or 'selectbox' in lower_label or 
        'combobox' in lower_parent or 'selectbox' in lower_parent or
        ('input' in lower_label and has_vector_in_right(lower_label))):
        return 'combobox'
    
    # InputBox/Input 감지
    if ('input' in lower_label or 'inputbox' in lower_label or
        'input' in lower_parent or 'inputbox' in lower_parent):
        return 'inputbox'
    
    # Pagination 감지
    if 'pagination' in lower_label or 'pageindexer' in lower_label:
        return 'pageindexer'
    
    # RadioButton 감지
    if ('radio' in lower_label or 'radiobutton' in lower_label or
        check_if_radio_button(lower_label)):
        return 'radiobutton'
    
    # Output 감지
    if 'output' in lower_label:
        return 'output'
    
    # Rectangle 감지 (Java에서 group으로 변환)
    if 'rectangle' in lower_label:
        return 'group'
    
    # 기본적으로 Button으로 처리
    return 'button'

def has_vector_in_right(label):
    """
    오른쪽에 벡터가 있는지 확인 (Java 코드 참고)
    """
    # 실제 구현에서는 더 복잡한 로직이 필요할 수 있음
    return 'right' in label and 'vector' in label

def check_if_radio_button(label):
    """
    라디오 버튼인지 확인 (Java 코드 참고)
    """
    return 'radio' in label

def has_multiple_radio_buttons(children):
    """
    자식 요소 중 라디오 버튼이 2개 이상인지 확인 (Java 코드 참고)
    """
    radio_count = 0
    for child in children:
        child_type = child.get('type', '')
        child_name = child.get('name', '')
        if child_type == 'INSTANCE' and 'radio' in child_name.lower():
            radio_count += 1
    return radio_count > 1

def find_first_text_value(node):
    """
    노드에서 첫 번째 텍스트 값을 찾기 (Java 코드 참고)
    """
    node_type = node.get('type', '')
    if node_type == 'TEXT':
        characters = node.get('characters')
        return characters if characters else None
    
    children = node.get('children', [])
    for child in children:
        result = find_first_text_value(child)
        if result and result.strip():
            return result
    return None

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

def yolo_to_figma_node(idx, row, image_w, image_h, parent_label=None):
    """
    YOLO 감지 결과(row)를 Figma node(dict)로 변환
    """
    label = row['name']
    figma_type = YOLO_TO_FIGMA_TYPE.get(label, 'RECTANGLE')
    
    # 더 정교한 타입 감지
    detected_component_type = detect_component_type(label, parent_label)
    component_type = COMPONENT_TYPE_MAPPING.get(label, detected_component_type)
    
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
    
    # INSTANCE 타입인 경우 componentType 정보 추가
    if figma_type == 'INSTANCE' and component_type:
        node["componentType"] = component_type
        # Java 코드의 cl: 타입을 참고하여 추가 속성 설정
        if component_type == 'button':
            node["value"] = label  # 버튼 텍스트
        elif component_type == 'inputbox':
            node["placeholder"] = label  # 입력박스 플레이스홀더
        elif component_type == 'combobox':
            node["options"] = []  # 콤보박스 옵션
        elif component_type == 'radiobutton':
            node["value"] = label  # 라디오버튼 값
        elif component_type == 'pageindexer':
            node["currentPage"] = 1  # 페이지네이션 현재 페이지
        elif component_type == 'output':
            node["value"] = label  # 출력 값
        elif component_type == 'grid':
            node["columns"] = 5  # 그리드 컬럼 수 (Java 코드 참고)
            node["rows"] = 1  # 그리드 행 수
        elif component_type == 'udc':
            node["title"] = label  # UDC 제목
        elif component_type == 'group':
            node["style"] = ""  # 그룹 스타일
    
    # INPUT 타입인 경우 처리
    if figma_type == 'INPUT':
        node["componentType"] = "inputbox"
        node["placeholder"] = label  # 입력박스 플레이스홀더
    
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

def build_hierarchical_structure(detected_elements_df):
    """
    감지된 요소들을 계층 구조로 정리
    """
    # 모든 요소를 노드로 변환 (부모 정보 없이 먼저 생성)
    nodes = []
    for idx, row in detected_elements_df.iterrows():
        node = yolo_to_figma_node(idx, row, 0, 0)  # image_w, image_h는 여기서는 사용하지 않음
        node['original_row'] = row  # 원본 데이터 보존
        nodes.append(node)
    
    # 그룹 컨트롤과 일반 요소 분리
    group_nodes = []
    regular_nodes = []
    
    for node in nodes:
        label = node['original_row']['name']
        if label in GROUP_CONTROLS:
            group_nodes.append(node)
        else:
            regular_nodes.append(node)
    
    # 각 그룹에 포함되는 요소들 찾기
    for group_node in group_nodes:
        group_box = group_node['original_row']['box']
        group_label = group_node['original_row']['name']
        
        # 이 그룹에 포함되는 일반 요소들 찾기
        contained_nodes = []
        remaining_nodes = []
        
        for regular_node in regular_nodes:
            if is_contained(group_box, regular_node['original_row']['box']):
                # 부모 정보를 고려하여 노드 재생성
                parent_label = group_label
                regular_label = regular_node['original_row']['name']
                
                # 부모 정보를 고려한 타입 감지로 노드 재생성
                new_node = yolo_to_figma_node(
                    regular_node['original_row'].name,  # idx
                    regular_node['original_row'],  # row
                    0, 0,  # image_w, image_h
                    parent_label  # parent_label
                )
                new_node['original_row'] = regular_node['original_row']
                contained_nodes.append(new_node)
            else:
                remaining_nodes.append(regular_node)
        
        # 라디오 버튼 그룹 처리 (Java 코드 참고)
        if has_multiple_radio_buttons(contained_nodes):
            # 라디오 버튼 그룹으로 처리
            group_node["radioGroup"] = True
            group_node["radioItems"] = []
            for radio_node in contained_nodes:
                if radio_node.get('componentType') == 'radiobutton':
                    text_value = find_first_text_value(radio_node)
                    if text_value:
                        group_node["radioItems"].append({
                            "label": text_value,
                            "value": text_value
                        })
        
        # 그룹의 children에 포함된 요소들 추가
        group_node['children'] = contained_nodes
        
        # 포함되지 않은 요소들만 남김
        regular_nodes = remaining_nodes
    
    # 최종 결과: 그룹들 + 포함되지 않은 일반 요소들
    final_nodes = group_nodes + regular_nodes
    
    # original_row 제거 (JSON 직렬화를 위해)
    for node in final_nodes:
        if 'original_row' in node:
            del node['original_row']
        # children도 정리
        for child in node.get('children', []):
            if 'original_row' in child:
                del child['original_row']
    
    return final_nodes

def yolo_results_to_figma_json(image_path, results):
    """
    YOLO 감지 결과를 Figma REST API 스타일 JSON으로 변환 (계층 구조 포함)
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

def main():
    print("🚀 YOLO to Figma JSON Converter (계층 구조 포함)")
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