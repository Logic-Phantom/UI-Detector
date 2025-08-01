import json
import cv2
import numpy as np
from ultralytics import YOLO
import os
from pathlib import Path
import datetime
import uuid
import re
from PIL import Image, ImageDraw, ImageFont
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    print("⚠️  Tesseract OCR not available. Using fallback text detection.")

try:
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️  Scikit-learn not available. Using fallback color analysis.")

import colorsys

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

# YOLO 모델 로드
MODEL_PATH = "runs/detect/train_aug_clean/weights/best.pt"
if not os.path.exists(MODEL_PATH):
    print(f"Error: Model file not found at {MODEL_PATH}")
    exit(1)
model = YOLO(MODEL_PATH)

# YOLO 클래스명 → Figma type 매핑
YOLO_TO_FIGMA_TYPE = {
    'Button': 'INSTANCE',
    'InputBox': 'INSTANCE',
    'TextArea': 'TEXT',
    'AppHeader': 'RECTANGLE',
    'ComboBox': 'INSTANCE',
    'CheckBox': 'RECTANGLE',
    'CheckBoxGroup': 'GROUP',
    'RadioButton': 'INSTANCE',
    'DateInput': 'INSTANCE',
    'Output': 'TEXT',
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
    'Pagination': 'INSTANCE',
    'SelectBox': 'INSTANCE',
}

# 그룹 컨트롤로 인식할 클래스들
GROUP_CONTROLS = ['Group', 'CheckBoxGroup', 'ButtonGroup', 'InputGroup', 'LabelGroup', 'ControlGroup']

# 프레임 컨트롤로 인식할 클래스들
FRAME_CONTROLS = ['Frame', 'Container', 'Panel', 'Section', 'Header', 'Footer', 'Sidebar', 
                  'MainContent', 'Navigation', 'SearchArea', 'FilterArea', 'ResultArea', 'FormArea']

# 컴포넌트 타입 매핑
COMPONENT_TYPE_MAPPING = {
    'Button': 'button',
    'InputBox': 'inputbox', 
    'ComboBox': 'combobox',
    'SelectBox': 'combobox',
    'RadioButton': 'radiobutton',
    'DateInput': 'inputbox',
    'Pagination': 'pageindexer',
    'Output': 'output'
}

def extract_text_from_region(image, box):
    """
    특정 영역에서 텍스트를 추출합니다.
    """
    try:
        x1, y1, x2, y2 = int(box['x1']), int(box['y1']), int(box['x2']), int(box['y2'])
        
        # 이미지 크기 확인
        height, width = image.shape[:2]
        x1 = max(0, min(x1, width))
        y1 = max(0, min(y1, height))
        x2 = max(0, min(x2, width))
        y2 = max(0, min(y2, height))
        
        if x1 >= x2 or y1 >= y2:
            return ""
        
        # ROI 추출
        roi = image[y1:y2, x1:x2]
        
        if roi.size == 0:
            return ""
        
        # Tesseract OCR이 사용 가능한 경우
        if TESSERACT_AVAILABLE:
            try:
                # 그레이스케일 변환
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                
                # 노이즈 제거
                denoised = cv2.medianBlur(gray, 3)
                
                # 이진화
                _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                # OCR 수행
                custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,!?@#$%^&*()_+-=[]{}|;:,.<>?/ '
                text = pytesseract.image_to_string(binary, config=custom_config, lang='eng')
                
                # 텍스트 정리
                text = text.strip()
                text = re.sub(r'\s+', ' ', text)  # 여러 공백을 하나로
                
                if text:
                    return text
            except Exception as e:
                print(f"OCR Error: {e}")
        
        # OCR이 실패하거나 사용할 수 없는 경우, 라벨 기반 텍스트 추정
        return estimate_text_from_label(box, image)
            
    except Exception as e:
        print(f"Text extraction error: {e}")
        return ""

def estimate_text_from_label(box, image):
    """
    라벨과 영역 크기를 기반으로 텍스트를 추정합니다.
    """
    try:
        x1, y1, x2, y2 = int(box['x1']), int(box['y1']), int(box['x2']), int(box['y2'])
        width = x2 - x1
        height = y2 - y1
        
        # 영역이 너무 작으면 텍스트가 없을 가능성이 높음
        if width < 20 or height < 10:
            return ""
        
        # 영역 크기와 비율을 기반으로 텍스트 추정
        if width > 100 and height > 20:
            # 큰 영역은 제목이나 긴 텍스트일 가능성
            return "Sample Text"
        elif width > 50 and height > 15:
            # 중간 크기 영역은 버튼이나 입력 필드일 가능성
            return "Button"
        else:
            # 작은 영역은 아이콘이나 작은 텍스트일 가능성
            return ""
            
    except Exception as e:
        print(f"Text estimation error: {e}")
        return ""

def analyze_style_from_region(image, box):
    """
    특정 영역의 스타일을 분석합니다 (색상, 폰트 크기 등).
    """
    try:
        x1, y1, x2, y2 = int(box['x1']), int(box['y1']), int(box['x2']), int(box['y2'])
        
        # 이미지 크기 확인
        height, width = image.shape[:2]
        x1 = max(0, min(x1, width))
        y1 = max(0, min(y1, height))
        x2 = max(0, min(x2, width))
        y2 = max(0, min(y2, height))
        
        if x1 >= x2 or y1 >= y2:
            return {}
        
        # ROI 추출
        roi = image[y1:y2, x1:x2]
        
        if roi.size == 0:
            return {}
        
        # 색상 분석
        colors = analyze_colors(roi)
        
        # 크기 분석
        width_px = x2 - x1
        height_px = y2 - y1
        
        # 폰트 크기 추정 (텍스트 영역의 높이 기반)
        estimated_font_size = max(12, min(48, height_px // 2))
        
        return {
            'colors': colors,
            'width': width_px,
            'height': height_px,
            'font_size': estimated_font_size,
            'background_color': colors.get('dominant', {'r': 1.0, 'g': 1.0, 'b': 1.0, 'a': 1.0}),
            'text_color': colors.get('text', {'r': 0.0, 'g': 0.0, 'b': 0.0, 'a': 1.0})
        }
        
    except Exception as e:
        print(f"Style analysis error: {e}")
        return {}

def analyze_colors(image):
    """
    이미지에서 주요 색상들을 분석합니다.
    """
    try:
        # 이미지를 RGB로 변환
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 이미지를 1차원 배열로 변환
        pixels = rgb_image.reshape(-1, 3)
        
        if SKLEARN_AVAILABLE:
            # K-means 클러스터링으로 주요 색상 추출
            n_colors = min(5, len(pixels))
            if n_colors < 2:
                return get_fallback_colors()
                
            kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
            kmeans.fit(pixels)
            
            # 클러스터 중심점들을 색상으로 변환
            colors = kmeans.cluster_centers_.astype(int)
            
            # 각 색상의 빈도 계산
            labels = kmeans.labels_
            color_counts = np.bincount(labels)
            
            # 빈도순으로 정렬
            sorted_indices = np.argsort(color_counts)[::-1]
            sorted_colors = colors[sorted_indices]
            sorted_counts = color_counts[sorted_indices]
            
            # Figma 색상 형식으로 변환 (0-1 범위)
            figma_colors = []
            for color in sorted_colors:
                r, g, b = color
                figma_color = {
                    'r': r / 255.0,
                    'g': g / 255.0,
                    'b': b / 255.0,
                    'a': 1.0
                }
                figma_colors.append(figma_color)
            
            # 텍스트 색상 추정 (가장 어두운 색상)
            text_color = min(figma_colors, key=lambda c: (c['r'] + c['g'] + c['b']) / 3)
            
            # 배경 색상 추정 (가장 밝은 색상)
            background_color = max(figma_colors, key=lambda c: (c['r'] + c['g'] + c['b']) / 3)
            
            return {
                'dominant': figma_colors[0] if figma_colors else {'r': 1.0, 'g': 1.0, 'b': 1.0, 'a': 1.0},
                'text': text_color,
                'background': background_color,
                'all_colors': figma_colors
            }
        else:
            return get_fallback_colors()
        
    except Exception as e:
        print(f"Color analysis error: {e}")
        return get_fallback_colors()

def get_fallback_colors():
    """
    색상 분석이 실패할 때 사용할 기본 색상들
    """
    return {
        'dominant': {'r': 1.0, 'g': 1.0, 'b': 1.0, 'a': 1.0},
        'text': {'r': 0.0, 'g': 0.0, 'b': 0.0, 'a': 1.0},
        'background': {'r': 1.0, 'g': 1.0, 'b': 1.0, 'a': 1.0},
        'all_colors': []
    }

def detect_component_type(label, parent_label=None):
    """
    Java 코드의 타입 감지 로직을 참고하여 컴포넌트 타입을 결정
    """
    lower_label = label.lower()
    lower_parent = parent_label.lower() if parent_label else ""
    
    # ComboBox/SelectBox 감지
    if ('combobox' in lower_label or 'selectbox' in lower_label or 
        'combobox' in lower_parent or 'selectbox' in lower_parent or
        ('input' in lower_label and has_vector_in_right(lower_label))):
        return 'combobox'
    
    # InputBox 감지
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
    
    # 기본적으로 Button으로 처리
    return 'button'

def has_vector_in_right(label):
    """
    오른쪽에 벡터가 있는지 확인 (Java 코드 참고)
    """
    return 'right' in label and 'vector' in label

def check_if_radio_button(label):
    """
    라디오 버튼인지 확인 (Java 코드 참고)
    """
    return 'radio' in label

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

def yolo_to_figma_node_v4(idx, row, image_w, image_h, image, parent_label=None):
    """
    YOLO 감지 결과(row)를 Figma node(dict)로 변환 (텍스트와 스타일 분석 포함)
    """
    label = row['name']
    figma_type = YOLO_TO_FIGMA_TYPE.get(label, 'RECTANGLE')
    
    # 더 정교한 타입 감지
    detected_component_type = detect_component_type(label, parent_label)
    component_type = COMPONENT_TYPE_MAPPING.get(label, detected_component_type)
    
    box = row['box']
    xmin, ymin, xmax, ymax = box['x1'], box['y1'], box['x2'], box['y2']

    # 텍스트 추출
    extracted_text = extract_text_from_region(image, box)
    
    # 스타일 분석
    style_info = analyze_style_from_region(image, box)

    # 기본 노드 구조
    node = {
        "id": make_figma_id(),
        "name": f"{label}-{idx}",
        "type": figma_type,
        "scrollBehavior": "SCROLLS",
        "blendMode": "PASS_THROUGH",
        "fills": [{
            "blendMode": "NORMAL",
            "type": "SOLID",
            "color": style_info.get('background_color', {
                "r": 1.0, "g": 1.0, "b": 1.0, "a": 1.0
            })
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
        "children": [],
        "overriddenFields": []
    }
    
    # INSTANCE 타입인 경우
    if figma_type == 'INSTANCE':
        component_id = f"1:{4000 + idx}"
        node["componentId"] = component_id
        
        component_properties = {}
        
        if component_type == 'button':
            component_properties["Button name#67:81"] = {
                "value": extracted_text if extracted_text else label,
                "type": "TEXT"
            }
            component_properties["right-Icon#67:215"] = {
                "value": True,
                "type": "BOOLEAN"
            }
        elif component_type == 'inputbox':
            component_properties["Text#2020:7"] = {
                "value": extracted_text if extracted_text else label,
                "type": "TEXT"
            }
            component_properties["SIze"] = {
                "value": "Small",
                "type": "VARIANT",
                "boundVariables": {}
            }
        elif component_type == 'combobox':
            component_properties["Text#2020:7"] = {
                "value": extracted_text if extracted_text else label,
                "type": "TEXT"
            }
            component_properties["State"] = {
                "value": "inactive",
                "type": "VARIANT",
                "boundVariables": {}
            }
        elif component_type == 'radiobutton':
            component_properties["Text#2020:7"] = {
                "value": extracted_text if extracted_text else label,
                "type": "TEXT"
            }
            component_properties["State"] = {
                "value": "inactive",
                "type": "VARIANT",
                "boundVariables": {}
            }
        elif component_type == 'pageindexer':
            component_properties["State"] = {
                "value": "inactive",
                "type": "VARIANT",
                "boundVariables": {}
            }
            component_properties["Type"] = {
                "value": "number",
                "type": "VARIANT",
                "boundVariables": {}
            }
        
        node["componentProperties"] = component_properties
        node["overrides"] = [{
            "id": node["id"],
            "overriddenFields": ["height", "width"]
        }]
        node["layoutSizingHorizontal"] = "FIXED"
        node["layoutSizingVertical"] = "FIXED"
        node["layoutAlign"] = "INHERIT"
        node["layoutGrow"] = 0.0
        node["effects"] = []
        node["interactions"] = []

    # TEXT 노드라면
    if figma_type == 'TEXT':
        node["characters"] = extracted_text if extracted_text else label
        node["styles"] = {
            "fill": "1:3172",
            "text": "1:4305"
        }
        node["characterStyleOverrides"] = []
        
        # 스타일 정보를 기반으로 폰트 설정
        font_size = style_info.get('font_size', 16.0)
        text_color = style_info.get('text_color', {'r': 0.0, 'g': 0.0, 'b': 0.0, 'a': 1.0})
        
        node["style"] = {
            "fontFamily": "Pretendard",
            "fontWeight": 400,
            "fontSize": float(font_size),
            "textAlignHorizontal": "LEFT",
            "textAlignVertical": "CENTER",
            "letterSpacing": 0.0,
            "lineHeightPx": font_size * 1.6,
            "lineHeightPercent": 125.6955795288086,
            "lineHeightPercentFontSize": 150.0,
            "lineHeightUnit": "FONT_SIZE_%",
            "fills": [{
                "type": "SOLID",
                "color": text_color
            }]
        }
        node["overriddenFields"] = ["characters", "text", "textAutoResize"]
        node["layoutVersion"] = 4
        node["effects"] = []
        node["interactions"] = []

    # 일반 노드 (RECTANGLE, GROUP, FRAME 등)의 경우 name에 텍스트 포함
    if figma_type in ['RECTANGLE', 'GROUP', 'FRAME'] and extracted_text:
        # name에 텍스트 정보 추가 (기존 라벨 + 추출된 텍스트)
        original_name = node["name"]
        node["name"] = f"{original_name} ({extracted_text})"

    # FRAME 타입이라면
    if figma_type == 'FRAME':
        node["clipsContent"] = False
        node["background"] = []
        node["backgroundColor"] = style_info.get('background_color', {
            "r": 1.0, "g": 1.0, "b": 1.0, "a": 1.0
        })
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

    # GROUP 타입이라면
    if figma_type == 'GROUP':
        node["layoutSizingHorizontal"] = "FIXED"
        node["layoutSizingVertical"] = "FIXED"
        node["layoutAlign"] = "INHERIT"
        node["layoutGrow"] = 0.0
        node["effects"] = []
        node["interactions"] = []

    # RECTANGLE 타입이라면
    if figma_type == 'RECTANGLE':
        node["rectangleCornerRadii"] = [0.0, 0.0, 0.0, 0.0]
        node["cornerSmoothing"] = 0.0
        node["layoutSizingHorizontal"] = "FIXED"
        node["layoutSizingVertical"] = "FIXED"
        node["layoutAlign"] = "INHERIT"
        node["layoutGrow"] = 0.0
        node["effects"] = []
        node["interactions"] = []

    return node

def build_hierarchical_structure_v4(detected_elements_df, image):
    """
    감지된 요소들을 복잡한 계층 구조로 정리 (텍스트와 스타일 분석 포함)
    """
    nodes = []
    node_id_to_node = {}
    
    for idx, row in detected_elements_df.iterrows():
        node = yolo_to_figma_node_v4(idx, row, 0, 0, image)
        node['original_row'] = row
        nodes.append(node)
        node_id_to_node[node['id']] = node
    
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
    
    # 각 프레임에 포함되는 요소들 찾기
    for frame_node in frame_nodes:
        frame_box = frame_node['original_row']['box']
        frame_label = frame_node['original_row']['name']
        
        contained_groups = []
        remaining_groups = []
        
        for group_node in group_nodes:
            if is_contained(frame_box, group_node['original_row']['box']):
                contained_groups.append(group_node)
            else:
                remaining_groups.append(group_node)
        
        contained_regulars = []
        remaining_regulars = []
        
        for regular_node in regular_nodes:
            if is_contained(frame_box, regular_node['original_row']['box']):
                parent_label = frame_label
                regular_label = regular_node['original_row']['name']
                
                detected_component_type = detect_component_type(regular_label, parent_label)
                component_type = COMPONENT_TYPE_MAPPING.get(regular_label, detected_component_type)
                
                if regular_node['type'] == 'INSTANCE':
                    component_properties = {}
                    
                    # 텍스트 추출
                    extracted_text = extract_text_from_region(image, regular_node['original_row']['box'])
                    
                    if component_type == 'button':
                        component_properties["Button name#67:81"] = {
                            "value": extracted_text if extracted_text else regular_label,
                            "type": "TEXT"
                        }
                        component_properties["right-Icon#67:215"] = {
                            "value": True,
                            "type": "BOOLEAN"
                        }
                    elif component_type == 'inputbox':
                        component_properties["Text#2020:7"] = {
                            "value": extracted_text if extracted_text else regular_label,
                            "type": "TEXT"
                        }
                        component_properties["SIze"] = {
                            "value": "Small",
                            "type": "VARIANT",
                            "boundVariables": {}
                        }
                    elif component_type == 'combobox':
                        component_properties["Text#2020:7"] = {
                            "value": extracted_text if extracted_text else regular_label,
                            "type": "TEXT"
                        }
                        component_properties["State"] = {
                            "value": "inactive",
                            "type": "VARIANT",
                            "boundVariables": {}
                        }
                    elif component_type == 'radiobutton':
                        component_properties["Text#2020:7"] = {
                            "value": extracted_text if extracted_text else regular_label,
                            "type": "TEXT"
                        }
                        component_properties["State"] = {
                            "value": "inactive",
                            "type": "VARIANT",
                            "boundVariables": {}
                        }
                    elif component_type == 'pageindexer':
                        component_properties["State"] = {
                            "value": "inactive",
                            "type": "VARIANT",
                            "boundVariables": {}
                        }
                        component_properties["Type"] = {
                            "value": "number",
                            "type": "VARIANT",
                            "boundVariables": {}
                        }
                    
                    regular_node["componentProperties"] = component_properties
                
                contained_regulars.append(regular_node)
            else:
                remaining_regulars.append(regular_node)
        
        # 각 그룹에 포함되는 일반 요소들 찾기
        for group_node in contained_groups:
            group_box = group_node['original_row']['box']
            group_label = group_node['original_row']['name']
            
            group_contained = []
            group_remaining = []
            
            for regular_node in contained_regulars:
                if is_contained(group_box, regular_node['original_row']['box']):
                    parent_label = group_label
                    regular_label = regular_node['original_row']['name']
                    
                    detected_component_type = detect_component_type(regular_label, parent_label)
                    component_type = COMPONENT_TYPE_MAPPING.get(regular_label, detected_component_type)
                    
                    if regular_node['type'] == 'INSTANCE':
                        component_properties = {}
                        
                        # 텍스트 추출
                        extracted_text = extract_text_from_region(image, regular_node['original_row']['box'])
                        
                        if component_type == 'button':
                            component_properties["Button name#67:81"] = {
                                "value": extracted_text if extracted_text else regular_label,
                                "type": "TEXT"
                            }
                            component_properties["right-Icon#67:215"] = {
                                "value": True,
                                "type": "BOOLEAN"
                            }
                        elif component_type == 'inputbox':
                            component_properties["Text#2020:7"] = {
                                "value": extracted_text if extracted_text else regular_label,
                                "type": "TEXT"
                            }
                            component_properties["SIze"] = {
                                "value": "Small",
                                "type": "VARIANT",
                                "boundVariables": {}
                            }
                        elif component_type == 'combobox':
                            component_properties["Text#2020:7"] = {
                                "value": extracted_text if extracted_text else regular_label,
                                "type": "TEXT"
                            }
                            component_properties["State"] = {
                                "value": "inactive",
                                "type": "VARIANT",
                                "boundVariables": {}
                            }
                        elif component_type == 'radiobutton':
                            component_properties["Text#2020:7"] = {
                                "value": extracted_text if extracted_text else regular_label,
                                "type": "TEXT"
                            }
                            component_properties["State"] = {
                                "value": "inactive",
                                "type": "VARIANT",
                                "boundVariables": {}
                            }
                        elif component_type == 'pageindexer':
                            component_properties["State"] = {
                                "value": "inactive",
                                "type": "VARIANT",
                                "boundVariables": {}
                            }
                            component_properties["Type"] = {
                                "value": "number",
                                "type": "VARIANT",
                                "boundVariables": {}
                            }
                        
                        regular_node["componentProperties"] = component_properties
                    
                    group_contained.append(regular_node)
                else:
                    group_remaining.append(regular_node)
            
            group_node['children'] = group_contained
            contained_regulars = group_remaining
        
        frame_node['children'] = contained_groups + contained_regulars
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
    
    final_nodes = frame_nodes + group_nodes + regular_nodes
    
    # original_row 제거
    def clean_node(node):
        if 'original_row' in node:
            del node['original_row']
        for child in node.get('children', []):
            clean_node(child)
    
    for node in final_nodes:
        clean_node(node)
    
    return final_nodes

def yolo_results_to_figma_json_v4(image_path, results):
    """
    YOLO 감지 결과를 Figma REST API 스타일 JSON으로 변환 (텍스트와 스타일 분석 포함)
    """
    img = cv2.imread(image_path)
    image_h, image_w = img.shape[:2]
    detected_elements_df = results[0].to_df()
    
    # 계층 구조로 정리된 children 생성
    children = build_hierarchical_structure_v4(detected_elements_df, img)
    
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

def save_detection_result_with_image_v4(image_path, results, output_dir):
    """
    감지 결과를 JSON과 PNG로 저장 (텍스트와 스타일 분석 포함)
    """
    # JSON 저장
    figma_json = yolo_results_to_figma_json_v4(image_path, results)
    json_filename = f'{Path(image_path).stem}_figma.json'
    json_path = os.path.join(output_dir, json_filename)
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(figma_json, f, indent=2, ensure_ascii=False)
    
    # PNG 저장 (텍스트와 스타일 정보 포함)
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Error: Failed to load image for PNG generation")
        return json_path, None
    
    annotated_img = img.copy()
    detected_elements_df = results[0].to_df()
    
    for idx, row in detected_elements_df.iterrows():
        box = row['box']
        x1, y1, x2, y2 = int(box['x1']), int(box['y1']), int(box['x2']), int(box['y2'])
        label = row['name']
        confidence = row['confidence']
        
        # 텍스트 추출
        extracted_text = extract_text_from_region(img, box)
        
        # 타입 결정
        figma_type = YOLO_TO_FIGMA_TYPE.get(label, 'RECTANGLE')
        
        # 색상 결정
        if figma_type == 'INSTANCE':
            color = (0, 255, 0)  # 초록색
        elif figma_type == 'TEXT':
            color = (255, 0, 0)  # 빨간색
        elif figma_type == 'FRAME':
            color = (0, 0, 255)  # 파란색
        elif figma_type == 'GROUP':
            color = (255, 255, 0)  # 노란색
        else:
            color = (128, 128, 128)  # 회색
        
        # 박스 그리기
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
        
        # 라벨 텍스트 준비 (텍스트 정보 포함)
        label_text = f"{label} ({figma_type})"
        if extracted_text:
            label_text += f" - '{extracted_text[:20]}...'" if len(extracted_text) > 20 else f" - '{extracted_text}'"
        confidence_text = f"{confidence:.2f}"
        
        # 라벨 배경 크기 계산
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        
        # 라벨 배경 그리기
        (label_width, label_height), _ = cv2.getTextSize(label_text, font, font_scale, thickness)
        cv2.rectangle(annotated_img, (x1, y1 - label_height - 10), (x1 + label_width + 10, y1), color, -1)
        
        # 라벨 텍스트 그리기
        cv2.putText(annotated_img, label_text, (x1 + 5, y1 - 5), font, font_scale, (255, 255, 255), thickness)
        
        # Confidence 텍스트 그리기
        (conf_width, conf_height), _ = cv2.getTextSize(confidence_text, font, font_scale, thickness)
        cv2.rectangle(annotated_img, (x2 - conf_width - 10, y1 - conf_height - 10), (x2, y1), (0, 0, 0), -1)
        cv2.putText(annotated_img, confidence_text, (x2 - conf_width - 5, y1 - 5), font, font_scale, (255, 255, 255), thickness)
    
    # 이미지 크기 조정
    height, width = annotated_img.shape[:2]
    max_size = 1200
    
    if max(height, width) > max_size:
        scale = max_size / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        annotated_img = cv2.resize(annotated_img, (new_width, new_height))
    
    # PNG 저장
    png_filename = f'{Path(image_path).stem}_detection.png'
    png_path = os.path.join(output_dir, png_filename)
    
    annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
    
    try:
        from PIL import Image
        pil_image = Image.fromarray(annotated_img_rgb)
        pil_image.save(png_path, 'PNG', quality=95)
    except ImportError:
        cv2.imwrite(png_path, annotated_img)
    
    return json_path, png_path

def main():
    print("🚀 YOLO to Figma JSON Converter v4 (텍스트 감지 + 스타일 분석)")
    print("=" * 60)
    
    # 라이브러리 상태 출력
    if TESSERACT_AVAILABLE:
        print("✅ Tesseract OCR available")
    else:
        print("⚠️  Tesseract OCR not available. Using fallback text detection.")
    
    if SKLEARN_AVAILABLE:
        print("✅ Scikit-learn available for color analysis")
    else:
        print("⚠️  Scikit-learn not available. Using fallback color analysis.")
    
    today = datetime.datetime.now().strftime('%Y-%m-%d')
    output_dir = f'./figma_json(style)/{today}'
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
                    json_path, png_path = save_detection_result_with_image_v4(image_path, results, output_dir)
                    print(f"💾 JSON saved to: {json_path}")
                    print(f"🖼️  PNG saved to: {png_path}")
                    processed_count += 1
                except Exception as e:
                    print(f"❌ Error processing {image_path}: {str(e)}")
    
    print(f"\n✅ Processing complete! {processed_count} files processed.")
    print(f"📁 Output directory: {output_dir}")

if __name__ == "__main__":
    main() 