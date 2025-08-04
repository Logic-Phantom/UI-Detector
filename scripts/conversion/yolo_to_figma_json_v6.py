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
DATASET_PATH = '../../yolo/datasets/screenshots/start'
IMAGES_PATH = os.path.join(DATASET_PATH, 'images')
LABELS_PATH = os.path.join(DATASET_PATH, 'labels')
CLASSES_PATH = os.path.join(LABELS_PATH, 'classes.txt')

# 클래스명 로드
try:
    with open(CLASSES_PATH, 'r', encoding='utf-8') as f:
        CLASS_NAMES = [line.strip() for line in f if line.strip()]
except FileNotFoundError:
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

# YOLO 클래스명 → Figma type 매핑 (Java 코드 기반으로 수정)
YOLO_TO_FIGMA_TYPE = {
    'Button': 'INSTANCE',
    'InputBox': 'INSTANCE',
    'TextArea': 'TEXT',
    'AppHeader': 'FRAME',  # Java 코드: title이 포함된 name은 AppHeader로 변환
    'ComboBox': 'INSTANCE',
    'CheckBox': 'RECTANGLE',
    'CheckBoxGroup': 'GROUP',
    'RadioButton': 'INSTANCE',
    'DateInput': 'INSTANCE',
    'Output': 'TEXT',
    'Group': 'GROUP',
    'Grid': 'FRAME',  # Java 코드: table이 포함된 name은 Grid로 변환
    'GridTitle': 'FRAME',  # Java 코드: FRAME 타입으로 변경
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

# 컴포넌트 타입 매핑
COMPONENT_TYPE_MAPPING = {
    'Button': 'button',
    'InputBox': 'inputbox',
    'ComboBox': 'combobox',
    'RadioButton': 'radiobutton',
    'Pagination': 'pageindexer',
    'SelectBox': 'combobox',
}

# FRAME 타입으로 처리할 컨트롤들 (Java 코드 기반)
FRAME_CONTROLS = ['AppHeader', 'Grid', 'GridTitle', 'Frame', 'Container', 'Panel', 'Section', 'Header', 'Footer', 'Sidebar', 'MainContent', 'Navigation', 'SearchArea', 'FilterArea', 'ResultArea', 'FormArea']

def detect_frame_type_by_name(name):
    """Java 코드 기반으로 name을 통해 Grid 또는 AppHeader 감지"""
    name_lower = name.lower()
    if "table" in name_lower:
        return "Grid"
    elif "title" in name_lower:
        return "AppHeader"
    return None

def extract_text_from_region(image, box):
    """이미지 영역에서 텍스트 추출 (v4의 고급 OCR 로직 적용)"""
    if not TESSERACT_AVAILABLE:
        return None
    
    try:
        x1, y1, x2, y2 = int(box['x1']), int(box['y1']), int(box['x2']), int(box['y2'])
        
        # 이미지 크기 확인
        if x2 <= x1 or y2 <= y1:
            return None
        
        # 이미지 자르기
        region = image[y1:y2, x1:x2]
        if region.size == 0:
            return None
        
        # 그레이스케일 변환
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        
        # 노이즈 제거
        denoised = cv2.fastNlMeansDenoising(gray)
        
        # 이미지 스케일링 (OCR 정확도 향상)
        scale_factor = 2
        scaled = cv2.resize(denoised, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
        
        # 다양한 임계값 방법 시도
        text_results = []
        
        # 1. 적응형 임계값
        adaptive_thresh = cv2.adaptiveThreshold(scaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        text = pytesseract.image_to_string(adaptive_thresh, lang='kor+eng', config='--psm 6')
        if text.strip():
            text_results.append(text.strip())
        
        # 2. Otsu 임계값
        _, otsu_thresh = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        text = pytesseract.image_to_string(otsu_thresh, lang='kor+eng', config='--psm 6')
        if text.strip():
            text_results.append(text.strip())
        
        # 3. 가우시안 블러 후 임계값
        blurred = cv2.GaussianBlur(scaled, (5, 5), 0)
        _, gaussian_thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        text = pytesseract.image_to_string(gaussian_thresh, lang='kor+eng', config='--psm 6')
        if text.strip():
            text_results.append(text.strip())
        
        # 4. 모폴로지 연산 적용
        kernel = np.ones((2, 2), np.uint8)
        morphed = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel)
        text = pytesseract.image_to_string(morphed, lang='kor+eng', config='--psm 6')
        if text.strip():
            text_results.append(text.strip())
        
        # 가장 긴 텍스트 결과 반환
        if text_results:
            return max(text_results, key=len)
        
        return None
        
    except Exception as e:
        print(f"⚠️  OCR 오류: {e}")
        return None

def estimate_text_from_label(box, image):
    """라벨을 기반으로 텍스트 추정"""
    label = box.get('name', '')
    
    # 일반적인 UI 텍스트 매핑
    text_mapping = {
        'Button': '버튼',
        'InputBox': '텍스트 입력',
        'TextArea': '텍스트 영역',
        'ComboBox': '선택',
        'CheckBox': '체크',
        'RadioButton': '선택',
        'DateInput': '날짜',
        'Output': '결과',
        'FormTitle': '제목',
        'GridTitle': '제목',
        'AppHeader': '헤더'
    }
    
    return text_mapping.get(label, label)

def analyze_style_from_region(image, box):
    """이미지 영역에서 스타일 정보 분석 (v4의 고급 분석 로직 적용)"""
    try:
        x1, y1, x2, y2 = int(box['x1']), int(box['y1']), int(box['x2']), int(box['y2'])
        
        if x2 <= x1 or y2 <= y1:
            return get_fallback_colors()
        
        region = image[y1:y2, x1:x2]
        if region.size == 0:
            return get_fallback_colors()
        
        # 색상 분석
        colors = analyze_colors(region)
        
        # 폰트 크기 추정 (높이 기반)
        height = y2 - y1
        font_size = max(12.0, min(height * 0.6, 48.0))  # 12-48px 범위
        
        # 텍스트 색상 (대비 기반)
        if colors['dominant_color']:
            # 대비 계산을 위한 밝기 추정
            brightness = sum(colors['dominant_color'][:3]) / 3
            if brightness > 0.5:
                text_color = {'r': 0.0, 'g': 0.0, 'b': 0.0, 'a': 1.0}  # 검은색
            else:
                text_color = {'r': 1.0, 'g': 1.0, 'b': 1.0, 'a': 1.0}  # 흰색
        else:
            text_color = {'r': 0.11372549086809158, 'g': 0.11372549086809158, 'b': 0.11372549086809158, 'a': 1.0}
        
        return {
            'background_color': colors['dominant_color'] or {'r': 1.0, 'g': 1.0, 'b': 1.0, 'a': 1.0},
            'font_size': font_size,
            'text_color': text_color,
            'color_palette': colors['palette']
        }
        
    except Exception as e:
        print(f"⚠️  스타일 분석 오류: {e}")
        return get_fallback_colors()

def analyze_colors(image):
    """이미지에서 색상 분석 (v4의 K-means 클러스터링 적용)"""
    if not SKLEARN_AVAILABLE:
        return {'dominant_color': None, 'palette': []}
    
    try:
        # 이미지를 RGB로 변환
        if len(image.shape) == 3:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            return {'dominant_color': None, 'palette': []}
        
        # 픽셀 데이터 재구성
        pixels = rgb_image.reshape(-1, 3)
        
        # K-means 클러스터링으로 주요 색상 추출
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        # 클러스터 중심을 색상으로 변환
        colors = kmeans.cluster_centers_
        
        # 각 색상의 빈도 계산
        labels = kmeans.labels_
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        # 빈도순으로 정렬
        color_counts = list(zip(colors, counts))
        color_counts.sort(key=lambda x: x[1], reverse=True)
        
        # 주요 색상 (가장 빈도가 높은 색상)
        dominant_color = color_counts[0][0] if color_counts else None
        
        # 색상 팔레트 생성 (RGB를 0-1 범위로 정규화)
        palette = []
        for color, count in color_counts:
            normalized_color = {
                'r': float(color[0] / 255),
                'g': float(color[1] / 255),
                'b': float(color[2] / 255),
                'a': 1.0
            }
            palette.append({
                'color': normalized_color,
                'frequency': int(count)
            })
        
        # 주요 색상도 정규화
        if dominant_color is not None:
            dominant_color = {
                'r': float(dominant_color[0] / 255),
                'g': float(dominant_color[1] / 255),
                'b': float(dominant_color[2] / 255),
                'a': 1.0
            }
        
        return {
            'dominant_color': dominant_color,
            'palette': palette
        }
        
    except Exception as e:
        print(f"⚠️  색상 분석 오류: {e}")
        return {'dominant_color': None, 'palette': []}

def get_fallback_colors():
    """기본 색상 반환"""
    return {
        'background_color': {'r': 1.0, 'g': 1.0, 'b': 1.0, 'a': 1.0},
        'font_size': 16.0,
        'text_color': {'r': 0.11372549086809158, 'g': 0.11372549086809158, 'b': 0.11372549086809158, 'a': 1.0},
        'color_palette': []
    }

def detect_component_type(label, parent_label=None):
    """컴포넌트 타입 감지 (v4의 고급 로직 적용)"""
    label_lower = label.lower()
    
    # 버튼 감지
    if 'button' in label_lower or 'btn' in label_lower:
        return 'button'
    
    # 입력 박스 감지
    if 'input' in label_lower or 'textbox' in label_lower or 'field' in label_lower:
        return 'inputbox'
    
    # 콤보박스 감지
    if 'combo' in label_lower or 'select' in label_lower or 'dropdown' in label_lower:
        return 'combobox'
    
    # 라디오 버튼 감지
    if 'radio' in label_lower or has_vector_in_right(label):
        return 'radiobutton'
    
    # 체크박스 감지
    if 'check' in label_lower or 'checkbox' in label_lower:
        return 'checkbox'
    
    # 페이지네이션 감지
    if 'page' in label_lower or 'pagination' in label_lower:
        return 'pageindexer'
    
    # 부모 라벨 기반 감지
    if parent_label:
        parent_lower = parent_label.lower()
        if 'form' in parent_lower and 'input' in label_lower:
            return 'inputbox'
        if 'navigation' in parent_lower and 'button' in label_lower:
            return 'button'
    
    return 'rectangle'  # 기본값

def has_vector_in_right(label):
    """라벨에 오른쪽 화살표가 있는지 확인"""
    return '>' in label or '→' in label or 'arrow' in label.lower()

def check_if_radio_button(label):
    """라디오 버튼인지 확인"""
    return 'radio' in label.lower() or 'option' in label.lower()

def make_figma_id():
    """Figma ID 생성"""
    return f"1:{uuid.uuid4().int % 1000000}"

def is_contained(box1, box2):
    """box1이 box2 안에 포함되어 있는지 확인"""
    x1_1, y1_1, x2_1, y2_1 = box1['x1'], box1['y1'], box1['x2'], box1['y2']
    x1_2, y1_2, x2_2, y2_2 = box2['x1'], box2['y1'], box2['x2'], box2['y2']
    
    return (x1_1 <= x1_2 and y1_1 <= y1_2 and x2_1 >= x2_2 and y2_1 >= y2_2) 

def yolo_to_figma_node_v6(idx, row, image_w, image_h, image, parent_label=None):
    """
    YOLO 감지 결과(row)를 Figma node(dict)로 변환 (Figma API v6 - 모든 속성 포함)
    """
    label = row['name']
    figma_type = YOLO_TO_FIGMA_TYPE.get(label, 'RECTANGLE')
    
    # Java 코드 기반 name 감지
    frame_type = detect_frame_type_by_name(label)
    if frame_type:
        label = frame_type
        figma_type = 'FRAME'
    
    # 더 정교한 타입 감지
    detected_component_type = detect_component_type(label, parent_label)
    component_type = COMPONENT_TYPE_MAPPING.get(label, detected_component_type)
    
    box = row['box']
    xmin, ymin, xmax, ymax = box['x1'], box['y1'], box['x2'], box['y2']

    # 텍스트 추출
    extracted_text = extract_text_from_region(image, box)
    
    # 스타일 분석
    style_info = analyze_style_from_region(image, box)

    # 기본 노드 구조 (Figma API v6 - 모든 필수 속성 포함)
    node = {
        "id": make_figma_id(),
        "name": f"{label}-{idx}",
        "type": figma_type,
        "visible": True,  # v6 추가: 가시성
        "locked": False,  # v6 추가: 잠금 상태
        "opacity": 1.0,  # v6 추가: 투명도
        "rotation": 0.0,  # v6 추가: 회전
        "blendMode": "PASS_THROUGH",
        "preserveRatio": False,  # v6 추가: 비율 유지
        "layoutAlign": "INHERIT",  # v6 추가: 레이아웃 정렬
        "layoutGrow": 0.0,  # v6 추가: 레이아웃 성장
        "constraints": {
            "vertical": "TOP",
            "horizontal": "LEFT"
        },
        "transitionNodeID": None,  # v6 추가: 전환 노드 ID
        "transitionDuration": None,  # v6 추가: 전환 지속시간
        "transitionEasing": None,  # v6 추가: 전환 이징
        "reactions": [],  # v6 추가: 반응
        "children": [],
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
        "size": f"{int(xmax - xmin)}x{int(ymax - ymin)}",  # v6 추가: 크기 문자열
        "relativeTransform": [[1, 0, float(xmin)], [0, 1, float(ymin)]],  # v6 추가: 상대 변환
        "clipsContent": False,  # v6 추가: 내용 클리핑
        "layoutMode": "NONE",  # v6 추가: 레이아웃 모드
        "counterAxisSizingMode": "FIXED",  # v6 추가: 반대축 크기 모드
        "primaryAxisSizingMode": "FIXED",  # v6 추가: 주축 크기 모드
        "counterAxisAlignItems": "MIN",  # v6 추가: 반대축 정렬
        "primaryAxisAlignItems": "MIN",  # v6 추가: 주축 정렬
        "paddingLeft": 0.0,  # v6 추가: 왼쪽 패딩
        "paddingRight": 0.0,  # v6 추가: 오른쪽 패딩
        "paddingTop": 0.0,  # v6 추가: 위쪽 패딩
        "paddingBottom": 0.0,  # v6 추가: 아래쪽 패딩
        "itemSpacing": 0.0,  # v6 추가: 아이템 간격
        "layoutWrap": "NO_WRAP",  # v6 추가: 레이아웃 래핑
        "fills": [{
            "blendMode": "NORMAL",
            "type": "SOLID",
            "color": style_info.get('background_color', {
                "r": 1.0, "g": 1.0, "b": 1.0, "a": 1.0
            })
        }] if figma_type in ["RECTANGLE", "FRAME", "GROUP"] else [],
        "strokes": [],  # v6 추가: 선 배열
        "strokeWeight": 1.0,
        "strokeAlign": "OUTSIDE",  # v6 추가: 선 정렬
        "strokeCap": "NONE",  # v6 추가: 선 끝 모양
        "strokeJoin": "MITER",  # v6 추가: 선 연결 모양
        "dashPattern": [],  # v6 추가: 점선 패턴
        "effects": [],
        "cornerRadius": 0.0,  # v6 추가: 모서리 반경
        "cornerSmoothing": 0.0,  # v6 추가: 모서리 스무딩
        "exportSettings": [],  # v6 추가: 내보내기 설정
        "overriddenFields": []
    }
    
    # INSTANCE 타입인 경우 (Figma API v6 구조)
    if figma_type == 'INSTANCE':
        component_id = f"1:{4000 + idx}"
        node["componentId"] = component_id
        
        component_property_references = {}
        
        if component_type == 'button':
            component_property_references["Button name#67:81"] = {
                "value": extracted_text if extracted_text else "버튼",
                "type": "TEXT"
            }
            component_property_references["right-Icon#67:215"] = {
                "value": True,
                "type": "BOOLEAN"
            }
        elif component_type == 'inputbox':
            component_property_references["Text#2020:7"] = {
                "value": extracted_text if extracted_text else "텍스트 입력",
                "type": "TEXT"
            }
            component_property_references["SIze"] = {
                "value": "Small",
                "type": "VARIANT",
                "boundVariables": {}
            }
        elif component_type == 'combobox':
            component_property_references["Text#2020:7"] = {
                "value": extracted_text if extracted_text else "텍스트 입력",
                "type": "TEXT"
            }
            component_property_references["State"] = {
                "value": "inactive",
                "type": "VARIANT",
                "boundVariables": {}
            }
        elif component_type == 'radiobutton':
            component_property_references["Text#2020:7"] = {
                "value": extracted_text if extracted_text else "텍스트 입력",
                "type": "TEXT"
            }
            component_property_references["State"] = {
                "value": "inactive",
                "type": "VARIANT",
                "boundVariables": {}
            }
        elif component_type == 'pageindexer':
            component_property_references["State"] = {
                "value": "inactive",
                "type": "VARIANT",
                "boundVariables": {}
            }
            component_property_references["Type"] = {
                "value": "number",
                "type": "VARIANT",
                "boundVariables": {}
            }
        
        node["componentPropertyReferences"] = component_property_references
        node["componentSetId"] = None  # v6 추가: 컴포넌트 세트 ID
        node["overrides"] = [{
            "id": node["id"],
            "overriddenFields": ["height", "width"]
        }]
        node["layoutSizingHorizontal"] = "FIXED"
        node["layoutSizingVertical"] = "FIXED"
        node["interactions"] = []

    # TEXT 노드라면 (Figma API v6 구조)
    if figma_type == 'TEXT':
        node["characters"] = extracted_text if extracted_text else label
        node["characterStyleOverrides"] = []
        node["styleOverrideTable"] = {}
        node["lineTypes"] = ["NONE"]
        node["lineIndentations"] = [0]  # v6 추가: 줄 들여쓰기
        
        # 스타일 정보를 기반으로 폰트 설정
        font_size = style_info.get('font_size', 16.0)
        text_color = style_info.get('text_color', {'r': 0.11372549086809158, 'g': 0.11372549086809158, 'b': 0.11372549086809158, 'a': 1.0})
        
        # fills 배열에 텍스트 색상 추가
        node["fills"] = [{
            "blendMode": "NORMAL",
            "type": "SOLID",
            "color": text_color
        }]
        
        # 텍스트 스타일 설정 (v6 확장)
        node["style"] = {
            "fontFamily": "Pretendard",
            "fontWeight": 400,
            "fontSize": float(font_size),
            "textAlignHorizontal": "LEFT",
            "textAlignVertical": "TOP",
            "letterSpacing": 0.0,
            "lineHeightPx": font_size * 1.6,
            "lineHeightPercent": 125.6955795288086,
            "lineHeightPercentFontSize": 150.0,
            "lineHeightUnit": "FONT_SIZE_%",
            "textCase": "ORIGINAL",  # v6 추가: 텍스트 대소문자
            "textDecoration": "NONE",  # v6 추가: 텍스트 장식
            "paragraphIndent": 0.0,  # v6 추가: 단락 들여쓰기
            "paragraphSpacing": 0.0,  # v6 추가: 단락 간격
            "autoRename": True  # v6 추가: 자동 이름 변경
        }
        
        node["overriddenFields"] = ["characters", "characterStyleOverrides", "inheritFillStyleId", "inheritTextStyleId", "layoutGrow", "lineIndentations", "lineTypes", "styleOverrideTable", "text", "textAutoResize"]
        node["layoutVersion"] = 4
        node["effects"] = []
        node["interactions"] = []

    # 일반 노드의 경우 name에 텍스트 포함
    if figma_type in ['RECTANGLE', 'GROUP', 'FRAME'] and extracted_text:
        original_name = node["name"]
        node["name"] = f"{original_name} ({extracted_text})"

    # FRAME 타입이라면 (Figma API v6 구조)
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
        node["cornerRadius"] = 0.0
        node["cornerSmoothing"] = 0.0

    # GROUP 타입이라면 (Figma API v6 구조)
    if figma_type == 'GROUP':
        node["layoutSizingHorizontal"] = "FIXED"
        node["layoutSizingVertical"] = "FIXED"
        node["layoutAlign"] = "INHERIT"
        node["layoutGrow"] = 0.0
        node["effects"] = []
        node["interactions"] = []

    # RECTANGLE 타입이라면 (Figma API v6 구조)
    if figma_type == 'RECTANGLE':
        node["rectangleCornerRadii"] = [0.0, 0.0, 0.0, 0.0]  # v6 추가: 사각형 모서리 반지름
        node["cornerSmoothing"] = 0.0
        node["layoutSizingHorizontal"] = "FIXED"
        node["layoutSizingVertical"] = "FIXED"
        node["layoutAlign"] = "INHERIT"
        node["layoutGrow"] = 0.0
        node["effects"] = []
        node["interactions"] = []

    return node

def build_hierarchical_structure_v6(detected_elements_df, image):
    """감지된 요소들을 Figma API v6 계층 구조로 정리"""
    nodes = []
    
    for idx, row in detected_elements_df.iterrows():
        node = yolo_to_figma_node_v6(idx, row, 0, 0, image)
        node['original_row'] = row
        nodes.append(node)
    
    # AppHeader 처리: 최상단에만 1개 유지 (Java 코드 기반)
    app_headers = [node for node in nodes if node['original_row']['name'] == 'AppHeader']
    if len(app_headers) > 1:
        # 가장 위쪽에 있는 AppHeader만 유지
        app_headers.sort(key=lambda x: x['original_row']['box']['y1'])
        keep_header = app_headers[0]
        nodes = [node for node in nodes if node['original_row']['name'] != 'AppHeader']
        nodes.append(keep_header)
        print(f"✅ AppHeader (FRAME): 최상단 1개만 유지 (총 {len(app_headers)}개 중)")
    elif len(app_headers) == 1:
        print(f"✅ AppHeader (FRAME): 1개 감지됨")
    
    # Grid와 GridTitle 처리 (Java 코드 기반)
    grids = [node for node in nodes if node['original_row']['name'] == 'Grid']
    grid_titles = [node for node in nodes if node['original_row']['name'] == 'GridTitle']
    
    print(f"📊 감지된 Grid (FRAME): {len(grids)}개, GridTitle (FRAME): {len(grid_titles)}개")
    
    # GridTitle을 각 Grid 위에 배치
    for i, grid in enumerate(grids):
        grid_box = grid['original_row']['box']
        
        # 해당 Grid 위에 있는 GridTitle 찾기
        matching_title = None
        for title in grid_titles:
            title_box = title['original_row']['box']
            # Grid 바로 위에 있는 GridTitle 찾기
            if (title_box['y2'] <= grid_box['y1'] and 
                abs(title_box['x1'] - grid_box['x1']) < 50 and 
                abs(title_box['x2'] - grid_box['x2']) < 50):
                matching_title = title
                break
        
        if matching_title:
            # GridTitle을 Grid의 자식으로 설정
            grid['children'].append(matching_title)
            nodes.remove(matching_title)
            print(f"✅ GridTitle을 Grid {i+1}의 자식으로 설정")
    
    # Grid 내부의 객체들 필터링
    filtered_nodes = []
    for node in nodes:
        node_box = node['original_row']['box']
        is_inside_grid = False
        
        for grid in grids:
            grid_box = grid['original_row']['box']
            if is_contained(node_box, grid_box) and node['original_row']['name'] != 'GridTitle':
                is_inside_grid = True
                break
        
        if not is_inside_grid:
            filtered_nodes.append(node)
        else:
            print(f"🚫 Grid 내부 객체 필터링: {node['original_row']['name']}")
    
    print(f"📋 최종 노드 수: {len(filtered_nodes)}개 (Grid 내부 객체 제외)")
    return filtered_nodes

def clean_node(node):
    """노드에서 original_row 제거"""
    if 'original_row' in node:
        del node['original_row']
    
    # 자식 노드들도 정리
    for child in node.get('children', []):
        clean_node(child)
    
    return node

def yolo_results_to_figma_json_v6(image_path, results):
    """YOLO 결과를 Figma JSON으로 변환 (v6)"""
    # 이미지 로드
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ 이미지를 로드할 수 없습니다: {image_path}")
        return None
    
    # 결과를 DataFrame으로 변환
    detected_elements = []
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())
                
                if cls < len(CLASS_NAMES):
                    label = CLASS_NAMES[cls]
                    detected_elements.append({
                        'name': label,
                        'confidence': conf,
                        'box': {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
                    })
    
    if not detected_elements:
        print("❌ 감지된 요소가 없습니다.")
        return None
    
    # DataFrame 생성
    import pandas as pd
    detected_elements_df = pd.DataFrame(detected_elements)
    
    # 계층 구조 생성
    nodes = build_hierarchical_structure_v6(detected_elements_df, image)
    
    # Figma JSON 구조 생성 (v6)
    figma_json = {
        "document": {
            "id": "1:1",
            "name": "Document",
            "type": "DOCUMENT",
            "visible": True,
            "children": [{
                "id": "1:2",
                "name": "Canvas",
                "type": "CANVAS",
                "visible": True,
                "backgroundColor": {
                    "r": 1.0, "g": 1.0, "b": 1.0, "a": 1.0
                },
                "children": []
            }]
        },
        "components": {},
        "componentSets": {},
        "schemaVersion": 0,
        "styles": {},
        "name": "UI Detection Result"
    }
    
    # 노드들을 Canvas의 자식으로 추가
    for node in nodes:
        clean_node(node)
        figma_json["document"]["children"][0]["children"].append(node)
    
    return figma_json

def save_detection_result_with_image_v6(image_path, results, output_dir):
    """감지 결과를 이미지와 함께 저장 (v6)"""
    # 이미지 로드
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ 이미지를 로드할 수 없습니다: {image_path}")
        return None
    
    # 결과를 DataFrame으로 변환
    detected_elements = []
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())
                
                if cls < len(CLASS_NAMES):
                    label = CLASS_NAMES[cls]
                    detected_elements.append({
                        'name': label,
                        'confidence': conf,
                        'box': {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
                    })
    
    if not detected_elements:
        print("❌ 감지된 요소가 없습니다.")
        return None
    
    # DataFrame 생성
    import pandas as pd
    detected_elements_df = pd.DataFrame(detected_elements)
    
    # 계층 구조 생성
    nodes = build_hierarchical_structure_v6(detected_elements_df, image)
    
    # Figma JSON 구조 생성 (v6)
    figma_json = {
        "document": {
            "id": "1:1",
            "name": "Document",
            "type": "DOCUMENT",
            "visible": True,
            "children": [{
                "id": "1:2",
                "name": "Canvas",
                "type": "CANVAS",
                "visible": True,
                "backgroundColor": {
                    "r": 1.0, "g": 1.0, "b": 1.0, "a": 1.0
                },
                "children": []
            }]
        },
        "components": {},
        "componentSets": {},
        "schemaVersion": 0,
        "styles": {},
        "name": "UI Detection Result"
    }
    
    # 노드들을 Canvas의 자식으로 추가
    for node in nodes:
        clean_node(node)
        figma_json["document"]["children"][0]["children"].append(node)
    
    # 결과 저장
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d")
    output_path = os.path.join(output_dir, timestamp)
    os.makedirs(output_path, exist_ok=True)
    
    # JSON 파일 저장
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    json_filename = f"{image_name}_figma.json"
    json_path = os.path.join(output_path, json_filename)
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(figma_json, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Figma JSON 저장됨: {json_path}")
    
    # 감지 결과 이미지 생성
    img_with_boxes = image.copy()
    
    for element in detected_elements:
        box = element['box']
        x1, y1, x2, y2 = int(box['x1']), int(box['y1']), int(box['x2']), int(box['y2'])
        label = element['name']
        conf = element['confidence']
        
        # 박스 그리기
        cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 라벨 텍스트
        text = f"{label} ({conf:.2f})"
        cv2.putText(img_with_boxes, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # 이미지 저장
    image_filename = f"{image_name}_detection.png"
    image_path_output = os.path.join(output_path, image_filename)
    cv2.imwrite(image_path_output, img_with_boxes)
    
    print(f"✅ 감지 결과 이미지 저장됨: {image_path_output}")
    
    return json_path

def main():
    """메인 함수 (Figma API v6)"""
    print("🚀 YOLO to Figma JSON Converter v6 (Figma API 완전 지원)")
    print("📋 Figma API v6의 모든 속성을 포함한 완전한 JSON 생성")
    
    # 이미지 파일들 찾기
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        image_files.extend(Path(IMAGES_PATH).glob(ext))
    
    if not image_files:
        print(f"❌ {IMAGES_PATH}에서 이미지 파일을 찾을 수 없습니다.")
        return
    
    print(f"📸 발견된 이미지: {len(image_files)}개")
    
    # 출력 디렉토리 설정
    output_dir = "../../figma_json(style)"
    os.makedirs(output_dir, exist_ok=True)
    
    # 각 이미지 처리
    for image_file in image_files:
        print(f"\n🔍 처리 중: {image_file.name}")
        
        # YOLO 감지 실행
        results = model(str(image_file))
        
        # 결과 저장
        json_path = save_detection_result_with_image_v6(str(image_file), results, output_dir)
        
        if json_path:
            print(f"✅ 완료: {image_file.name} -> {json_path}")
        else:
            print(f"❌ 실패: {image_file.name}")
    
    print(f"\n🎉 모든 이미지 처리 완료! 결과는 {output_dir}에 저장되었습니다.")

if __name__ == "__main__":
    main() 