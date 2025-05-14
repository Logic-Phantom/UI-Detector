import json
import cv2

from ultralytics import YOLO
import os

# YOLOv5 모델 로드
model_path = '../runs/detect/train9/weights/best.pt'
if os.path.exists(model_path):
    print(f"Model file found at: {model_path}")
    model = YOLO(model_path)
else:
    print(f"Model file not found at: {model_path}")

def detect_ui_elements(image_path):
    # 이미지 로드 및 리사이즈
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, (640, 640))  # YOLO 입력 크기에 맞게 리사이즈
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)  # BGR -> RGB 변환

    # YOLOv5 모델로 이미지 분석
    results = model(img_rgb)

    # 결과에서 pandas 형식으로 변환 후 첫 번째 객체에 접근
    detected_elements_df = results[0].to_df()  # pandas DataFrame 형식으로 변환

    # 디버깅용: 감지된 객체 출력 (데이터프레임)
    print("Detected Elements:")
    print(detected_elements_df)

    # 추가된 로그: 감지된 객체 리스트
    if len(detected_elements_df) == 0:
        print("No objects detected.")
    else:
        print(f"Total {len(detected_elements_df)} objects detected.")
        for idx, row in detected_elements_df.iterrows():
            print(f"Detected object {idx}: {row['name']} at box {row['box']}")

    ui_json = {
        "name": "test",
        "elements": []
    }

    # 각 요소의 부모-자식 관계를 추적하기 위한 데이터 구조
    element_map = {}

    # DataFrame의 각 행을 순회하면서 요소 추가
    for idx, row in detected_elements_df.iterrows():
        # 'box' 컬럼에서 좌표 정보를 추출
        box = row['box']  # 'box' 컬럼은 딕셔너리 형태로 좌표를 포함하고 있음
        ymin, xmin, xmax, ymax = box['y1'], box['x1'], box['x2'], box['y2']  # 'box' 딕셔너리에서 좌표 추출

        # 감지된 요소의 고유 ID 생성
        element_id = f"{row['name']}-{idx}"

        # 요소 객체 생성
        element = {
            "type": row['name'],  # 감지된 객체의 이름
            "id": element_id,
            "position": {
                "top": f"{ymin}px",
                "left": f"{xmin}px",
                "width": f"{xmax - xmin}px",
                "height": f"{ymax - ymin}px"
            },
            "children": []  # 자식 요소가 있으면 추가
        }

        # element_map에 해당 요소를 저장
        element_map[element_id] = element

        # elements 리스트에 추가 (최초에는 부모가 없는 요소는 최상위에 추가)
        ui_json['elements'].append(element)

    # 계층 구조 생성: 부모 자식 관계 설정
    for element in ui_json['elements']:
        element_id = element["id"]
        # 다른 요소들과 겹치는 부분이 있는 경우 자식으로 추가
        for other_element in ui_json['elements']:
            if other_element["id"] != element_id:
                # 이 예시에서는 'top'과 'left'가 더 작은 요소를 부모로 보고,
                # 겹치는 요소를 자식으로 판단
                element_top = int(element["position"]["top"][:-2])
                element_left = int(element["position"]["left"][:-2])
                other_element_top = int(other_element["position"]["top"][:-2])
                other_element_left = int(other_element["position"]["left"][:-2])

                # 기준: 다른 요소의 'top'과 'left'가 더 작으면 부모로 설정
                if (element_top > other_element_top and element_left > other_element_left):
                    # 자식으로 추가
                    other_element["children"].append(element)

    # 결과 JSON 반환
    return json.dumps(ui_json, indent=2)

# 테스트 실행
image_path = '../screenshots/test.png'  # 경로를 실제 이미지로 설정
ui_json = detect_ui_elements(image_path)

# 최종 결과 출력
print("Resulting UI JSON:")
print(ui_json)

# 학습된 클래스 이름 출력
print("Class Names in the model:")
print(model.names)
