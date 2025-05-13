import json
import cv2
from ultralytics import YOLO

# YOLOv5 모델 로드
model = YOLO("yolov5s.pt")  # 모델 로딩

def detect_ui_elements(image_path):
    # 이미지 로드
    img = cv2.imread(image_path)

    # YOLOv5 모델로 이미지 분석
    results = model(img)

    # 결과에서 pandas 형식으로 변환 후 첫 번째 객체에 접근
    detected_elements_df = results[0].to_df()  # pandas DataFrame 형식으로 변환

    # DataFrame 출력하여 실제 구조 확인 (디버깅용)
    print(detected_elements_df)

    ui_json = {
        "name": "test",
        "elements": []
    }

    # DataFrame의 각 행을 순회하면서 요소 추가
    for idx, row in detected_elements_df.iterrows():
        # 'box' 컬럼에서 좌표 정보를 추출
        box = row['box']  # 'box' 컬럼은 딕셔너리 형태로 좌표를 포함하고 있음
        ymin, xmin, xmax, ymax = box['y1'], box['x1'], box['x2'], box['y2']  # 'box' 딕셔너리에서 좌표 추출

        element = {
            "type": row['name'],  # 감지된 객체의 이름
            "id": f"{row['name']}-{idx}",
            "position": {
                "top": f"{ymin}px",
                "left": f"{xmin}px",
                "width": f"{xmax - xmin}px",
                "height": f"{ymax - ymin}px"
            },
            "children": []  # 자식 요소가 있으면 추가
        }
        ui_json['elements'].append(element)

    # 결과 JSON 반환
    return json.dumps(ui_json, indent=2)

# 테스트 실행
image_path = 'C:/Users/LCM/git/UI-Detector/img/test.png'  # 경로를 실제 이미지로 설정
ui_json = detect_ui_elements(image_path)
print(ui_json)
print(model.names)