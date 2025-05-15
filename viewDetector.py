import torch
import json
import cv2
from pathlib import Path
from ultralytics import YOLO

# YOLO 모델 로드
model = YOLO("runs/detect/train4/weights/best.pt")  # 모델 로딩

def detect_ui_elements(image_path, save_path='detected_output.png'):
    # 이미지 로드
    img = cv2.imread(image_path)
    results = model(img)  # 예측 수행

    # 첫 번째 결과 객체 추출
    result = results[0]

    # 박스가 없는 경우 처리
    if result.boxes is None or result.boxes.cls is None:
        print("감지된 요소가 없습니다.")
        return {
            "name": Path(image_path).stem,
            "elements": []
        }

    # 박스 정보와 클래스 ID
    boxes = result.boxes.xywh.cpu().numpy()        # 중심 좌표 (x, y, w, h)
    class_ids = result.boxes.cls.cpu().numpy()     # 클래스 ID
    names = result.names                           # 클래스 ID → 클래스명 매핑 dict

    img_with_boxes = img.copy()
    detected_elements = []

    for idx, (box, class_id) in enumerate(zip(boxes, class_ids)):
        x, y, w, h = box
        label = names[int(class_id)]

        # 좌상단/우하단 좌표 계산
        top_left = (int(x - w / 2), int(y - h / 2))
        bottom_right = (int(x + w / 2), int(y + h / 2))

        # 박스 시각화
        cv2.rectangle(img_with_boxes, top_left, bottom_right, (0, 255, 0), 2)
        cv2.putText(img_with_boxes, label, (top_left[0], top_left[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # JSON 요소 추가
        detected_elements.append({
            "type": label,
            "id": f"{label}-{idx}",
            "position": {
                "top": f"{int(y)}px",
                "left": f"{int(x)}px",
                "width": f"{int(w)}px",
                "height": f"{int(h)}px"
            },
            "children": []
        })

    # 이미지 저장
    cv2.imwrite(save_path, img_with_boxes)
    print(f"탐지 결과 이미지 저장 완료: {save_path}")

    # JSON 반환
    return {
        "name": Path(image_path).stem,
        "elements": detected_elements
    }



# 예시 실행
image_path = './screenshots/test.png'
output_image_path = './result/detected_result.png'
ui_json = detect_ui_elements(image_path, output_image_path)

# JSON도 출력
print(json.dumps(ui_json, indent=2))
