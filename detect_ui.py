import torch
import json

# 이미지 경로 (로컬 이미지 경로로 수정하세요)
image_path = 'test.png'  # 예: 'C:/Users/LCM/Desktop/test.png'

# 사전 학습된 YOLOv5s 모델 로드
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# 이미지 감지 실행
results = model(image_path)

# 결과 추출 (Tensor 형식 → Python dict로 변환)
detections = []
for *box, conf, cls in results.xyxy[0]:
    detections.append({
        "class": model.names[int(cls)],
        "confidence": float(conf),
        "x1": float(box[0]),
        "y1": float(box[1]),
        "x2": float(box[2]),
        "y2": float(box[3]),
    })

# 결과 JSON 출력
print(json.dumps(detections, indent=2))

# 결과 이미지 저장 (runs/detect/exp/ 경로에 저장됨)
results.save()
