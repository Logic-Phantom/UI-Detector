import torch
import xml.etree.ElementTree as ET

# 모델 로드
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# 이미지 분석
image_path = 'test.png'  # 분석할 이미지 파일 경로
results = model(image_path)

# 결과 추출 (Pandas DataFrame을 사용하여 객체 데이터 추출)
detections = results.pandas().xyxy[0].to_dict(orient="records")

# XML 구조 생성
annotation = ET.Element("annotation")

for detection in detections:
    obj = ET.SubElement(annotation, "object")
    name = ET.SubElement(obj, "name")
    name.text = detection["name"]
    
    confidence = ET.SubElement(obj, "confidence")
    confidence.text = str(detection["confidence"])
    
    bndbox = ET.SubElement(obj, "bndbox")
    xmin = ET.SubElement(bndbox, "xmin")
    xmin.text = str(detection["xmin"])
    ymin = ET.SubElement(bndbox, "ymin")
    ymin.text = str(detection["ymin"])
    xmax = ET.SubElement(bndbox, "xmax")
    xmax.text = str(detection["xmax"])
    ymax = ET.SubElement(bndbox, "ymax")
    ymax.text = str(detection["ymax"])

# XML 트리 생성 및 파일 저장
tree = ET.ElementTree(annotation)
xml_file_path = "runs/detect/exp/output.xml"  # 파일 경로 설정
tree.write(xml_file_path)

print(f"XML file saved to: {xml_file_path}")
