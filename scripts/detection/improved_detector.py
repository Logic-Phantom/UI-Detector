import json
import cv2
import numpy as np
from ultralytics import YOLO
import os
from pathlib import Path
import datetime

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

class ImprovedUIDetector:
    def __init__(self, model_path="runs/detect/train_aug_clean/weights/best.pt"):
        self.model_path = model_path
        if not os.path.exists(model_path):
            print(f"Warning: Custom model not found at {model_path}")
            print("Using default YOLOv5s model...")
            self.model = YOLO("yolov5s.pt")
        else:
            self.model = YOLO(model_path)
        print(f"✅ Model loaded: {self.model_path}")
        print(f"📋 Available classes: {len(CLASS_NAMES)}")

    def detect_with_multiple_thresholds(self, image_path, save_visualization=True):
        if not os.path.exists(image_path):
            print(f"❌ Error: Image file not found at {image_path}")
            return None
        img = cv2.imread(image_path)
        if img is None:
            print(f"❌ Error: Failed to load image from {image_path}")
            return None
        print(f"📸 Image loaded: {img.shape}")
        thresholds = [0.05, 0.1, 0.15, 0.2, 0.25]
        best_result = None
        best_score = 0
        for conf_thresh in thresholds:
            print(f"\n🔍 Testing with confidence threshold: {conf_thresh}")
            try:
                results = self.model(img, conf=conf_thresh, verbose=False)
                if len(results) > 0:
                    result = results[0]
                    if hasattr(result, 'boxes') and result.boxes is not None:
                        boxes = result.boxes
                        if len(boxes) > 0:
                            confidences = [float(box.conf[0].cpu().numpy()) for box in boxes]
                            avg_confidence = np.mean(confidences)
                            score = len(boxes) * avg_confidence
                            print(f"  Detected {len(boxes)} objects (avg conf: {avg_confidence:.3f}, score: {score:.3f})")
                            if score > best_score:
                                best_score = score
                                best_result = {
                                    'threshold': conf_thresh,
                                    'results': results,
                                    'score': score,
                                    'num_detections': len(boxes),
                                    'avg_confidence': avg_confidence
                                }
                        else:
                            print("  No objects detected")
                    else:
                        print("  No detection boxes found")
                else:
                    print("  No detection results")
            except Exception as e:
                print(f"  ❌ Error during inference: {e}")
        if best_result is None:
            print("❌ No successful detections found")
            return None
        print(f"\n🎯 Best detection result:")
        print(f"  Threshold: {best_result['threshold']}")
        print(f"  Detections: {best_result['num_detections']}")
        print(f"  Average confidence: {best_result['avg_confidence']:.3f}")
        print(f"  Score: {best_result['score']:.3f}")
        ui_json = self._create_json_result(best_result['results'], image_path, best_result)
        if save_visualization:
            self._create_visualization(best_result['results'], img, image_path)
        return ui_json

    def _create_json_result(self, results, image_path, detection_info):
        detected_elements = []
        class_counter = {}
        if len(results) > 0:
            result = results[0]
            if hasattr(result, 'boxes') and result.boxes is not None:
                boxes = result.boxes
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    cls = int(box.cls[0].cpu().numpy())
                    conf = float(box.conf[0].cpu().numpy())
                    label = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else f'class_{cls}'
                    class_counter[label] = class_counter.get(label, 0) + 1
                    element = {
                        "type": label,
                        "id": f"{label}-{i}",
                        "confidence": conf,
                        "position": {
                            "top": f"{y1}px",
                            "left": f"{x1}px",
                            "width": f"{x2 - x1}px",
                            "height": f"{y2 - y1}px"
                        },
                        "bbox": [x1, y1, x2, y2],
                        "area": (x2 - x1) * (y2 - y1),
                        "children": []
                    }
                    detected_elements.append(element)
        ui_json = {
            "name": Path(image_path).stem,
            "image_path": image_path,
            "detection_info": {
                "threshold": detection_info['threshold'],
                "total_detections": detection_info['num_detections'],
                "average_confidence": detection_info['avg_confidence'],
                "detection_score": detection_info['score'],
                "model_path": self.model_path
            },
            "elements": detected_elements,
            "summary": {
                "total_elements": len(detected_elements),
                "unique_types": len(set([elem["type"] for elem in detected_elements])),
                "total_area": sum([elem["area"] for elem in detected_elements]),
                "class_counts": class_counter
            }
        }
        return ui_json

    def _create_visualization(self, results, img, image_path):
        result_img = img.copy()
        if len(results) > 0:
            result = results[0]
            if hasattr(result, 'boxes') and result.boxes is not None:
                boxes = result.boxes
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    cls = int(box.cls[0].cpu().numpy())
                    conf = float(box.conf[0].cpu().numpy())
                    label = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else f'class_{cls}'
                    color = self._get_color(cls)
                    cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)
                    label_text = f"{label}: {conf:.2f}"
                    (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(result_img, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
                    cv2.putText(result_img, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        output_dir = Path("result")
        output_dir.mkdir(exist_ok=True)
        result_path = output_dir / "improved_detection_result.png"
        cv2.imwrite(str(result_path), result_img)
        print(f"📊 Visualization saved to: {result_path}")

    def _get_color(self, class_id):
        np.random.seed(class_id)
        color = tuple(int(x) for x in np.random.randint(0, 255, 3))
        return color

    def analyze_results(self, ui_json):
        if not ui_json or "elements" not in ui_json:
            print("❌ No detection results to analyze")
            return
        elements = ui_json["elements"]
        if len(elements) == 0:
            print("❌ No elements detected")
            return
        print("\n📊 Detection Analysis:")
        print("=" * 50)
        print(f"Total elements: {len(elements)}")
        print(f"Detection threshold: {ui_json['detection_info']['threshold']}")
        print(f"Average confidence: {ui_json['detection_info']['average_confidence']:.3f}")
        print(f"Detection score: {ui_json['detection_info']['detection_score']:.3f}")
        type_counts = ui_json['summary']['class_counts']
        print(f"\nElement types ({len(type_counts)} unique):")
        for elem_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {elem_type}: {count}")
        confidence_scores = [elem["confidence"] for elem in elements]
        areas = [elem["area"] for elem in elements]
        print(f"\nConfidence statistics:")
        print(f"  Min: {np.min(confidence_scores):.3f}")
        print(f"  Max: {np.max(confidence_scores):.3f}")
        print(f"  Mean: {np.mean(confidence_scores):.3f}")
        print(f"  Std: {np.std(confidence_scores):.3f}")
        print(f"\nArea statistics:")
        print(f"  Total area: {sum(areas):,} pixels")
        print(f"  Average area: {np.mean(areas):.0f} pixels")
        print(f"  Largest element: {max(areas):,} pixels")

def main():
    print("🚀 Improved UI Detector (screenshots/ 하위 이미지 전체 테스트)")
    print("=" * 50)
    detector = ImprovedUIDetector(model_path="runs/detect/train_aug_clean/weights/best.pt")
    image_dir = './screenshots/'
    today = datetime.datetime.now().strftime('%Y-%m-%d')
    json_dir = f'./json/{today}'
    png_dir = f'./result/{today}'
    os.makedirs(json_dir, exist_ok=True)
    os.makedirs(png_dir, exist_ok=True)
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, file)
                print(f"\n📸 Processing image: {image_path}")
                ui_json = detector.detect_with_multiple_thresholds(image_path, save_visualization=False)
                if ui_json:
                    detector.analyze_results(ui_json)
                    # JSON 저장
                    output_json = os.path.join(json_dir, f'{Path(file).stem}_detection.json')
                    with open(output_json, 'w', encoding='utf-8') as f:
                        json.dump(ui_json, f, indent=2, ensure_ascii=False)
                    print(f"\n💾 Results saved to: {output_json}")
                    # PNG 저장 (시각화)
                    img = cv2.imread(image_path)
                    results = detector.model(img, conf=ui_json['detection_info']['threshold'], verbose=False)
                    if len(results) > 0:
                        result_img = img.copy()
                        result = results[0]
                        if hasattr(result, 'boxes') and result.boxes is not None:
                            boxes = result.boxes
                            for i, box in enumerate(boxes):
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                                cls = int(box.cls[0].cpu().numpy())
                                conf = float(box.conf[0].cpu().numpy())
                                label = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else f'class_{cls}'
                                color = detector._get_color(cls)
                                cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)
                                label_text = f"{label}: {conf:.2f}"
                                (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                                cv2.rectangle(result_img, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
                                cv2.putText(result_img, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        output_png = os.path.join(png_dir, f'{Path(file).stem}_detection.png')
                        cv2.imwrite(output_png, result_img)
                        print(f"🖼️  Visualization saved to: {output_png}")
                else:
                    print("❌ Detection failed!")

if __name__ == "__main__":
    main() 