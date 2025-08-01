import os
import torch
from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path

def diagnose_model(model_path):
    """
    모델을 진단하고 문제점을 찾습니다.
    """
    print(f"=== Model Diagnosis for {model_path} ===")
    
    # 1. 모델 파일 존재 확인
    if not os.path.exists(model_path):
        print(f"❌ Error: Model file not found at {model_path}")
        return False
    
    print(f"✅ Model file exists: {model_path}")
    print(f"📁 File size: {os.path.getsize(model_path) / (1024*1024):.2f} MB")
    
    # 2. 모델 로드 시도
    try:
        model = YOLO(model_path)
        print(f"✅ Model loaded successfully")
        print(f"📊 Model info: {type(model)}")
        
        # 3. 모델 구조 확인
        if hasattr(model, 'names'):
            print(f"📋 Available classes: {len(model.names)}")
            print(f"📝 Class names: {model.names}")
        else:
            print("❌ Warning: Model has no class names")
        
        # 4. 테스트 이미지로 추론 테스트
        test_image_path = './screenshots/test.png'
        if os.path.exists(test_image_path):
            print(f"\n🔍 Testing inference on {test_image_path}")
            
            img = cv2.imread(test_image_path)
            print(f"📸 Image loaded: {img.shape}")
            
            # 다양한 임계값으로 테스트
            thresholds = [0.1, 0.25, 0.5, 0.75]
            
            for conf_thresh in thresholds:
                print(f"\n--- Testing with confidence threshold: {conf_thresh} ---")
                try:
                    results = model(img, conf=conf_thresh, verbose=False)
                    
                    if len(results) > 0:
                        result = results[0]
                        if hasattr(result, 'boxes') and result.boxes is not None:
                            boxes = result.boxes
                            print(f"  Detected {len(boxes)} objects")
                            
                            for i, box in enumerate(boxes):
                                cls = int(box.cls[0].cpu().numpy())
                                conf = float(box.conf[0].cpu().numpy())
                                label = model.names[cls] if cls < len(model.names) else f"class_{cls}"
                                print(f"    {i}: {label} (conf: {conf:.3f})")
                        else:
                            print("  No detection boxes found")
                    else:
                        print("  No detection results")
                        
                except Exception as e:
                    print(f"  ❌ Error during inference: {e}")
        else:
            print(f"❌ Test image not found: {test_image_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

def compare_models():
    """
    기본 모델과 커스텀 모델을 비교합니다.
    """
    print("\n=== Model Comparison ===")
    
    # 기본 YOLOv5 모델
    print("\n1. Testing default YOLOv5s model:")
    try:
        default_model = YOLO("yolov5s.pt")
        print("✅ Default model loaded successfully")
        
        # 테스트 이미지로 추론
        test_image_path = './screenshots/test.png'
        if os.path.exists(test_image_path):
            img = cv2.imread(test_image_path)
            results = default_model(img, conf=0.25, verbose=False)
            
            if len(results) > 0:
                result = results[0]
                if hasattr(result, 'boxes') and result.boxes is not None:
                    boxes = result.boxes
                    print(f"  Default model detected {len(boxes)} objects")
                    
                    for i, box in enumerate(boxes):
                        cls = int(box.cls[0].cpu().numpy())
                        conf = float(box.conf[0].cpu().numpy())
                        label = default_model.names[cls]
                        print(f"    {i}: {label} (conf: {conf:.3f})")
        
    except Exception as e:
        print(f"❌ Error with default model: {e}")
    
    # 커스텀 모델들
    custom_models = [
        "runs/detect/train6/weights/best.pt",
        "runs/detect/train5/weights/best.pt",
        "runs/detect/train4/weights/best.pt",
        "runs/detect/train3/weights/best.pt",
        "runs/detect/train2/weights/best.pt",
        "runs/detect/train/weights/best.pt"
    ]
    
    print("\n2. Testing custom models:")
    for model_path in custom_models:
        if os.path.exists(model_path):
            print(f"\n--- Testing {model_path} ---")
            diagnose_model(model_path)

def check_training_data():
    """
    학습 데이터를 확인합니다.
    """
    print("\n=== Training Data Check ===")
    
    # 데이터셋 구조 확인
    data_yaml = "data.yaml"
    if os.path.exists(data_yaml):
        print(f"✅ Data configuration file exists: {data_yaml}")
        
        with open(data_yaml, 'r', encoding='utf-8') as f:
            content = f.read()
            print(f"📄 Data config content:\n{content}")
    else:
        print(f"❌ Data configuration file not found: {data_yaml}")
    
    # 학습 이미지 확인
    train_dir = "screenshots/start"
    if os.path.exists(train_dir):
        print(f"\n📁 Training directory: {train_dir}")
        
        images_dir = os.path.join(train_dir, "images")
        labels_dir = os.path.join(train_dir, "labels")
        
        if os.path.exists(images_dir):
            image_files = [f for f in os.listdir(images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
            print(f"  📸 Training images: {len(image_files)}")
            for img_file in image_files:
                print(f"    - {img_file}")
        
        if os.path.exists(labels_dir):
            label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt') and f != 'classes.txt']
            print(f"  🏷️  Label files: {len(label_files)}")
            for label_file in label_files:
                print(f"    - {label_file}")
                
                # 라벨 파일 내용 확인
                label_path = os.path.join(labels_dir, label_file)
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                    print(f"      Lines: {len(lines)}")
                    if lines:
                        print(f"      Sample line: {lines[0].strip()}")
    else:
        print(f"❌ Training directory not found: {train_dir}")

def main():
    """
    메인 진단 함수
    """
    print("🔍 UI Detector Model Diagnosis")
    print("=" * 50)
    
    # 1. 학습 데이터 확인
    check_training_data()
    
    # 2. 모델 비교
    compare_models()
    
    # 3. 특정 모델 상세 진단
    print("\n" + "=" * 50)
    print("🔍 Detailed diagnosis of current model")
    diagnose_model("runs/detect/train6/weights/best.pt")
    
    print("\n" + "=" * 50)
    print("✅ Diagnosis completed!")

if __name__ == "__main__":
    main() 