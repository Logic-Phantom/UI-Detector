'''
Created on 2025. 5. 14.

@author: LCM
'''
from ultralytics import YOLO
import os
import argparse
from pathlib import Path

def train_model(model_size="s", epochs=300, batch_size=16, img_size=640, 
                data_augmentation=True, early_stopping=True, resume=False):
    """
    YOLOv5 모델을 사용하여 UI 요소 감지 모델을 학습시킵니다.
    
    Args:
        model_size (str): 모델 크기 ('n', 's', 'm', 'l', 'x')
        epochs (int): 학습 에포크 수
        batch_size (int): 배치 크기
        img_size (int): 입력 이미지 크기
        data_augmentation (bool): 데이터 증강 사용 여부
        early_stopping (bool): 조기 종료 사용 여부
        resume (bool): 이전 학습 재개 여부
    """
    data_yaml = "data.yaml"
    if not os.path.exists(data_yaml):
        print(f"Error: {data_yaml} file not found!")
        return None
    
    print("🚀 Starting YOLOv5 model training...")
    print(f"📋 Data configuration: {data_yaml}")
    print(f"🤖 Model size: yolov5{model_size}")
    print(f"⏱️  Epochs: {epochs}")
    print(f"📦 Batch size: {batch_size}")
    print(f"🖼️  Image size: {img_size}")
    
    # 모델 선택
    model_path = f"yolov5{model_size}.pt"
    if not os.path.exists(model_path):
        print(f"⚠️  Warning: {model_path} not found, using default YOLOv5s")
        model_path = "yolov5s.pt"
    
    model = YOLO(model_path)
    
    # 기본 학습 설정
    training_config = {
        "data": data_yaml,
        "epochs": epochs,
        "batch": batch_size,
        "imgsz": img_size,
        "patience": 50 if early_stopping else 0,  # 조기 종료 인내심 증가
        "save": True,
        "save_period": 10,
        "cache": True,  # 캐시 활성화로 학습 속도 향상
        "device": "cpu",  # GPU 사용 가능시 "0"으로 변경
        "workers": 4,
        "project": "runs/detect",
        "name": f"train_improved_{model_size}",
        "exist_ok": True,
        "pretrained": True,
        "resume": resume,
        
        # 최적화기 설정 개선
        "optimizer": "AdamW",  # SGD 대신 AdamW 사용
        "lr0": 0.001,  # 학습률 조정
        "lrf": 0.01,  # 최종 학습률 비율
        "momentum": 0.937,
        "weight_decay": 0.0005,
        
        # 워밍업 설정
        "warmup_epochs": 5,  # 워밍업 에포크 증가
        "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.1,
        
        # 손실 함수 가중치 조정
        "box": 7.5,  # 바운딩 박스 손실 가중치
        "cls": 0.5,  # 분류 손실 가중치
        "dfl": 1.5,  # DFL 손실 가중치
        
        # 검증 설정
        "val": True,
        "plots": True,  # 학습 과정 플롯 생성
        
        # 정규화 설정
        "nbs": 64,
        "overlap_mask": True,
        "mask_ratio": 4,
        "dropout": 0.0,
    }
    
    # 데이터 증강 설정 (UI 요소 특성에 맞게 조정)
    if data_augmentation:
        training_config.update({
            # 색상 증강 (UI 요소의 색상 변화에 강건하게)
            "hsv_h": 0.015,  # 색조 변화
            "hsv_s": 0.7,    # 채도 변화
            "hsv_v": 0.4,    # 명도 변화
            
            # 기하학적 증강
            "degrees": 0.1,   # 회전 (UI 요소는 회전이 적어야 함)
            "translate": 0.1, # 이동
            "scale": 0.5,     # 크기 변화
            "shear": 0.1,     # 기울기 (UI 요소는 기울기가 적어야 함)
            "perspective": 0.0, # 원근 (UI에서는 사용하지 않음)
            
            # 뒤집기
            "flipud": 0.0,    # 상하 뒤집기 (UI에서는 사용하지 않음)
            "fliplr": 0.5,    # 좌우 뒤집기
            
            # 고급 증강
            "mosaic": 1.0,    # 모자이크 증강
            "mixup": 0.1,     # 믹스업 (UI에서는 적게 사용)
            
            # 추가 증강
            "copy_paste": 0.1,  # 복사-붙여넣기 증강
            "cutmix": 0.1,      # 컷믹스 증강
        })
    
    # UI 요소 특화 설정
    training_config.update({
        "anchor_t": 4.0,      # 앵커 임계값
        "fl_gamma": 0.0,      # Focal Loss 감마
        "label_smoothing": 0.1, # 라벨 스무딩
        "nbs": 64,            # 정규화 배치 크기
    })
    
    try:
        print("\n📊 Training configuration:")
        for key, value in training_config.items():
            if isinstance(value, (int, float, str, bool)):
                print(f"  {key}: {value}")
        
        print(f"\n🎯 Starting training with {epochs} epochs...")
        results = model.train(**training_config)
        
        print("\n✅ Training completed successfully!")
        print(f"📁 Best model saved at: {results.save_dir}")
        
        # 결과 요약
        if hasattr(results, 'results_dict'):
            metrics = results.results_dict
            print(f"\n📈 Training metrics:")
            print(f"  - mAP50: {metrics.get('metrics/mAP50(B)', 'N/A')}")
            print(f"  - mAP50-95: {metrics.get('metrics/mAP50-95(B)', 'N/A')}")
            print(f"  - Precision: {metrics.get('metrics/precision(B)', 'N/A')}")
            print(f"  - Recall: {metrics.get('metrics/recall(B)', 'N/A')}")
        
        return results
        
    except Exception as e:
        print(f"❌ Training failed with error: {e}")
        return None

def main():
    """메인 함수 - 명령행 인자 처리"""
    parser = argparse.ArgumentParser(description='YOLOv5 UI Element Detection Training')
    parser.add_argument('--model', '-m', type=str, default='s', 
                       choices=['n', 's', 'm', 'l', 'x'],
                       help='Model size (n=nano, s=small, m=medium, l=large, x=xlarge)')
    parser.add_argument('--epochs', '-e', type=int, default=300,
                       help='Number of training epochs')
    parser.add_argument('--batch', '-b', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--img-size', '-i', type=int, default=640,
                       help='Input image size')
    parser.add_argument('--no-aug', action='store_true',
                       help='Disable data augmentation')
    parser.add_argument('--no-early-stop', action='store_true',
                       help='Disable early stopping')
    parser.add_argument('--resume', '-r', action='store_true',
                       help='Resume from previous training')
    
    args = parser.parse_args()
    
    # 데이터셋 확인
    data_yaml = "data.yaml"
    if not os.path.exists(data_yaml):
        print(f"❌ Error: {data_yaml} file not found!")
        print("💡 Please ensure data.yaml exists in the current directory")
        return
    
    # 데이터셋 정보 출력
    print("🔍 Checking dataset configuration...")
    try:
        import yaml
        with open(data_yaml, 'r', encoding='utf-8') as f:
            data_config = yaml.safe_load(f)
        
        print(f"📁 Dataset path: {data_config.get('path', 'N/A')}")
        print(f"📸 Train images: {data_config.get('train', 'N/A')}")
        print(f"📸 Val images: {data_config.get('val', 'N/A')}")
        print(f"🏷️  Number of classes: {data_config.get('nc', 'N/A')}")
        
        # 클래스 이름 출력
        if 'names' in data_config:
            print(f"📋 Classes: {', '.join(data_config['names'][:5])}{'...' if len(data_config['names']) > 5 else ''}")
    
    except Exception as e:
        print(f"⚠️  Warning: Could not read data.yaml: {e}")
    
    # 학습 실행
    results = train_model(
        model_size=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.img_size,
        data_augmentation=not args.no_aug,
        early_stopping=not args.no_early_stop,
        resume=args.resume
    )
    
    if results:
        print("\n🎉 Training completed successfully!")
        print("💡 Next steps:")
        print("   1. Test the model: python scripts/detection/improved_detector.py")
        print("   2. Analyze results: python scripts/analysis/model_diagnosis.py")
        print("   3. Convert to JSON: python scripts/conversion/yolo_to_figma_json_v3.py")
    else:
        print("\n❌ Training failed!")

if __name__ == "__main__":
    main()