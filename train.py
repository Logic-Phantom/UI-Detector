'''
Created on 2025. 5. 14.

@author: LCM
'''
from ultralytics import YOLO
import os

def train_model():
    """
    YOLOv5 모델을 사용하여 UI 요소 감지 모델을 학습시킵니다.
    """
    # 데이터 설정 파일 확인
    data_yaml = "data.yaml"
    if not os.path.exists(data_yaml):
        print(f"Error: {data_yaml} file not found!")
        return
    
    print("Starting YOLOv5 model training...")
    print(f"Data configuration: {data_yaml}")
    
    # YOLOv5 모델 로드 (YOLOv8 대신 YOLOv5 사용)
    model = YOLO("yolov5s.pt")  # 사전 훈련된 YOLOv5s 모델 사용
    
    # 학습 설정
    training_config = {
        "data": data_yaml,
        "epochs": 100,  # 에포크 수 증가
        "batch": 16,    # 배치 크기
        "imgsz": 640,   # 이미지 크기
        "patience": 20, # Early stopping patience
        "save": True,   # 모델 저장
        "save_period": 10,  # 10 에포크마다 저장
        "cache": False, # 메모리 절약
        "device": "cpu", # CPU 사용 (GPU가 있다면 "0"으로 변경)
        "workers": 4,   # 데이터 로더 워커 수
        "project": "runs/detect",  # 프로젝트 디렉토리
        "name": "train_improved",  # 실험 이름
        "exist_ok": True,  # 기존 실험 덮어쓰기
        "pretrained": True,  # 사전 훈련된 가중치 사용
        "optimizer": "SGD",  # 옵티마이저
        "lr0": 0.01,  # 초기 학습률
        "lrf": 0.1,   # 최종 학습률 비율
        "momentum": 0.937,  # 모멘텀
        "weight_decay": 0.0005,  # 가중치 감쇠
        "warmup_epochs": 3,  # 워밍업 에포크
        "warmup_momentum": 0.8,  # 워밍업 모멘텀
        "warmup_bias_lr": 0.1,  # 워밍업 바이어스 학습률
        "box": 7.5,  # 박스 손실 가중치
        "cls": 0.5,  # 클래스 손실 가중치
        "dfl": 1.5,  # DFL 손실 가중치
        "fl_gamma": 0.0,  # Focal loss gamma
        "label_smoothing": 0.0,  # 라벨 스무딩
        "nbs": 64,  # 명목 배치 크기
        "overlap_mask": True,  # 마스크 오버랩
        "mask_ratio": 4,  # 마스크 다운샘플 비율
        "dropout": 0.0,  # 드롭아웃
        "val": True,  # 검증 수행
    }
    
    try:
        # 모델 학습 시작
        print("Training configuration:")
        for key, value in training_config.items():
            print(f"  {key}: {value}")
        
        results = model.train(**training_config)
        
        print("\nTraining completed successfully!")
        print(f"Best model saved at: {results.save_dir}")
        print(f"Training metrics: {results.results_dict}")
        
        return results
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        return None

if __name__ == "__main__":
    train_model()