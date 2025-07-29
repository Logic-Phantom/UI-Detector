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
    data_yaml = "data.yaml"
    if not os.path.exists(data_yaml):
        print(f"Error: {data_yaml} file not found!")
        return
    print("Starting YOLOv5 model training...")
    print(f"Data configuration: {data_yaml}")
    model = YOLO("yolov5s.pt")
    training_config = {
        "data": data_yaml,
        "epochs": 200,
        "batch": 16,
        "imgsz": 640,
        "patience": 30,
        "save": True,
        "save_period": 10,
        "cache": False,
        "device": "cpu",
        "workers": 4,
        "project": "runs/detect",
        "name": "train_aug_clean",
        "exist_ok": True,
        "pretrained": True,
        "optimizer": "SGD",
        "lr0": 0.01,
        "lrf": 0.1,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "warmup_epochs": 3,
        "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.1,
        "box": 7.5,
        "cls": 0.5,
        "dfl": 1.5,
        "nbs": 64,
        "overlap_mask": True,
        "mask_ratio": 4,
        "dropout": 0.0,
        "val": True,
        # 증강 옵션
        "hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4, "degrees": 0.2, "translate": 0.2, "scale": 0.5, "shear": 0.2, "perspective": 0.0, "flipud": 0.5, "fliplr": 0.5, "mosaic": 1.0, "mixup": 0.2,
    }
    try:
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