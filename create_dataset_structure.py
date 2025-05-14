'''
Created on 2025. 5. 14.

@author: LCM
'''
# create_dataset_structure.py
# 학습용 폴더 구축 
import os
import shutil

def create_yolo_structure(base_dir='ui-dataset'):
    folders = [
        f"{base_dir}/images/train",
        f"{base_dir}/images/val",
        f"{base_dir}/labels/train",
        f"{base_dir}/labels/val"
    ]

    for folder in folders:
        os.makedirs(folder, exist_ok=True)

    print("YOLOv5 학습용 폴더 구조가 생성되었습니다.")
    print(f"""
{base_dir}/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
""")

if __name__ == "__main__":
    create_yolo_structure()
