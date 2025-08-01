'''
Created on 2025. 5. 14.

@author: LCM
'''
import os

def print_tree(root_dir, indent="", exclude_dir="yolov5"):
    for item in sorted(os.listdir(root_dir)):
        path = os.path.join(root_dir, item)
        # 'yolov5' 디렉토리를 제외
        if item == exclude_dir:
            continue
        if os.path.isdir(path):
            print(indent + "📁 " + item)
            print_tree(path, indent + "    ", exclude_dir)
        else:
            print(indent + "📄 " + item)

print_tree(".")  # 현재 폴더 기준 출력