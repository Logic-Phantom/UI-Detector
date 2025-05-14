'''
Created on 2025. 5. 14.

@author: LCM
'''
import os

def print_tree(root_dir, indent=""):
    for item in sorted(os.listdir(root_dir)):
        path = os.path.join(root_dir, item)
        if os.path.isdir(path):
            print(indent + "📁 " + item)
            print_tree(path, indent + "    ")
        else:
            print(indent + "📄 " + item)

print_tree(".")  # 현재 폴더 기준 출력