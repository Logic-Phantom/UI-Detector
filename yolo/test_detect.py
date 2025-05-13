'''
Created on 2025. 5. 13.

@author: LCM
'''
# test_detect.py
from yolo.detector import detect
import json

if __name__ == '__main__':
    result = detect("../img/test.png")
    print(json.dumps(result, indent=2))
    
detect("../img/test.png", "output/detection.json")