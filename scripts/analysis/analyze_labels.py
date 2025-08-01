import os
from collections import Counter
from pathlib import Path
import sys

def analyze_label_distribution(labels_dir, classes_path, output_path=None):
    # 클래스명 로드
    try:
        with open(classes_path, 'r', encoding='utf-8') as f:
            class_names = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        # 대안 경로 시도
        alternative_paths = [
            'screenshots/start/labels/classes.txt',
            'output/result/classes.txt',
            'yolo/datasets/screenshots/val/labels/classes.txt'
        ]
        
        class_names = []
        for alt_path in alternative_paths:
            try:
                with open(alt_path, 'r', encoding='utf-8') as f:
                    class_names = [line.strip() for line in f if line.strip()]
                    print(f"✅ Loaded classes from: {alt_path}")
                    break
            except FileNotFoundError:
                continue
        
        if not class_names:
            print("⚠️  Warning: Could not find classes.txt file. Using default class names.")
            class_names = ['Button', 'InputBox', 'TextArea', 'Group', 'Frame']
    
    label_counter = Counter()
    total_labels = 0
    label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt') and f != 'classes.txt']
    output_lines = []
    output_lines.append(f"라벨 파일 개수: {len(label_files)}\n")
    
    for label_file in label_files:
        with open(os.path.join(labels_dir, label_file), 'r') as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split()
                    if len(parts) >= 1:
                        class_idx = int(parts[0])
                        if 0 <= class_idx < len(class_names):
                            label_counter[class_names[class_idx]] += 1
                            total_labels += 1
                        else:
                            output_lines.append(f"[Warning] Invalid class index {class_idx} in {label_file}\n")
    output_lines.append("\n=== 라벨 분포 분석 ===\n")
    output_lines.append(f"총 라벨 수: {total_labels}\n")
    for cls, cnt in label_counter.most_common():
        output_lines.append(f"  {cls:20s}: {cnt}\n")
    output_lines.append(f"클래스 종류 수: {len(label_counter)} / {len(class_names)}\n")
    if len(label_counter) < len(class_names):
        output_lines.append("[경고] 일부 클래스는 라벨이 전혀 없음!\n")
    output_lines.append(f"\n(디버그) label_counter: {label_counter}\n")
    output_lines.append(f"(디버그) class_names: {class_names}\n")
    
    # 라벨링 오류 점검 (좌표값 범위 체크)
    output_lines.append("\n=== 라벨링 오류 점검 ===\n")
    for label_file in label_files:
        with open(os.path.join(labels_dir, label_file), 'r') as f:
            for i, line in enumerate(f):
                parts = line.strip().split()
                if len(parts) == 5:
                    class_idx, xc, yc, w, h = parts
                    try:
                        xc, yc, w, h = map(float, (xc, yc, w, h))
                        if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < w <= 1 and 0 < h <= 1):
                            output_lines.append(f"[오류] {label_file} {i+1}번째 줄: 좌표값 비정상 xc={xc}, yc={yc}, w={w}, h={h}\n")
                    except Exception as e:
                        output_lines.append(f"[오류] {label_file} {i+1}번째 줄: 변환 실패 {e}\n")
                else:
                    output_lines.append(f"[오류] {label_file} {i+1}번째 줄: 라벨 포맷 이상 -> {line.strip()}\n")
    
    result = ''.join(output_lines)
    print(result)
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(result)
        print(f"분석 결과가 {output_path}에 저장되었습니다.")

def main():
    labels_dir = '../../yolo/datasets/screenshots/start/labels'
    classes_path = os.path.join(labels_dir, 'classes.txt')
    output_path = './label_analysis_result.txt'
    if not os.path.exists(labels_dir) or not os.path.exists(classes_path):
        print('라벨 디렉토리 또는 classes.txt가 존재하지 않습니다.')
        return
    analyze_label_distribution(labels_dir, classes_path, output_path)

if __name__ == '__main__':
    main() 