#!/usr/bin/env python3
"""
UI Detector - Quick Start Script
간단한 명령으로 UI 탐지를 시작할 수 있는 스크립트입니다.
"""

import os
import sys
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='UI Detector - Quick Start')
    parser.add_argument('--image', '-i', type=str, default='./screenshots/test.png',
                       help='탐지할 이미지 경로 (기본값: ./screenshots/test.png)')
    parser.add_argument('--model', '-m', type=str, default='train4',
                       choices=['train4', 'train6', 'default'],
                       help='사용할 모델 (기본값: train4)')
    parser.add_argument('--output', '-o', type=str, default='./json/quick_result.json',
                       help='결과 JSON 파일 경로 (기본값: ./json/quick_result.json)')
    parser.add_argument('--visualize', '-v', action='store_true',
                       help='시각화 결과 저장')
    parser.add_argument('--diagnose', '-d', action='store_true',
                       help='모델 진단 실행')
    
    args = parser.parse_args()
    
    print("🚀 UI Detector - Quick Start")
    print("=" * 40)
    
    # 1. 모델 진단 (선택사항)
    if args.diagnose:
        print("🔍 Running model diagnosis...")
        try:
            import sys
            sys.path.append('../analysis')
            import model_diagnosis
            model_diagnosis.main()
            print()
        except ImportError:
            print("❌ model_diagnosis.py not found")
    
    # 2. 이미지 존재 확인
    if not os.path.exists(args.image):
        print(f"❌ Error: Image file not found at {args.image}")
        print("💡 Available images:")
        screenshots_dir = Path("./screenshots")
        if screenshots_dir.exists():
            for img_file in screenshots_dir.glob("*.png"):
                print(f"   - {img_file}")
        return
    
    # 3. 모델 경로 설정
    if args.model == 'default':
        model_path = "yolov5s.pt"
    else:
        model_path = f"runs/detect/{args.model}/weights/best.pt"
        if not os.path.exists(model_path):
            print(f"❌ Error: Model not found at {model_path}")
            print("💡 Available models:")
            runs_dir = Path("./runs/detect")
            if runs_dir.exists():
                for train_dir in runs_dir.iterdir():
                    if train_dir.is_dir():
                        weights_dir = train_dir / "weights"
                        if weights_dir.exists():
                            print(f"   - {train_dir.name}")
            return
    
    # 4. 탐지 실행
    print(f"📸 Processing image: {args.image}")
    print(f"🤖 Using model: {model_path}")
    
    try:
        # 개선된 탐지기 사용
        import sys
        sys.path.append('../detection')
        from improved_detector import ImprovedUIDetector
        
        detector = ImprovedUIDetector(model_path)
        ui_json = detector.detect_with_multiple_thresholds(args.image, args.visualize)
        
        if ui_json:
            # 결과 저장
            os.makedirs(os.path.dirname(args.output), exist_ok=True)
            with open(args.output, 'w', encoding='utf-8') as f:
                import json
                json.dump(ui_json, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Detection completed!")
            print(f"📊 Detected {ui_json['summary']['total_elements']} elements")
            print(f"💾 Results saved to: {args.output}")
            
            if args.visualize:
                print(f"📊 Visualization saved to: result/improved_detection_result.png")
            
            # 간단한 결과 요약
            print("\n📋 Quick Summary:")
            print(f"   - Total elements: {ui_json['summary']['total_elements']}")
            print(f"   - Unique types: {ui_json['summary']['unique_types']}")
            print(f"   - Average confidence: {ui_json['detection_info']['average_confidence']:.3f}")
            
            # 요소 타입별 개수
            type_counts = {}
            for elem in ui_json['elements']:
                elem_type = elem["type"]
                type_counts[elem_type] = type_counts.get(elem_type, 0) + 1
            
            print(f"   - Element types: {', '.join([f'{k}({v})' for k, v in type_counts.items()])}")
            
        else:
            print("❌ Detection failed!")
            
    except ImportError:
        print("❌ Error: improved_detector.py not found")
        print("💡 Please run: python improved_detector.py")
    except Exception as e:
        print(f"❌ Error during detection: {e}")

if __name__ == "__main__":
    main() 