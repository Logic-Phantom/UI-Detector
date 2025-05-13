# YOLOv5 기반 이미지 분석기

이 프로젝트는 YOLOv5 모델을 기반으로 특정 이미지를 분석하는 Python 프로그램입니다.

## 설치 방법

1. 프로젝트 클론
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

2. 가상 환경 생성 및 활성화 (선택)
```bash
python -m venv venv
source venv/bin/activate  # 윈도우는 venv\Scripts\activate
```

3. 의존성 설치
```bash
pip install -r requirements.txt
```

4. 모델 다운로드 (필요시)
```bash
# 예시: yolov5s.pt 다운로드
wget https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5s.pt
```

5. 실행
```bash
python your_main_script.py --source images/
```

## 결과
분석 결과는 `runs/detect/exp` 폴더에 저장됩니다.
