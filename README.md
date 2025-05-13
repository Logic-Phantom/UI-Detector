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

---  



# Eclipse에서 YOLOv5 Python 프로젝트 세팅 가이드

이 문서는 Eclipse IDE에서 YOLOv5 기반 Python 프로젝트를 설정하고 실행하는 방법을 안내합니다.

## 1. Eclipse IDE 설치

- [Eclipse 공식 사이트](https://www.eclipse.org/downloads/)에서 "Eclipse IDE for Python Developers" 또는 "Eclipse IDE for Java Developers" + 플러그인 설치
- 설치 후 실행

## 2. PyDev 플러그인 설치 (필요 시)

1. Eclipse 상단 메뉴에서 `Help` > `Eclipse Marketplace...` 클릭
2. 검색창에 `PyDev` 입력
3. `PyDev for Eclipse` 설치

설치 후 Eclipse 재시작

## 3. Python 인터프리터 설정

1. 상단 메뉴 `Window` > `Preferences` 클릭
2. `PyDev` > `Interpreters` > `Python Interpreter` 메뉴 선택
3. `New...` 버튼 클릭 → 시스템의 Python 설치 경로 선택 (예: `C:\Python311\python.exe`)
4. OK 후 적용

## 4. 프로젝트 가져오기

1. Git에서 클론
```bash
git clone https://github.com/your-username/your-repo-name.git
```

2. Eclipse에서
   - `File` > `Import...` > `General` > `Existing Projects into Workspace`
   - 경로: 위에서 클론한 폴더 지정

## 5. 가상환경(vEnv) 설정 (선택)

- Python 가상환경을 만들었다면, PyDev 프로젝트에서 해당 가상환경을 인터프리터로 설정할 수 있습니다.

## 6. 의존성 설치

Eclipse 내부 터미널 또는 외부 터미널에서:
```bash
pip install -r requirements.txt
```

## 7. 실행

- `your_main_script.py` 우클릭 → `Run As` > `Python Run`
- 또는 상단 실행 버튼 클릭

---

✅ 참고:
- 모델 파일(`yolov5s.pt`)은 실행 전 다운로드 필요
- PyDev가 잘 작동하지 않을 경우 Visual Studio Code도 대안입니다
