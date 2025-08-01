# YOLOv5 기반 UI 리버스엔지니어링 → .clx 변환 프로젝트

## 🎯 프로젝트 개요

이 프로젝트는 YOLOv5를 사용하여 UI 스크린샷에서 UI 요소들을 자동으로 감지하고, 이를 JSON 형태로 변환하는 도구입니다. 최종적으로는 CLEOPATRA .clx 형식으로 변환하는 것을 목표로 합니다.

## ✅ 현재 상태

### 🧱 1단계: YOLOv5 모델 환경 구성 및 테스트 ✅
- [x] YOLOv5 모델 로딩 (ultralytics 사용)
- [x] 이미지 입력 및 객체 감지 확인
- [x] UI 요소 클래스 정의 (43개 클래스)
- [x] 커스텀 데이터셋(스크린샷 기반) 만들기 및 YOLOv5 재학습
- [x] 감지 결과를 JSON 또는 구조화된 데이터로 변환

### 🧩 2단계: 감지 결과 → CLEOPATRA .clx 변환기 🔄
- [x] CLEOPATRA .clx XML 구조 파악
- [x] UI 요소와 `<cl:*>` 매핑 룰 정의
- [ ] 요소 위치(x, y, width, height) → `<cl:xylayoutdata>` 생성
- [ ] 각 UI 요소 유형별 `<cl:button>`, `<cl:inputbox>` 등 생성 코드 작성
- [ ] 스타일/속성 자동 부여 로직 (필요 시)

## 🚀 최근 개선사항

### 🔧 주요 수정사항
1. **모델 성능 개선**: train4 모델 사용으로 탐지 성공률 향상
2. **다중 임계값 탐지**: 여러 신뢰도 임계값으로 최적 결과 선택
3. **상세한 분석 기능**: 탐지 결과에 대한 통계 및 시각화
4. **에러 처리 강화**: 파일 존재 확인 및 예외 처리
5. **경로 문제 해결**: 상대 경로 및 절대 경로 처리 개선

### 📊 성능 개선 결과
- **이전**: 탐지 실패 (0개 객체)
- **현재**: 평균 5개 객체 탐지 (신뢰도 0.208)
- **탐지 정확도**: 다양한 임계값으로 최적화된 결과

## 📁 프로젝트 구조

```
UI-Detector/
├── 📄 train.py                   # 모델 학습 스크립트
├── 📄 data.yaml                  # 데이터셋 설정
├── 📄 requirements.txt           # 의존성 패키지
├── 📁 scripts/                   # 스크립트 디렉토리
│   ├── 📁 conversion/            # 변환 스크립트
│   │   ├── yolo_to_figma_json.py
│   │   ├── yolo_to_figma_json_v2.py
│   │   └── yolo_to_figma_json_v3.py
│   ├── 📁 detection/             # 감지 스크립트
│   │   ├── improved_detector.py
│   │   ├── detectJson.py
│   │   ├── detectJsonDetail.py
│   │   ├── detectJsonTree.py
│   │   └── viewDetector.py
│   ├── 📁 analysis/              # 분석 스크립트
│   │   ├── analyze_labels.py
│   │   └── model_diagnosis.py
│   └── 📁 utils/                 # 유틸리티 스크립트
│       ├── quick_start.py
│       └── osStr.py
├── 📁 screenshots/               # 테스트 이미지
│   ├── test.png                  # 메인 테스트 이미지
│   └── start/                    # 학습 데이터
├── 📁 runs/detect/               # 학습된 모델들
│   ├── train4/weights/best.pt    # 최적 모델 (권장)
│   └── train6/weights/best.pt    # 대안 모델
├── 📁 json/                      # 탐지 결과 JSON
├── 📁 figma_json/                # Figma JSON 변환 결과
└── 📁 result/                    # 시각화 결과
```

## 🛠️ 설치 및 사용법

### 1. 환경 설정
```bash
# 의존성 설치
pip install -r requirements.txt
```

### 2. 기본 탐지 실행
```bash
# 기본 탐지기 사용
python scripts/detection/detectJson.py

# 개선된 감지기 사용 (권장)
python scripts/detection/improved_detector.py
```

### 3. JSON 변환 실행
```bash
# 기본 변환
python scripts/conversion/yolo_to_figma_json.py

# v3 변환 (권장 - Java 코드 호환)
python scripts/conversion/yolo_to_figma_json_v3.py
```

### 4. 모델 진단
```bash
# 모델 상태 확인
python scripts/analysis/model_diagnosis.py
```

### 4. 모델 재학습
```bash
# 새로운 데이터로 모델 학습
python train.py
```

## 📊 탐지 결과 예시

### JSON 출력 형식
```json
{
  "name": "test",
  "image_path": "./screenshots/test.png",
  "detection_info": {
    "threshold": 0.05,
    "total_detections": 5,
    "average_confidence": 0.208,
    "detection_score": 1.038,
    "model_path": "runs/detect/train4/weights/best.pt"
  },
  "elements": [
    {
      "type": "Group",
      "id": "Group-0",
      "confidence": 0.398,
      "position": {
        "top": "111px",
        "left": "75px",
        "width": "949px",
        "height": "310px"
      },
      "bbox": [75, 111, 1024, 421],
      "area": 294190,
      "children": []
    }
  ],
  "summary": {
    "total_elements": 5,
    "unique_types": 1,
    "total_area": 1660521
  }
}
```

## 🔍 지원하는 UI 요소 클래스

현재 43개의 UI 요소 클래스를 지원합니다:

- **기본 요소**: Button, InputBox, TextArea, CheckBox, RadioButton
- **컨테이너**: Group, Grid, TabFolder, Accordion
- **네비게이션**: NavigationBar, SideNavigation, Menu
- **입력 요소**: SearchInput, DateInput, NumberEditor, MaskEditor
- **선택 요소**: ComboBox, ListBox, Tree, Calendar
- **미디어**: Image, Video, Audio
- **기타**: Progress, Notification, FileUpload, Shell

## 🎯 다음 단계

### ⚙️ 3단계: 전체 파이프라인 통합
- [ ] 이미지 → YOLOv5 → 감지결과(JSON) → `.clx` 변환까지 자동화
- [ ] 디렉터리 감시 기능 (예: `screenshots/` 폴더에 이미지 생기면 자동 변환)
- [ ] 변환 결과 미리보기 (선택사항: HTML or CLEOPATRA로 열기)

### 📦 4단계: 사용자 인터페이스/CLI 및 도구화
- [ ] CLI(Command Line Interface) 도구화
- [ ] 설정파일(`config.json`)에서 클래스 매핑/출력 디렉토리 등 지정
- [ ] 로그/에러 처리/예외 처리

### 🧪 5단계: 테스트/검증 및 개선
- [ ] 다양한 스크린샷 테스트
- [ ] 검출 정확도 평가 (precision/recall)
- [ ] 오탐/누락된 요소에 대한 보완 (후처리 logic 추가)

## 📈 데이터셋 확장/라벨링 가이드

### 1. 데이터셋 확장 전략
- **클래스별 최소 20~30개 이상 라벨 확보**: 감지 성능이 급격히 향상됩니다.
- **다양한 UI/화면/상황에서 스크린샷 수집**: 실제 사용 환경과 유사한 데이터 확보
- **각 클래스별로 골고루 라벨링**: 특정 클래스에만 치우치지 않게 주의
- **라벨링 파일명은 이미지와 동일하게** (예: workScr4.png → workScr4.txt)

### 2. 라벨링 툴 추천
- **[LabelImg](https://github.com/tzutalin/labelImg)** (YOLO 포맷 지원, 무료/오픈소스)
- **[Roboflow](https://roboflow.com/)** (웹 기반, 증강/변환/라벨링 지원)
- **[makesense.ai](https://www.makesense.ai/)** (설치 없이 웹에서 바로 라벨링)

#### LabelImg 사용법 요약
1. 설치: `pip install labelImg`
2. 실행: `labelImg`
3. YOLO 포맷 선택 후 이미지 폴더 열기
4. 객체 박스 지정 → 클래스명 입력 → 저장

### 3. 데이터 증강 자동화 예시
```python
from PIL import Image, ImageEnhance
import os
import random

def augment_image(image_path, output_dir, num_aug=5):
    img = Image.open(image_path)
    base = os.path.splitext(os.path.basename(image_path))[0]
    for i in range(num_aug):
        aug = img.copy()
        # 밝기/대비/색상/회전/좌우반전 등 랜덤 적용
        if random.random() < 0.5:
            aug = aug.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            aug = aug.rotate(random.randint(-10, 10))
        enhancer = ImageEnhance.Brightness(aug)
        aug = enhancer.enhance(random.uniform(0.7, 1.3))
        enhancer = ImageEnhance.Contrast(aug)
        aug = enhancer.enhance(random.uniform(0.7, 1.3))
        aug.save(os.path.join(output_dir, f"{base}_aug{i}.png"))

# 사용 예시
# augment_image('yolo/datasets/screenshots/start/images/workScr4.png', 'yolo/datasets/screenshots/start/images')
```
- **라벨 파일도 동일하게 복사/이름변경 필요** (증강 이미지와 동일한 이름)
- YOLO 라벨은 이미지 변환(좌우반전 등)에 맞게 좌표도 변환 필요 (LabelImg/Roboflow에서 자동 지원)

### 4. 데이터셋 구조 예시
```
yolo/datasets/screenshots/start/
├── images/
│   ├── workScr4.png
│   ├── ...
├── labels/
│   ├── workScr4.txt
│   ├── classes.txt
```

### 5. 데이터셋 품질 체크
- `python analyze_labels.py` 실행 시 클래스별 라벨 개수, 오류 자동 분석
- 라벨이 없는 클래스가 있으면 추가 라벨링 권장

### 6. 라벨링/증강 후 학습
- `python train.py`로 재학습
- `python improved_detector.py`로 감지 성능 확인

---

(이외에도 데이터 증강 자동화 스크립트, 라벨링 툴 사용법, 라벨 포맷 변환 등 추가 지원이 필요하면 언제든 요청해주세요!)

## 🔧 문제 해결

### 일반적인 문제들

1. **모델이 아무것도 감지하지 못하는 경우**
   - `improved_detector.py` 사용 (다중 임계값 탐지)
   - 신뢰도 임계값을 낮춰보세요 (0.05 ~ 0.1)

2. **모델 로딩 오류**
   - `model_diagnosis.py`로 모델 상태 확인
   - train4 모델 사용 권장

3. **경로 오류**
   - 절대 경로 사용 또는 상대 경로 확인
   - 파일 존재 여부 확인

### 성능 최적화 팁

1. **더 나은 탐지를 위해**:
   - `improved_detector.py` 사용
   - 다양한 임계값으로 테스트
   - 시각화 결과 확인

2. **모델 개선을 위해**:
   - 더 많은 학습 데이터 수집
   - 데이터 증강(augmentation) 적용
   - 하이퍼파라미터 튜닝

## 📝 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 🤝 기여하기

버그 리포트, 기능 요청, 풀 리퀘스트를 환영합니다!

---

**마지막 업데이트**: 2025년 1월
**버전**: 2.0 (개선된 탐지기 포함)
