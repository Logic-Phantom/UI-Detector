# YOLOv5 기반 UI 리버스엔지니어링 →  .clx 변환 프로젝트 계획

## ✅ 전체 구성 및 단계별 할 일 (마일스톤 기반)

---

### 🧱 1단계: YOLOv5 모델 환경 구성 및 테스트
- [x] YOLOv5 모델 로딩 (torch hub 사용)
- [x] 이미지 입력 및 객체 감지 확인
- [ ] UI 요소 클래스 정의 (예: `button`, `input`, `textbox` 등)
- [ ] 커스텀 데이터셋(스크린샷 기반) 만들기 및 YOLOv5 재학습 (필요 시)
- [ ] 감지 결과를 JSON 또는 구조화된 데이터로 변환

> 🧠 **산출물**: 감지된 UI 요소 리스트(JSON)

---

### 🧩 2단계: 감지 결과 → CLEOPATRA .clx 변환기
- [x] CLEOPATRA .clx XML 구조 파악
- [x] UI 요소와 `<cl:*>` 매핑 룰 정의
- [ ] 요소 위치(x, y, width, height) → `<cl:xylayoutdata>` 생성
- [ ] 각 UI 요소 유형별 `<cl:button>`, `<cl:inputbox>` 등 생성 코드 작성
- [ ] 스타일/속성 자동 부여 로직 (필요 시)

> 🧠 **산출물**: `.clx` 형태의 XML 파일

---

### ⚙️ 3단계: 전체 파이프라인 통합
- [ ] 이미지 → YOLOv5 → 감지결과(JSON) → `.clx` 변환까지 자동화
- [ ] 디렉터리 감시 기능 (예: `screenshots/` 폴더에 이미지 생기면 자동 변환)
- [ ] 변환 결과 미리보기 (선택사항: HTML or CLEOPATRA로 열기)

---

### 📦 4단계: 사용자 인터페이스/CLI 및 도구화
- [ ] CLI(Command Line Interface) 도구화
- [ ] 설정파일(`config.json`)에서 클래스 매핑/출력 디렉토리 등 지정
- [ ] 로그/에러 처리/예외 처리

```bash
python run_converter.py --input test.png --output out.clx
```

---

### 🧪 5단계: 테스트/검증 및 개선
- [ ] 다양한 스크린샷 테스트
- [ ] 검출 정확도 평가 (precision/recall)
- [ ] 오탐/누락된 요소에 대한 보완 (후처리 logic 추가)

---

### 🌐 6단계: 배포 및 문서화
- [ ] `requirements.txt`, `README.md`, `사용자 가이드` 문서화
- [ ] `GitHub`에 오픈소스화 및 예제 포함
- [ ] 필요시 Web GUI 도구로 확장

---

## 📁 폴더 구조 예시

```
ui-detector-project/
├── models/                # YOLOv5 모델 weights
├── screenshots/           # 분석할 이미지
├── output/                # 생성된 .clx 파일
├── converter/             # 감지결과 → .clx 로 변환하는 코드
│   ├── clx_writer.py
│   └── tag_mapper.py
├── yolo/                  # YOLOv5 추론 코드
│   └── detector.py
├── run_converter.py       # 메인 파이프라인 스크립트
├── requirements.txt
└── README.md
```
