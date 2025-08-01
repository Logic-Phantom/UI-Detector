# Detection Scripts

이 디렉토리는 YOLO 모델을 사용한 UI 요소 감지 스크립트들을 포함합니다.

## 파일 목록

### improved_detector.py
- **개선된 감지기**: 향상된 YOLO 감지 기능
- **특징**: 더 정확한 감지, 다양한 출력 형식 지원
- **사용법**: `python improved_detector.py`

### detectJson.py
- **기본 JSON 감지**: YOLO 감지 결과를 JSON으로 저장
- **특징**: 간단한 JSON 출력 형식
- **사용법**: `python detectJson.py`

### detectJsonDetail.py
- **상세 JSON 감지**: 더 상세한 감지 정보를 JSON으로 저장
- **특징**: 바운딩 박스, 신뢰도, 클래스 정보 등 상세 정보 포함
- **사용법**: `python detectJsonDetail.py`

### detectJsonTree.py
- **트리 구조 JSON**: 계층적 구조로 감지 결과를 JSON으로 저장
- **특징**: 부모-자식 관계를 반영한 트리 구조
- **사용법**: `python detectJsonTree.py`

### viewDetector.py
- **감지 결과 시각화**: 감지 결과를 시각적으로 확인
- **특징**: 바운딩 박스, 라벨, 신뢰도 표시
- **사용법**: `python viewDetector.py`

## 주요 기능

1. **YOLO 모델 로드**: 사전 훈련된 모델 사용
2. **이미지 처리**: 다양한 이미지 형식 지원
3. **감지 결과 저장**: JSON, PNG 등 다양한 형식으로 저장
4. **시각화**: 감지 결과를 시각적으로 확인

## 출력 형식

- **JSON 파일**: `json/YYYY-MM-DD/` 디렉토리에 저장
- **PNG 파일**: 감지 결과 시각화
- **텍스트 파일**: 감지 결과 요약 