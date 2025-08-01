# Utility Scripts

이 디렉토리는 프로젝트에서 사용하는 유틸리티 스크립트들을 포함합니다.

## 파일 목록

### quick_start.py
- **빠른 시작**: 프로젝트 초기 설정 및 기본 실행
- **특징**: 환경 설정, 모델 다운로드, 기본 감지 실행
- **사용법**: `python quick_start.py`
- **기능**: 
  - 필요한 패키지 설치 확인
  - YOLO 모델 다운로드
  - 기본 이미지 감지 실행

### osStr.py
- **OS 문자열 처리**: 운영체제별 경로 및 문자열 처리
- **특징**: Windows/Linux 호환 경로 처리
- **사용법**: 다른 스크립트에서 import하여 사용
- **기능**:
  - 경로 구분자 정규화
  - OS별 경로 처리
  - 문자열 인코딩 처리

## 주요 기능

1. **환경 설정**: 프로젝트 실행에 필요한 기본 설정
2. **경로 처리**: 운영체제별 경로 정규화
3. **문자열 처리**: 인코딩 및 문자열 정규화
4. **초기화**: 프로젝트 초기 설정 자동화

## 사용 방법

### 직접 실행
```bash
python quick_start.py
```

### 다른 스크립트에서 import
```python
from scripts.utils.osStr import normalize_path
```

## 의존성

- **quick_start.py**: ultralytics, opencv-python, numpy
- **osStr.py**: 표준 라이브러리만 사용 