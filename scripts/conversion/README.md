# Conversion Scripts

이 디렉토리는 YOLO 감지 결과를 다양한 형식으로 변환하는 스크립트들을 포함합니다.

## 파일 목록

### yolo_to_figma_json.py
- **기본 버전**: YOLO 감지 결과를 Figma JSON 형식으로 변환
- **특징**: 계층 구조 포함, 기본적인 타입 매핑
- **사용법**: `python yolo_to_figma_json.py`

### yolo_to_figma_json_v2.py
- **v2 버전**: 개선된 JSON 변환 스크립트
- **특징**: 더 상세한 감지 결과 저장, PNG 시각화 포함
- **사용법**: `python yolo_to_figma_json_v2.py`

### yolo_to_figma_json_v3.py
- **v3 버전**: 최신 JSON 변환 스크립트
- **특징**: 복잡한 계층 구조, Java 코드 호환 타입 매핑
- **사용법**: `python yolo_to_figma_json_v3.py`

## 주요 기능

1. **타입 매핑**: YOLO 클래스명을 Figma 타입으로 변환
2. **계층 구조**: 감지된 요소들을 부모-자식 관계로 정리
3. **컴포넌트 타입**: Java 코드의 `cl:` 태그와 호환되는 구조
4. **라디오 버튼 그룹**: 다중 라디오 버튼 감지 및 처리

## 출력 형식

- **JSON 파일**: `figma_json/YYYY-MM-DD/` 디렉토리에 저장
- **PNG 파일**: 감지 결과 시각화 (v2, v3 버전) 