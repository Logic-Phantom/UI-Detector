'''
Created on 2025. 5. 13.

@author: LCM
'''
import json
import os
import xml.etree.ElementTree as ET

def convert_json_to_clx(json_path: str, clx_path: str):
    # JSON 파일 로드
    with open(json_path, 'r', encoding='utf-8') as f:
        detections = json.load(f)

    # 기본 XML 구조 생성
    html = ET.Element("{http://www.w3.org/1999/xhtml}html", {
        "xmlns:cl": "http://tomatosystem.co.kr/cleopatra",
        "xmlns:std": "http://tomatosystem.co.kr/cleopatra/studio",
        "std:sid": "html-8d422525",
        "version": "1.0.5560"
    })

    head = ET.SubElement(html, "head", {
        "std:sid": "head-42c3d630"
    })
    screen = ET.SubElement(head, "screen", {
        "std:sid": "screen-8efcf144",
        "id": "PC",
        "name": "PC",
        "width": "1654px",
        "height": "940px",
        "useCustomWidth": "false",
        "useCustomHeight": "false",
        "customHeight": "600",
        "customWidth": "800"
    })
    model = ET.SubElement(head, "cl:model", {
        "std:sid": "model-fb158fd8"
    })
    appspec = ET.SubElement(head, "cl:appspec")

    body = ET.SubElement(html, "body", {
        "std:sid": "body-1d18a402"
    })
    group = ET.SubElement(body, "cl:group", {
        "std:sid": "group-70147ed0"
    })

    # 기본 레이아웃 데이터
    group_layout = ET.SubElement(group, "cl:xylayoutdata", {
        "std:sid": "xyl-data-c4531db8",
        "top": "119px",
        "left": "237px",
        "width": "561px",
        "height": "284px",
        "horizontalAnchor": "LEFT",
        "verticalAnchor": "TOP"
    })

    # 감지된 객체를 .clx 파일에 추가
    for i, obj in enumerate(detections):
        class_name = obj['class']
        x = int(obj['x1'])
        y = int(obj['y1'])
        width = int(obj['x2'] - obj['x1'])
        height = int(obj['y2'] - obj['y1'])

        # 버튼이라면 cl:button 태그 추가
        if class_name.lower() == 'button':
            button = ET.SubElement(group, "cl:button", {
                "std:sid": f"button-{i}",
                "value": "Button"
            })
            button_layout = ET.SubElement(button, "cl:xylayoutdata", {
                "std:sid": f"xyl-data-{i}",
                "top": f"{y}px",
                "left": f"{x}px",
                "width": f"{width}px",
                "height": f"{height}px",
                "horizontalAnchor": "LEFT",
                "verticalAnchor": "TOP"
            })

        # 다른 UI 요소도 처리하려면 여기 추가

    # 그룹 끝
    xylayout = ET.SubElement(group, "cl:xylayout", {
        "std:sid": "xylayout-ab0621fb"
    })

    # 바디 끝
    xylayout_end = ET.SubElement(body, "cl:xylayout", {
        "std:sid": "xylayout-c676e8cb"
    })

    # studiosetting
    studiosetting = ET.SubElement(html, "std:studiosetting")
    hruler = ET.SubElement(studiosetting, "std:hruler")
    vruler = ET.SubElement(studiosetting, "std:vruler")

    # XML 저장
    tree = ET.ElementTree(html)
    os.makedirs(os.path.dirname(clx_path), exist_ok=True)
    tree.write(clx_path, encoding='utf-8', xml_declaration=True)

    print(f".clx file saved to: {clx_path}")
