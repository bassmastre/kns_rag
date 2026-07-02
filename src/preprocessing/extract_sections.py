"""NUREG-1431 PDF에서 섹션 텍스트를 추출/정제.

- REVIEWER'S NOTE 블록 정규식 제거
- 원문 좌표(page, char offset) 보존 -> span_index에서 사용
"""

"""
NUREG-1431 PDF에서 특정 섹션 페이지 범위만 잘라내는 스크립트.

사용 전 준비:
  1) PDF 뷰어(Acrobat, Chrome 등)로 원본 PDF를 열어서
     각 섹션이 시작/끝나는 "실제 PDF 페이지 번호"를 확인한다.
     (목차에 적힌 "3.4-1" 같은 섹션 페이지 번호가 아니라,
      뷰어 하단/상단에 표시되는 실제 페이지 인덱스여야 함)
  2) 아래 SECTION_RANGES 딕셔너리에 1-indexed, inclusive 기준으로 채운다.

설치:
    pip install pypdf --break-system-packages

실행:
    python extract_sections.py --input NUREG-1431_Vol1.pdf --output-dir ./sections
"""

import argparse
from pathlib import Path
from pypdf import PdfReader, PdfWriter

# TODO: 실제 PDF 페이지 번호로 채울 것 (1-indexed, inclusive)
# 예시 값이며 반드시 실제 확인 후 수정해야 함
SECTION_RANGES = {
    "3.4.1_DNB_Limits":        (0, 0),   # (start_page, end_page)
    "3.4.10_Pressurizer_SV":   (0, 0),
    "3.4.13_RCS_LEAKAGE":      (0, 0),
    "3.4.16_RCS_Specific_Act": (0, 0),
}


def extract_section(reader: PdfReader, start: int, end: int) -> PdfWriter:
    """1-indexed, inclusive page range를 새 PdfWriter로 추출."""
    writer = PdfWriter()
    for page_num in range(start, end + 1):
        writer.add_page(reader.pages[page_num - 1])  # pypdf는 0-indexed
    return writer


def main():
    parser = argparse.ArgumentParser(description="Extract section page ranges from a PDF")
    parser.add_argument("--input", required=True, help="원본 PDF 경로")
    parser.add_argument("--output-dir", required=True, help="잘라낸 PDF 저장 폴더")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reader = PdfReader(str(input_path))
    total_pages = len(reader.pages)
    print(f"원본 PDF 총 페이지 수: {total_pages}")

    for name, (start, end) in SECTION_RANGES.items():
        if start == 0 or end == 0:
            print(f"[SKIP] {name}: 페이지 범위가 아직 설정되지 않음 (SECTION_RANGES 수정 필요)")
            continue
        if start > total_pages or end > total_pages:
            print(f"[ERROR] {name}: 지정된 범위({start}-{end})가 PDF 총 페이지({total_pages})를 초과")
            continue
        if start > end:
            print(f"[ERROR] {name}: start({start}) > end({end})")
            continue

        writer = extract_section(reader, start, end)
        out_path = output_dir / f"{name}.pdf"
        with open(out_path, "wb") as f:
            writer.write(f)
        print(f"[OK] {name}: page {start}-{end} ({end - start + 1}p) -> {out_path}")


if __name__ == "__main__":
    main()