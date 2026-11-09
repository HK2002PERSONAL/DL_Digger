#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
merge_pdf_horizontal.py

通用脚本：将两个 PDF 按页横向拼接（第 i 页左右并排合并为一页）。
- 仅支持恰好 2 个 PDF 输入
- 若两个 PDF 页数不同，则发出警告并仅合并较少页数
- 每页输出尺寸：宽度 = 两页宽度之和，高度 = 两页高度的最大值（各自垂直居中）

用法:
    python merge_pdf_horizontal.py <pdf1> <pdf2> [-o OUTPUT]

示例:
    python merge_pdf_horizontal.py pdf_testbench/gcn_test_allneut.pdf pdf_testbench/gcn_test_lncap.pdf -o pdf_testbench/merged.pdf
"""
import argparse
import os
import sys
import warnings

from pypdf import PdfReader, PdfWriter, Transformation, PageObject


def merge_horizontal(pdf_paths, output_path):
    """将两个 PDF 的每一页横向拼接并写入 output_path。"""
    for p in pdf_paths:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"文件不存在: {p}")

    reader1 = PdfReader(pdf_paths[0])
    reader2 = PdfReader(pdf_paths[1])

    n1, n2 = len(reader1.pages), len(reader2.pages)
    if n1 != n2:
        warnings.warn(
            f"页数不一致: {pdf_paths[0]} 有 {n1} 页, {pdf_paths[1]} 有 {n2} 页，将仅合并前 {min(n1, n2)} 页",
            UserWarning,
        )

    writer = PdfWriter()

    for i in range(min(n1, n2)):
        page1 = reader1.pages[i]
        page2 = reader2.pages[i]

        w1 = float(page1.mediabox.width)
        h1 = float(page1.mediabox.height)
        w2 = float(page2.mediabox.width)
        h2 = float(page2.mediabox.height)

        new_width = w1 + w2
        new_height = max(h1, h2)

        blank = PageObject.create_blank_page(width=new_width, height=new_height)

        # 第一个 PDF 的页放在左侧，垂直居中
        ty1 = (new_height - h1) / 2
        blank.merge_transformed_page(page1, Transformation().translate(tx=0, ty=ty1))

        # 第二个 PDF 的页放在右侧，垂直居中
        ty2 = (new_height - h2) / 2
        blank.merge_transformed_page(page2, Transformation().translate(tx=w1, ty=ty2))

        writer.add_page(blank)

    with open(output_path, "wb") as f:
        writer.write(f)

    return min(n1, n2)


def main():
    parser = argparse.ArgumentParser(
        description="将两个 PDF 的每一页横向拼接（页数必须相同）"
    )
    parser.add_argument("--pdf1", help="第一个 PDF 文件（放在左侧）")
    parser.add_argument("--pdf2", help="第二个 PDF 文件（放在右侧）")
    parser.add_argument(
        "-o",
        "--output",
        default="merged_horizontal.pdf",
        help="输出 PDF 路径（默认: merged_horizontal.pdf）",
    )
    args = parser.parse_args()

    try:
        n = merge_horizontal([args.pdf1, args.pdf2], args.output)
    except FileNotFoundError as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"拼接完成: 共 {n} 页，输出文件: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()