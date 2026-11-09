import os
import string
from pypdf import PdfWriter, PdfReader
from pypdf.errors import PyPdfError


def add_bookmarks_in_range(outline, writer, start, end, reader, parent=None):
    """
    递归处理目录：如果目录指向的页码在当前拆分范围内，则修正页码并添加到新文件中。
    """
    last_added_item = None
    for item in outline:
        # 如果是列表，说明是子目录（层级结构）
        if isinstance(item, list):
            if last_added_item: # 只有当父级目录在范围内时才添加子目录
                add_bookmarks_in_range(item, writer, start, end, reader, parent=last_added_item)
            continue
        
        # 获取该目录项指向的原始页码
        try:
            page_no = reader.get_destination_page_number(item)
            # 如果原页码在当前拆分的 [start, end) 范围内
            if start <= page_no < end:
                # 计算在新 PDF 中的相对偏移页码
                new_page_no = page_no - start
                # 将书签添加到新文件，并保存引用供递归使用
                last_added_item = writer.add_outline_item(item.title, new_page_no, parent=parent)
        # 捕获 PDF 解析错误或参数类型/取值错误
        except (PyPdfError, ValueError, TypeError) as e:
            print(f"pdf 目录处理出错：{e}")
            continue
        

def split_large_pdf(input_path, step=20):
    """
    如果文件超过 10MB，按 step 步长拆分，并保留对应范围的目录。
    """
    file_size = os.path.getsize(input_path)
    limit_size = 10 * 1024 * 1024  # 10MB
    
    if file_size <= limit_size:
        print(f"无需拆分: {input_path} ({file_size/1024/1024:.2f}MB)")
        return

    try:
        print(f"检测到大文件，开始拆分并迁移目录: {input_path}")
        reader = PdfReader(input_path)
        total_pages = len(reader.pages)
        file_prefix = os.path.splitext(input_path)[0]
        alphabet = string.ascii_lowercase
        
        full_outline = reader.outline

        for i, start_page in enumerate(range(0, total_pages, step)):
            writer = PdfWriter()
            end_page = min(start_page + step, total_pages)
            
            writer.append(reader, pages=(start_page, end_page))
            add_bookmarks_in_range(full_outline, writer, start_page, end_page, reader)

            suffix = alphabet[i] if i < 26 else str(i)
            output_path = f"{file_prefix}_PART{suffix}.pdf"
            
            with open(output_path, "wb") as f:
                writer.write(f)
            print(f"  -> 已生成: {os.path.basename(output_path)}")
            
    except (OSError, PyPdfError) as e:  # 替换为具体异常组
        print(f"拆分过程中出错（文件读写或PDF损坏）: {e}")


def merge_all_pdfs(root_dir, output_filename="merged_output.pdf", split_pdf: bool = False):
    """
    遍历 root_dir 及其子目录，找到所有 .pdf 文件并合并。
    """
    merger = PdfWriter()
    pdf_files = []

    # 1. 递归遍历目录寻找 pdf
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(".pdf"):
                # 记录完整路径
                full_path = os.path.join(root, file)
                pdf_files.append(full_path)

    # 2. 对文件路径进行排序（确保合并顺序可控）
    pdf_files.sort()

    if not pdf_files:
        print("未在目录下找到任何 PDF 文件。")
        return

    # 3. 执行合并
    print(f"正在合并 {len(pdf_files)} 个文件...")
    try:
        for pdf in pdf_files:
            bookmark_name = os.path.basename(pdf)[:-4]
            merger.append(pdf, outline_item=bookmark_name)
        
        with open(output_filename, "wb") as output_file:
            merger.write(output_file)
        
        merger.close()
        print(f"合并成功！输出文件为: {os.path.abspath(output_filename)}")

        if split_pdf:
            split_large_pdf(output_filename)
            
    except (OSError, PyPdfError) as e:  # 精准捕获系统IO错误和PDF损坏错误
        print(f"文件处理或合并过程中出错: {e}")
