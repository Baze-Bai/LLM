# 导入所需库
import fitz  # PyMuPDF库，用于处理PDF文件
from PIL import Image  # Pillow图像处理库
import pytesseract  # Tesseract OCR的Python封装
import io  # 处理二进制数据流
import os  # 处理文件路径
import re
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

def extract_text_from_pdf(directory):
    """Extract text from images in a PDF file (including embedded images and full scan pages)
    
    parameters:
        file_path (str): PDF file path to be processed
        
    returns:
        str: Extracted text content from images in the PDF file
    """
    pdf_files = []  # Initialize a list to store all PDF files

    for root, dirs, files in os.walk(directory):
        for file in files:
            # Select PDF files only
            if file.lower().endswith(".pdf"):
                # Append the full file path to the list
                pdf_files.append(os.path.join(root, file))

    full_text = ""  # Initialize a variable to store the full text content

    # Traverse each PDF file in the directory
    for file_path in pdf_files:
        doc = fitz.open(file_path)
        text = ""  # Initialize a variable to store the text content of the current PDF file
        # Traverse each page in the PDF file
        for page_num in range(len(doc)):
            
            # 加载当前页的Page对象
            page = doc.load_page(page_num)

            text = page.get_text()

            full_text += text + "\n"  # Append the text content of the current PDF file to the full text

    return full_text  # Return the full text content


def text_chunk(content):

    pattern = '^[\uFFFD■•◆!]\\s*'
    content = re.sub(pattern, '', content, flags=re.MULTILINE)

    final_chunks = []
    section_pattern = re.compile(r'^(\d+\.\d+)\s*:\s*(.*)$')

    lines = content.split('\n')

    sections = {}
    current_section_id = None
    token_len = 0
    tem_chunk = []
    line_len = 0

    for line in lines:
        if not line.strip():
            print(f"Empty line, skipping: {line}")
            # 空行直接跳过
            continue

        # 尝试匹配是否是新的小节标题
        match = section_pattern.match(line)

        if match:
            if current_section_id is not None:
                sections[current_section_id]["content"].append("".join(tem_chunk))
                tem_chunk = []
                token_len = 0

            print(f"匹配到小节标题: {match.group(1)} - {match.group(2)}")

            # 如果匹配到，则开启一个新的小节
            section_id = match.group(1)    # 例如 "1.1"
            section_title = match.group(2) # 例如 "Introduction to the course"
            if section_id not in sections:    
                current_section_id = section_id
                # 初始化存储
                sections[current_section_id] = {
                    "title": section_title,
                    "content": []
                } # current_section_id: {"title": section_title, "content": []}

            else:
                current_section_id = section_id + "a"
                # 初始化存储
                sections[current_section_id] = {
                    "title": section_title,
                    "content": []
                } # current_section_id: {"title": section_title, "content": []}


            
        elif current_section_id is not None:
            if line.startswith("ECE4710") or line.startswith("ECE5720") or re.fullmatch(r'\d+-\d+', line) or line.startswith('Lecture notes') or line.startswith(' 2011–2019')or line.startswith('2011–2019'):
                continue
            # 如果不是小节标题行，则将其视为当前小节的正文
            if current_section_id is not None:
                token_ids = tokenizer.encode(line, add_special_tokens=False)
                line_len = len(token_ids)
                if token_len + line_len > 256:
                    sections[current_section_id]["content"].append("".join(tem_chunk))
                    token_len = line_len
                    tem_chunk = []
                    tem_chunk.append(line)
                
                else:
                    tem_chunk.append(line)
                    token_len += line_len

    for sec_id, sec_data in sections.items():
        for chunk in sec_data["content"]:
            final_chunks.append(chunk)
        # 打印查看结果
    for sec_id, sec_data in sections.items():
        print(f"小节编号: {sec_id}")
        print(f"小节标题: {sec_data['title']}")
        print("正文内容:")
        print(sec_data["content"])
        print("------")

    return sections, final_chunks


