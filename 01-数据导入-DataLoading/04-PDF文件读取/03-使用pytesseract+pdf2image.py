# 扫描图片型 PDF，建议用 pytesseract + pdf2image  
# sudo apt-get install tesseract-ocr
# sudo apt-get install tesseract-ocr-chi-sim
from pathlib import Path

import pdf2image
import pytesseract
import os

# 创建 output 目录
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)

# 将 PDF 转换为图片并保存
current_dir = Path(__file__).resolve().parent.parent.parent
file_path = f"{current_dir}/90-文档-Data/黑悟空/黑神话悟空.pdf"
images = pdf2image.convert_from_path(file_path)
for i, image in enumerate(images):
    image.save(f'{output_dir}/page_{i+1}.png')

# 使用 pytesseract 提取文本
for i, image in enumerate(images):
    text = pytesseract.image_to_string(image, lang='chi_sim')
    print(f"第 {i+1} 页文本:")
    print(text)
    print("\n") 