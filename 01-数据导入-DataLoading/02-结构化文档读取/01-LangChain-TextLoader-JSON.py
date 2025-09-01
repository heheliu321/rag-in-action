from pathlib import Path

from langchain_community.document_loaders import TextLoader
print("=== TextLoader 加载结果 ===")
current_dir = Path(__file__).resolve().parent.parent.parent
text_loader = TextLoader(f"{current_dir}/90-文档-Data/灭神纪/人物角色.json", encoding='utf-8')
text_documents = text_loader.load()
print(text_documents)
