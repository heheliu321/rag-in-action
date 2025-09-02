from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(
    file_path="C:/github/rag-in-action/90-文档-Data/黑悟空/黑神话悟空.csv",
    encoding="utf-8"  # 显式指定编码
)
documents = loader.load()
print(documents)

docs = loader.load()
print(f"文档数：{len(docs)}")  # 输出文档总数
print(docs[0])
