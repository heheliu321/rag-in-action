from langchain_community.document_loaders import TextLoader
print("=== TextLoader 加载结果 ===")
text_loader = TextLoader("/Users/niumingjie.nmj/github/rag-in-action//Users/niumingjie.nmj/github/rag-in-action/90-文档-Data/灭神纪/人物角色.json")
text_documents = text_loader.load()
print(text_documents)
