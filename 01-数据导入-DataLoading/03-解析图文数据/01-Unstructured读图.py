from langchain_community.document_loaders import UnstructuredImageLoader
image_path = "/Users/niumingjie.nmj/github/rag-in-action/90-文档-Data/黑悟空/黑悟空英文.jpg"
loader = UnstructuredImageLoader(image_path)

data = loader.load()
print(data)
