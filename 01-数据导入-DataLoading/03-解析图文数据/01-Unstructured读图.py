from pathlib import Path

from langchain_community.document_loaders import UnstructuredImageLoader
current_dir = Path(__file__).resolve().parent.parent.parent
image_path = f"{current_dir}/90-文档-Data/黑悟空/黑悟空英文.jpg"
loader = UnstructuredImageLoader(image_path)

data = loader.load()
print(data)
